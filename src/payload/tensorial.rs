use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{Elementary, Shape, Tensor};

/// The recordable vocabulary: the operations derivative rules — and
/// any payload-generic algorithm — are written against.
///
/// The trait has exactly two in-crate interpretations, and that pair
/// is its purpose. [`Tensor`] computes each operation over its
/// buffers, so the engine's backward scan runs the rules for real;
/// [`Trace`](crate::Trace) appends the corresponding node to a tape
/// and answers with a handle, so `differentiate` records the very
/// same rules as graph. One body of derivative knowledge, two
/// interpretations — the rules cannot tell which one they run under.
///
/// Membership is the recordable operation set and nothing else:
/// every method here corresponds to something a tape can record.
/// What a rule cannot call is deliberately absent — `max_along` (the
/// stability shift inside the fused log-domain forwards, not a
/// recorded operation), the `counted` constructor (a rule has no
/// tape to mint literals on; `zero_like` and `one_like` cover the
/// identities rules need), and the fused executors
/// (`windowed_product` and friends are plan-tier kernel faces).
/// Those live as inherent methods on [`Tensor`] alone, which is what
/// lets both interpretations implement this trait without a single
/// panicking member.
pub trait Tensorial:
    Clone
    + Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Returns the shape of this value: its extent along every axis.
    fn shape(&self) -> Shape;

    /// Returns a zero shaped like `self`, seeding gradient
    /// accumulators.
    fn zero_like(&self) -> Self;

    /// Returns a one shaped like `self`, seeding the output gradient.
    fn one_like(&self) -> Self;

    /// Returns `e` raised elementwise to `self`.
    fn exp(&self) -> Self;

    /// Returns the elementwise natural logarithm of `self`.
    fn ln(&self) -> Self;

    /// Returns the elementwise square root of `self`.
    fn sqrt(&self) -> Self;

    /// Returns the elementwise hyperbolic tangent of `self`.
    fn tanh(&self) -> Self;

    /// Returns `self` raised elementwise to the power of `exponent`.
    fn powf(&self, exponent: Self) -> Self;

    /// Returns the elementwise maximum of `self` and `other`.
    fn maximum(&self, other: &Self) -> Self;

    /// Returns the elementwise 0/1 indicator of `self >= threshold`:
    /// the Heaviside step, ties answering one. It carries the
    /// derivative of the `maximum` family.
    fn step(&self, threshold: &Self) -> Self;

    /// Returns the matrix product of `self` and `rhs`; ranks above
    /// two multiply batched over identical leading axes.
    fn matmul(&self, rhs: &Self) -> Self;

    /// Returns `self` with its two axes swapped.
    fn transpose(&self) -> Self;

    /// Returns the sum of every value in `self`, shaped as a single
    /// value.
    fn sum(&self) -> Self;

    /// Returns `self` with `axis` reduced by summation.
    fn sum_along(&self, axis: usize) -> Self;

    /// Returns this value's single element spread across
    /// `reference`'s shape.
    fn broadcast_like(&self, reference: &Self) -> Self;

    /// Returns `self` repeated along `axis` to match `reference`'s
    /// shape.
    fn broadcast_along(&self, axis: usize, reference: &Self) -> Self;

    /// Returns `self` reinterpreted with `shape`, preserving logical
    /// row-major order.
    fn reshape(&self, shape: Shape) -> Self;

    /// Returns `self` with its axes reordered so that axis `i` of the
    /// result takes axis `order[i]` of `self`.
    fn permute(&self, order: &[usize]) -> Self;

    /// Returns the window of `len` elements from `start` along `axis`.
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self;

    /// Returns `self` placed into zeros whose `axis` has extent
    /// `full_extent`, at `start ..`: the adjoint of
    /// [`narrow`](Tensorial::narrow).
    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self;

    /// Returns the sliding windows of `self` along `axis`: the axis
    /// becomes a `(count, size)` pair.
    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self;

    /// Returns the `(count, size)` window pair at `axis`, `axis + 1`
    /// folded back onto an axis of `extent`: the adjoint of
    /// [`unfold`](Tensorial::unfold).
    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self;

    /// Returns the rows of `self` selected by the one-hot `selection`.
    fn gather(&self, selection: &Self) -> Self;

    /// Scatter-adds the rows of `self` into `rows` rows by
    /// `selection`'s indices: the adjoint of
    /// [`gather`](Tensorial::gather).
    fn scatter(&self, selection: &Self, rows: usize) -> Self;
}

/// Composes the unfused im2col formula — pad, two unfolds, permute,
/// and the patch reshape — through the plain tensor operations: the
/// bitwise reference the fused fast path is tested against.
pub fn composed_windowed_patches<Element: Elementary>(
    input: &Tensor<Element>,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
) -> Tensor<Element> {
    let shape = input.shape();
    let axes = shape.axes();
    let (batch, channels, height, width) = (axes[0], axes[1], axes[2], axes[3]);
    let mut padded = input.clone();
    if padding > 0 {
        padded = padded.pad(2, padding, height + 2 * padding);
        padded = padded.pad(3, padding, width + 2 * padding);
    }
    let windows = padded
        .unfold(2, kernel_height, stride, 1)
        .unfold(4, kernel_width, stride, 1);
    let windows_shape = windows.shape();
    let out_height = windows_shape.axes()[2];
    let out_width = windows_shape.axes()[4];
    windows.permute(&[0, 2, 4, 1, 3, 5]).reshape(Shape::new([
        batch * out_height * out_width,
        channels * kernel_height * kernel_width,
    ]))
}

/// Composes the recorded batch-normalization formula — mean by
/// `sum_along` over a counted divisor, centering, biased variance,
/// the epsilon-stabilized deviation, and the learned affine — in the
/// exact operation order the tape records: the bitwise reference the
/// fused fast path is graded against.
pub fn composed_batch_norm<Element: Elementary>(
    input: &Tensor<Element>,
    scale: &Tensor<Element>,
    shift: &Tensor<Element>,
    epsilon: &Tensor<Element>,
) -> (Tensor<Element>, Tensor<Element>, Tensor<Element>) {
    let shape = input.shape();
    let batch = shape.axes()[0];
    let reduced = shape.without_axis(0);
    let mean = input.sum_along(0) / Tensor::counted(reduced.clone(), batch);
    let centered = input.clone() - mean.broadcast_along(0, input);
    let variance =
        (centered.clone() * centered.clone()).sum_along(0) / Tensor::counted(reduced, batch);
    let deviation = (variance.clone() + epsilon.broadcast_like(&variance)).sqrt();
    let normalized = centered.clone() / deviation.broadcast_along(0, &centered);
    let output =
        normalized * scale.broadcast_along(0, &centered) + shift.broadcast_along(0, &centered);
    (output, mean, variance)
}

/// Composes the recorded max-pool formula — two square unfolds, the
/// lane permute and merging reshape, a left-associated `maximum`
/// fold in row-major lane order, and the trailing squeeze — in the
/// exact operation order the tape records: the bitwise reference the
/// fused direct walk is graded against.
pub fn composed_max_pool<Element: Elementary>(
    input: &Tensor<Element>,
    size: usize,
    stride: usize,
) -> Tensor<Element> {
    let shape = input.shape();
    let axes = shape.axes();
    let (batch, channels, height, width) = (axes[0], axes[1], axes[2], axes[3]);
    let out_height = (height - size) / stride + 1;
    let out_width = (width - size) / stride + 1;
    let lanes = input
        .unfold(2, size, stride, 1)
        .unfold(4, size, stride, 1)
        .permute(&[0, 1, 2, 4, 3, 5])
        .reshape(Shape::new([
            batch,
            channels,
            out_height,
            out_width,
            size * size,
        ]));
    let mut largest = lanes.narrow(4, 0, 1);
    for lane in 1..size * size {
        largest = largest.maximum(&lanes.narrow(4, lane, 1));
    }
    largest.reshape(Shape::new([batch, channels, out_height, out_width]))
}
