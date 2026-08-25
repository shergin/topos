use smallvec::SmallVec;

use crate::op::Op;
use crate::{Element, Shape, Tensor};

use super::candidates::Candidate;
use super::pattern::Pattern;
use super::view::View;

/// A matched window-GEMM group: the `matmul` node computes
/// [`Tensorial::windowed_product`](crate::Tensorial::windowed_product)
/// directly from the source, and the im2col chain between them —
/// pads, unfolds, permute, reshape — is never materialized.
///
/// Matching is structural and provenance-blind: any recording of the
/// canonical im2col composition matches, whichever facade (or hand)
/// wrote it. A keep-set node inside the chain is a barrier. The
/// pattern is stored only on forward-only plans; engine-backward
/// stays unfused so the reverse scan reads what the recording named.
#[derive(Debug, Clone)]
pub(crate) struct WindowProduct {
    /// The rank-4 `[batch, channels, height, width]` source.
    pub(crate) source: usize,
    /// The GEMM-shaped `[columns, filters]` kernel operand.
    pub(crate) kernel: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
}

impl WindowProduct {
    /// Returns the slots the fused call reads past the root's operand
    /// links; liveness must keep them alive until the call.
    pub(crate) fn reads(&self) -> [usize; 2] {
        [self.source, self.kernel]
    }

    /// Computes the fused call over the already-evaluated `values`:
    /// one windowed product from the source and kernel, the im2col
    /// chain between them never materialized.
    pub(crate) fn apply<E: Element>(&self, values: &[Tensor<E>]) -> Tensor<E> {
        values[self.source].windowed_product(
            &values[self.kernel],
            self.kernel_height,
            self.kernel_width,
            self.stride,
            self.padding,
        )
    }
}

/// Matches the canonical im2col chain feeding a `matmul` rooted at
/// `index` — `reshape(permute(unfold(unfold(pad(pad(x)?)?))))` with
/// the conv parameterization — and returns the candidate group.
/// Interiors must pass [`View::interior_ok`]: wanted, outside the
/// keep-set, and consumed exactly once inside the closure.
pub(crate) fn match_at<E: Element>(index: usize, view: &View<Tensor<E>>) -> Option<Candidate> {
    let Some(Op::MatMul(_)) = view.op(index) else {
        return None;
    };
    let lhs = view.operand(index, 0);
    let kernel = view.operand(index, 1);

    let Some(Op::Reshape(reshape)) = view.op(lhs) else {
        return None;
    };
    if !view.interior_ok(lhs) || reshape.shape.rank() != 2 {
        return None;
    }
    let permuted = view.sole_operand(lhs);
    let Some(Op::Permute(permute)) = view.op(permuted) else {
        return None;
    };
    if !view.interior_ok(permuted) || permute.order.as_slice() != [0, 2, 4, 1, 3, 5] {
        return None;
    }
    let windows_w = view.sole_operand(permuted);
    let Some(Op::Unfold(unfold_w)) = view.op(windows_w) else {
        return None;
    };
    if !view.interior_ok(windows_w) || unfold_w.axis != 4 || unfold_w.dilation != 1 {
        return None;
    }
    let windows_h = view.sole_operand(windows_w);
    let Some(Op::Unfold(unfold_h)) = view.op(windows_h) else {
        return None;
    };
    if !view.interior_ok(windows_h)
        || unfold_h.axis != 2
        || unfold_h.dilation != 1
        || unfold_h.step != unfold_w.step
    {
        return None;
    }
    let mut chain: SmallVec<[usize; 8]> =
        SmallVec::from_slice(&[lhs, permuted, windows_w, windows_h]);
    let mut source = view.sole_operand(windows_h);
    let mut padding = 0;
    // Symmetric zero pads fold into the fused call; anything else
    // simply leaves the pad output as the (materialized) source.
    if let Some(Op::Pad(pad_w)) = view.op(source)
        && view.interior_ok(source)
        && pad_w.axis == 3
    {
        let below = view.sole_operand(source);
        if let Some(Op::Pad(pad_h)) = view.op(below) {
            let base = view.sole_operand(below);
            let base_axes = view.shape(base).axes();
            if view.interior_ok(below)
                && pad_h.axis == 2
                && base_axes.len() == 4
                && pad_h.start == pad_w.start
                && pad_h.full_extent == base_axes[2] + 2 * pad_h.start
                && pad_w.full_extent == base_axes[3] + 2 * pad_w.start
            {
                chain.push(source);
                chain.push(below);
                padding = pad_h.start;
                source = base;
            }
        }
    }
    let source_axes = view.shape(source).axes();
    if source_axes.len() != 4 {
        return None;
    }
    let (batch, channels) = (source_axes[0], source_axes[1]);
    let padded_height = source_axes[2] + 2 * padding;
    let padded_width = source_axes[3] + 2 * padding;
    let (kernel_height, kernel_width) = (unfold_h.size, unfold_w.size);
    let stride = unfold_h.step;
    let out_height = (padded_height - kernel_height) / stride + 1;
    let out_width = (padded_width - kernel_width) / stride + 1;
    let expected = Shape::new([
        batch * out_height * out_width,
        channels * kernel_height * kernel_width,
    ]);
    if reshape.shape != expected {
        return None;
    }
    Some(Candidate {
        pattern: Pattern::WindowProduct(WindowProduct {
            source,
            kernel,
            kernel_height,
            kernel_width,
            stride,
            padding,
        }),
        root: index,
        interiors: chain,
        named: SmallVec::new(),
    })
}

#[cfg(test)]
#[path = "tests/window_tests.rs"]
mod tests;
