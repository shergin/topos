use crate::{Element, Parameters, Shape, Symbol, Tensor};

use super::Optimizer;
use super::optimizer::assert_single_value;

/// The Newton-Schulz iteration Muon orthogonalizes its updates with:
/// a fixed number of matrix products that drive a matrix's singular
/// values toward one while keeping its singular vectors, so the result
/// approximates the polar factor `U V^T` of the input.
///
/// Each iteration maps `X` to `a X + (b A + c A A) X` with the Gram
/// matrix `A = X X^T`. The input is first divided by its Frobenius
/// norm plus `epsilon`, which bounds its spectral norm by one, where
/// the polynomial converges; a tall matrix iterates on its transpose
/// so the Gram matrix is the smaller square, and the result is
/// transposed back. The polynomial is the caller's, written as
/// single-value payloads: the reference implementation's tuned
/// quintic is `(3.4445, -4.7750, 2.0315)` over five iterations, which
/// trades an exact orthogonalization for a fast one (its output's
/// singular values land in roughly `[0.7, 1.1]` rather than at one),
/// and the classical cubic `(1.5, -0.5, 0)` converges to the polar
/// factor given enough iterations. Every operation is public payload
/// arithmetic, so a hand-rolled loop of the same products computes
/// the same bits.
#[derive(Debug, Clone)]
pub struct NewtonSchulz<E> {
    coefficients: [Tensor<E>; 3],
    steps: usize,
    epsilon: Tensor<E>,
}

impl<E: Element> NewtonSchulz<E> {
    /// Creates the iteration with the polynomial `coefficients`
    /// `[a, b, c]`, the number of iterations `steps`, and the
    /// normalization stabilizer `epsilon` added to the Frobenius norm.
    ///
    /// Zero steps return the normalized input unchanged.
    ///
    /// # Panics
    /// Panics if any coefficient or `epsilon` holds more than one
    /// value.
    pub fn new(coefficients: [Tensor<E>; 3], steps: usize, epsilon: Tensor<E>) -> Self {
        for (coefficient, name) in coefficients.iter().zip(["a", "b", "c"]) {
            assert_single_value(coefficient, name);
        }
        assert_single_value(&epsilon, "epsilon");
        Self {
            coefficients,
            steps,
            epsilon,
        }
    }

    /// Returns the approximate polar factor of the rank-2 `matrix`: the
    /// nearest matrix with orthonormal rows or columns, up to the
    /// polynomial's accuracy.
    ///
    /// # Panics
    /// Panics if `matrix` is not rank 2.
    pub fn orthogonalize(&self, matrix: &Tensor<E>) -> Tensor<E> {
        let shape = matrix.shape();
        assert_eq!(shape.rank(), 2, "orthogonalize requires a rank-2 matrix");
        let tall = shape.axes()[0] > shape.axes()[1];
        let oriented = if tall {
            matrix.transpose()
        } else {
            matrix.clone()
        };

        let norm = (oriented.clone() * oriented.clone()).sum().sqrt() + self.epsilon.clone();
        let mut x = oriented.clone() / norm.broadcast_like(&oriented);
        let [a, b, c] = &self.coefficients;
        for _ in 0..self.steps {
            let gram = x.matmul(&x.transpose());
            let blend = gram.clone() * b.broadcast_like(&gram)
                + gram.matmul(&gram) * c.broadcast_like(&gram);
            x = x.clone() * a.broadcast_like(&x) + blend.matmul(&x);
        }
        if tall { x.transpose() } else { x }
    }
}

/// The Muon optimizer: momentum SGD whose matrix updates are
/// orthogonalized (Jordan et al., 2024), so every singular direction
/// of a weight matrix moves by about the same amount regardless of how
/// the gradient's energy was spread across them.
///
/// Each step refreshes the momentum table `B = momentum * B + G` and
/// takes the Nesterov direction `G + momentum * B`, the reference
/// implementation's default and the fixed choice here. A parameter the
/// policy selects is reshaped to the matrix `[axes[0], rest]` (a
/// weight matrix as itself, a convolution filter bank as one row per
/// output channel), orthogonalized through the [`NewtonSchulz`]
/// iteration, scaled by `sqrt(max(1, rows / columns))` so a tall
/// matrix's per-row step keeps the size a square one gets (an integer
/// ratio of the shape, not a float constant), and stepped by the
/// learning rate. Every other parameter steps along the plain
/// direction: momentum SGD at the same rate.
///
/// The default policy is structural, like [`AdamW`](super::AdamW)'s
/// decay: orthogonalize parameters of rank two and above, never
/// rank-one biases and gains. The reference practice also keeps
/// embeddings and output heads, rank two though they are, out of Muon
/// and gives them AdamW; that is a caller predicate through
/// [`Muon::step_where`], and the pairing itself is caller-side table
/// algebra: step both optimizers at their own rates and select each
/// parameter's next payload by symbol through
/// [`Parameters::step_each`].
///
/// Hyperparameters are single-value payloads the caller writes, like
/// every learning rate in the examples; the conventional momentum is
/// `0.95`. The momentum table is parameter-aligned and initializes
/// lazily from the first step's gradients, and every step is pure
/// table algebra over pure payload arithmetic, so two identical runs
/// cannot differ.
#[derive(Debug, Clone)]
pub struct Muon<E> {
    momentum: Tensor<E>,
    orthogonalizer: NewtonSchulz<E>,
    buffer: Option<Parameters<E>>,
}

impl<E: Element> Muon<E> {
    /// Creates the optimizer with the momentum decay and the
    /// orthogonalizing iteration.
    ///
    /// # Panics
    /// Panics if `momentum` holds more than one value.
    pub fn new(momentum: Tensor<E>, orthogonalizer: NewtonSchulz<E>) -> Self {
        assert_single_value(&momentum, "momentum");
        Self {
            momentum,
            orthogonalizer,
            buffer: None,
        }
    }

    /// Steps like [`Optimizer::step`] with `policy` deciding, per
    /// parameter, whether the update is orthogonalized; a parameter it
    /// declines steps along the plain momentum direction. The current
    /// payload carries the parameter's shape, and a symbol set works
    /// for excluding embeddings and heads, since [`Symbol`] is
    /// `Eq + Hash`.
    ///
    /// # Panics
    /// Panics if `policy` selects a parameter below rank 2, which has
    /// no matrix to orthogonalize.
    pub fn step_where(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
        mut policy: impl FnMut(Symbol, &Tensor<E>) -> bool,
    ) -> Parameters<E> {
        let zeros = || gradients.map(|gradient| gradient.zero_like());
        let buffer = self.buffer.take().unwrap_or_else(zeros);
        let buffer = buffer.scale(&self.momentum) + gradients.clone();
        let direction = gradients + &buffer.scale(&self.momentum);
        self.buffer = Some(buffer);

        parameters.step_each(&direction, |symbol, current, direction| {
            let update = if policy(symbol, current) {
                self.orthogonalized(direction)
            } else {
                direction.clone()
            };
            current.clone() - update.clone() * learning_rate.broadcast_like(&update)
        })
    }

    /// Returns `direction` orthogonalized as the matrix `[axes[0],
    /// rest]` and scaled by `sqrt(max(1, rows / columns))`, back in
    /// its own shape.
    fn orthogonalized(&self, direction: &Tensor<E>) -> Tensor<E> {
        let shape = direction.shape();
        assert!(
            shape.rank() >= 2,
            "Muon orthogonalizes parameters of rank 2 and above, not {shape}"
        );
        let rows = shape.axes()[0];
        let columns = shape.volume() / rows;
        let matrix = direction.reshape(Shape::new([rows, columns]));
        let orthogonal = self.orthogonalizer.orthogonalize(&matrix);

        let ratio: Tensor<E> =
            Tensor::counted(Shape::scalar(), rows) / Tensor::counted(Shape::scalar(), columns);
        let scale = ratio.maximum(&ratio.one_like()).sqrt();
        (orthogonal.clone() * scale.broadcast_like(&orthogonal)).reshape(shape)
    }
}

impl<E: Element> Optimizer<E> for Muon<E> {
    fn step(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
    ) -> Parameters<E> {
        self.step_where(parameters, gradients, learning_rate, |_, parameter| {
            parameter.shape().rank() >= 2
        })
    }
}

#[cfg(test)]
#[path = "tests/muon_tests.rs"]
mod tests;
