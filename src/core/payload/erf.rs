//! The error function pair behind the `Erf` and `ErfDerivative` map
//! operations.
//!
//! `erf` delegates to the pure-Rust `libm` crate (the Sun/FDLIBM
//! implementation, accurate to about one ulp and bit-identical on
//! every platform), like the rest of the transcendental vocabulary;
//! Rust's standard library has no `erf`, and the platform's C library
//! would need `unsafe` and vary by target. A crate-owned three-piece
//! series implementation preceded the delegation and lives in the
//! history should the dependency ever need replacing.
//!
//! The derivative is computed here: `libm` has no scaled Gaussian,
//! and the composed `exp(-x*x)` would let the rounding of `x*x` grow
//! through the exponential. Splitting the argument into an exactly
//! squared sixteenth-step head and a small tail keeps the rounding a
//! few epsilon at every magnitude. This module is also where the
//! pair's transcendental constant lives: a concrete `f64` is in
//! scope, so the kernel states the correctly rounded `2/sqrt(pi)` in
//! its own precision — the same standing as the constants inside
//! `libm` itself.

/// Computes the error function of `x`.
///
/// It is odd, saturates to `+-1`, keeps the sign of zero, and
/// propagates NaN.
pub(crate) fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Computes the derivative of the error function of `x`:
/// `(2/sqrt(pi)) * e^(-x^2)`, the scaled Gaussian.
///
/// It is even, underflows to zero past `|x| ~ 27`, and propagates
/// NaN.
pub(crate) fn erf_derivative(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y >= 27.5 {
        // `e^(-756.25)` is far below the smallest subnormal.
        return 0.0;
    }
    // The split: `head` carries at most ten significant bits, so
    // `head * head` is exact, and the tail stays small enough that
    // one rounding of it cannot amplify.
    let head = (y * 16.0).floor() / 16.0;
    let tail = (y - head) * (y + head);
    std::f64::consts::FRAC_2_SQRT_PI * libm::exp(-head * head) * libm::exp(-tail)
}

#[cfg(test)]
#[path = "tests/erf_tests.rs"]
mod tests;
