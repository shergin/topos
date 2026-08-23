//! The crate-owned error function: the reference computation behind
//! the `Erf` and `ErfDerivative` map operations.
//!
//! Rust's standard library has no `erf`, and calling the platform's C
//! library would need `unsafe`, which the default build forbids — so
//! the crate owns the competence, from three first-principles pieces
//! a reader can derive rather than a coefficient table they must
//! trust:
//!
//! - `|x| <= 27/32`: the alternating Maclaurin series. Its terms
//!   decay from the very first ratio, so the cancellation never
//!   exceeds one bit.
//! - `27/32 < |x| < 4`: the scaled series (Abramowitz & Stegun
//!   7.1.6), `erf x = (2/sqrt(pi)) x e^(-x^2) * sum (2x^2)^n /
//!   (1*3*...*(2n+1))` — every term positive, so the sum is stable at
//!   any argument.
//! - `4 <= |x| < 6`: Laplace's continued fraction for the complement,
//!   `erfc x * sqrt(pi) * e^(x^2) = 1/(x + (1/2)/(x + 1/(x +
//!   (3/2)/(x + ...))))`, which makes `1 - erfc` sub-ulp exactly
//!   where `erf` saturates. From 6 on, `erfc` falls below half an ulp
//!   of one, and the answer is exactly `+-1`.
//!
//! `e^(-x^2)` in the last two pieces splits the argument into an
//! exactly squared sixteenth-step head and a small tail, so its
//! rounding stays a few epsilon at every magnitude instead of growing
//! with `x^2`. The grid test pins the result within a few ulps of the
//! true function across the whole line (relative error stays under
//! ten epsilon; the platform's `exp` supplies its usual last-bit
//! variance, exactly as it does for every other map operation).

/// The f64 nearest `1/sqrt(pi)`: the exact half of the standard
/// library's `FRAC_2_SQRT_PI`, since halving only decrements the
/// exponent.
const FRAC_1_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI / 2.0;

/// Computes `e^(-y*y)` for `0 <= y < 27.5` with the argument split
/// into an exactly squared head and a small tail, so the rounding of
/// `y*y` never amplifies through the exponential.
fn negated_square_exp(y: f64) -> f64 {
    let head = (y * 16.0).floor() / 16.0;
    let tail = (y - head) * (y + head);
    (-head * head).exp() * (-tail).exp()
}

/// Computes the error function of `x`.
///
/// It is odd, saturates to exactly `+-1` from `|x| = 6` on, keeps the
/// sign of zero, and propagates NaN.
pub(crate) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y >= 6.0 {
        return 1.0_f64.copysign(x);
    }
    if y <= 0.84375 {
        let z = y * y;
        let mut term = y;
        let mut total = 0.0;
        let mut n = 0;
        loop {
            let contribution = term / (2 * n + 1) as f64;
            total = if n % 2 == 0 {
                total + contribution
            } else {
                total - contribution
            };
            if contribution <= f64::EPSILON * total {
                break;
            }
            n += 1;
            term = term * z / n as f64;
        }
        return (std::f64::consts::FRAC_2_SQRT_PI * total).copysign(x);
    }
    if y < 4.0 {
        let z = y * y;
        let mut term = y;
        let mut total = y;
        let mut n = 0;
        while term > f64::EPSILON * total {
            n += 1;
            term = term * (2.0 * z) / (2 * n + 1) as f64;
            total += term;
        }
        return (std::f64::consts::FRAC_2_SQRT_PI * negated_square_exp(y) * total).copysign(x);
    }
    // Depth 32 converges past f64 precision for every `y >= 4`; the
    // grid test pins it.
    let mut descent = 0.0;
    for n in (1..=32).rev() {
        descent = (n as f64 / 2.0) / (y + descent);
    }
    let complement = FRAC_1_SQRT_PI * negated_square_exp(y) / (y + descent);
    (1.0 - complement).copysign(x)
}

/// Computes the derivative of the error function of `x`:
/// `(2/sqrt(pi)) * e^(-x^2)`, the scaled Gaussian.
///
/// It is even, underflows to zero past `|x| ~ 27`, and propagates
/// NaN. This is where the pair's transcendental constant lives: a
/// concrete `f64` is in scope here, so the kernel states the
/// correctly rounded value in its own precision — the same standing
/// as the constants inside the platform's `tanh`.
pub(crate) fn erf_derivative(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y >= 27.5 {
        // `e^(-756.25)` is far below the smallest subnormal.
        return 0.0;
    }
    std::f64::consts::FRAC_2_SQRT_PI * negated_square_exp(y)
}

#[cfg(test)]
#[path = "tests/erf_tests.rs"]
mod tests;
