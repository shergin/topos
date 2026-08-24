use super::{erf, erf_derivative};

/// Asserts `actual` within `tolerance` epsilons of `expected`,
/// relative to the expected magnitude.
fn assert_close(actual: f64, expected: f64, tolerance: f64, probe: f64) {
    let bound = tolerance * f64::EPSILON * expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= bound,
        "erf({probe}) = {actual}, expected {expected}"
    );
}

#[test]
fn erf_matches_the_reference_table() {
    // Expected values from CPython's `math.erf` (the platform libm),
    // covering the small-argument, mid-range, tail, and saturation
    // regimes; `libm`'s FDLIBM port must agree within a few ulps.
    let table = [
        (1e-300, 1.1283791670955125e-300),
        (1e-20, 1.1283791670955125e-20),
        (0.03125, 0.035250373867322826),
        (0.1, 0.1124629160182849),
        (0.46875, 0.4926134732179379),
        (0.5, 0.5204998778130465),
        (0.84375, 0.7672256612323416),
        (0.875, 0.7840750610598597),
        (1.0, 0.8427007929497148),
        (1.5, 0.9661051464753108),
        (2.0, 0.9953222650189527),
        (2.5, 0.999593047982555),
        (3.0, 0.9999779095030015),
        (3.5, 0.9999992569016276),
        (3.9990234375, 0.9999999844582503),
        (4.0, 0.9999999845827421),
        (4.5, 0.9999999998033839),
        (5.0, 0.9999999999984626),
        (5.5, 0.9999999999999927),
        (5.9, 1.0),
        (6.0, 1.0),
        (10.0, 1.0),
    ];
    for (probe, expected) in table {
        assert_close(erf(probe), expected, 10.0, probe);
        assert_close(erf(-probe), -expected, 10.0, -probe);
    }
}

#[test]
fn erf_is_exactly_odd() {
    for step in 1..600 {
        let probe = step as f64 * 0.01;
        assert_eq!(erf(-probe), -erf(probe), "asymmetry at {probe}");
    }
}

#[test]
fn erf_pieces_join_monotonically() {
    // The three pieces and the saturation cut must hand over without
    // a downward step; erf is strictly increasing.
    let mut previous = erf(-6.5);
    for step in -6_500..=6_500 {
        let value = erf(step as f64 * 0.001);
        assert!(
            value >= previous,
            "erf steps down near {}",
            step as f64 * 0.001
        );
        previous = value;
    }
}

#[test]
fn erf_handles_the_edges() {
    assert_eq!(erf(0.0), 0.0);
    assert!(erf(-0.0).is_sign_negative());
    assert_eq!(erf(f64::INFINITY), 1.0);
    assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    assert!(erf(f64::NAN).is_nan());
}

#[test]
fn erf_derivative_matches_the_scaled_gaussian() {
    // Expected values are `FRAC_2_SQRT_PI * exp(-x*x)` from CPython.
    let table = [
        (0.0, std::f64::consts::FRAC_2_SQRT_PI),
        (0.1, 1.1171516067889369),
        (0.5, 0.8787825789354448),
        (1.0, 0.4151074974205947),
        (2.0, 0.020666985354092053),
        (3.0, 0.00013925305194674786),
        (5.0, 1.5670866531017336e-11),
    ];
    for (probe, expected) in table {
        assert_close(erf_derivative(probe), expected, 10.0, probe);
        assert_close(erf_derivative(-probe), expected, 10.0, -probe);
    }
    assert_eq!(erf_derivative(f64::INFINITY), 0.0);
    assert_eq!(erf_derivative(30.0), 0.0);
    assert!(erf_derivative(f64::NAN).is_nan());
}
