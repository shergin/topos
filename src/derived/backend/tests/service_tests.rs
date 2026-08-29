use crate::{Backend, Formula, Numerics, Precision, Tensor};

/// A deterministic dense payload, so no constant-fill shortcut keeps
/// a task from reaching the chain.
fn dense(len: usize, seed: u64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            ((index as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed)
                % 1000) as f64
                / 500.0
                - 1.0
        })
        .collect()
}

#[test]
fn a_tallied_product_lands_in_one_row() {
    let left = Tensor::new([256, 256], dense(256 * 256, 1));
    let right = Tensor::new([256, 256], dense(256 * 256, 2));
    let (_, services) = Backend::tallied(|| left.matmul(&right));

    assert_eq!(services.len(), 1);
    let row = services[0];
    assert_eq!(row.formula, Formula::Gemm);
    assert_eq!(row.precision, Precision::F64);
    assert_eq!(row.count, 1);
    // The product is above every chain member's threshold, so who
    // served is exactly a question of what the build compiled and
    // the machine initialized; the row reports it either way.
    let served = Formula::Gemm
        .chain(Precision::F64)
        .iter()
        .any(|backend| backend.compiled() && backend.status().is_ok());
    assert_eq!(row.backend.is_some(), served);
}

#[test]
fn exact_scopes_fall_to_the_reference_row() {
    let payload = Tensor::new([64, 64], dense(64 * 64, 3));
    let (_, services) = Backend::tallied(|| {
        Numerics::exactly(|| {
            let _ = payload.tanh();
            let _ = payload.tanh();
        })
    });

    // Two map tasks reached the chain, no admitted member in any
    // build, one aggregated reference row.
    assert_eq!(services.len(), 1);
    let row = services[0];
    assert_eq!(row.formula, Formula::Map);
    assert_eq!(row.precision, Precision::F64);
    assert_eq!(row.backend, None);
    assert_eq!(row.count, 2);
}

#[test]
fn nested_tallies_capture_innermost() {
    let payload = Tensor::new([64, 64], dense(64 * 64, 4));
    let ((_, inner), outer) =
        Backend::tallied(|| Backend::tallied(|| Numerics::exactly(|| payload.sqrt())));

    assert_eq!(inner.len(), 1);
    assert!(
        outer.is_empty(),
        "the enclosing tally must not double-count the inner scope"
    );
}
