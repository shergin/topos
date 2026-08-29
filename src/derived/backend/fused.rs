use super::backend::BackendUnavailable;
use super::coverage::{Coverage, Dispatch, Fidelity};
use super::formula::{Formula, Precision};
use super::manifest::Manifest;

/// The crate's own fused kernels for composed formulas, elected
/// onto plans at compile time and executing in-process through the
/// payload seam.
///
/// The actions live with their consumer (the kernel table beside
/// the plan, per the symmetry decision); this manifest holds only
/// the declared coverage the election reads.
pub(super) struct Fused;

impl Manifest for Fused {
    const DISPATCH: Dispatch = Dispatch::Elected;

    fn coverage(formula: Formula) -> Coverage {
        match formula {
            // `windowed_product` computes through the gemm seam in
            // the recorded accumulation order: exactly as faithful as
            // the composition it replaces, under either posture,
            // because the interior product walks the same chain.
            Formula::WindowProduct => Coverage::Serves {
                fidelity: Fidelity::Composed,
                precisions: Precision::ALL,
            },
            // `batch_normalized` offers the whole group down the
            // chain and falls back to composing the recorded formula
            // through the same payload operations the rules make, so
            // the chain's own admission keeps `Exact` runs on the
            // reference and the envelope enters only through an
            // admitted hardware kernel under `Fast`.
            Formula::BatchNormTraining => Coverage::Serves {
                fidelity: Fidelity::Composed,
                precisions: Precision::ALL,
            },
            // `max_pooled` folds each window with `maximum` in the
            // recorded lane order — a direct walk that materializes
            // no lane views and consults no chain — so its bits are
            // the reference bits in every build, under either
            // posture: the one absolute bit-identity cell.
            Formula::ReduceWindow => Coverage::Serves {
                fidelity: Fidelity::BitIdentical,
                precisions: Precision::ALL,
            },
            // The inference-mode normalization stays raise-only
            // until a consumer earns it.
            Formula::Gemm | Formula::Map | Formula::BatchNormInference => Coverage::Absent,
        }
    }

    fn compiled() -> bool {
        true
    }

    fn status() -> Result<(), BackendUnavailable> {
        // In-process code compiled into every build: nothing to
        // initialize, nothing to lose.
        Ok(())
    }
}
