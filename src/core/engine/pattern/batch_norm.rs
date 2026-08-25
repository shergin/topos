use smallvec::{SmallVec, smallvec};

use crate::op::{Map, Op};
use crate::{Element, MapOperation, Tensor};

use super::candidates::Candidate;
use super::pattern::Pattern;
use super::view::View;

/// A matched batch normalization: the recorded normalization diamond —
/// centering, the epsilon-stabilized deviation, the learned affine —
/// rooted at the trailing shift `Add`. Both modes share the group;
/// the mode lives on the `Pattern` variant wrapping it. In training
/// the statistics are the batch's own reductions and become the
/// group's named results: they may sit in the keep-set (training
/// loops observe them for running estimates),
/// `stablehlo.batch_norm_training` writes their SSA names at the
/// root, and a fusing forward run writes their slots back beside the
/// root's. In inference they are supplied values, ordinary extra
/// reads of `stablehlo.batch_norm_inference`, and the variant stays
/// raise-only.
#[derive(Debug, Clone)]
pub(crate) struct BatchNormalization {
    /// The rank-2 `[batch, features]` input.
    pub(crate) input: usize,
    /// The rank-1 `[features]` learned scale.
    pub(crate) scale: usize,
    /// The rank-1 `[features]` learned shift.
    pub(crate) shift: usize,
    /// The single-value epsilon leaf, rendered as the raised
    /// operation's attribute.
    pub(crate) epsilon: usize,
    /// The `[features]` mean: a named result in training, a supplied
    /// extra read in inference.
    pub(crate) mean: usize,
    /// The `[features]` biased variance: a named result in training, a
    /// supplied extra read in inference.
    pub(crate) variance: usize,
}

impl BatchNormalization {
    /// Returns the slots the fused call reads past the root's operand
    /// links; liveness must keep them alive until the call.
    pub(crate) fn reads(&self) -> [usize; 4] {
        [self.input, self.scale, self.shift, self.epsilon]
    }

    /// Computes the fused call over the already-evaluated `values`:
    /// the output with the batch statistics, the diamond between the
    /// reads never materialized. The caller writes the statistics
    /// back into their named-result slots.
    pub(crate) fn apply<E: Element>(
        &self,
        values: &[Tensor<E>],
    ) -> (Tensor<E>, Tensor<E>, Tensor<E>) {
        values[self.input].batch_normalized(
            &values[self.scale],
            &values[self.shift],
            &values[self.epsilon],
        )
    }
}

/// The shared tail both variants record: everything from the trailing
/// shift `Add` down to the centering `Sub`, with the statistic
/// operands left unclassified.
struct Tail {
    /// The group as matched so far, its statistics unclassified.
    group: BatchNormalization,
    /// The tail's own nodes, all unnamed interiors.
    interiors: SmallVec<[usize; 8]>,
    /// The centering `Sub`, the diamond's fan-out point.
    centered: usize,
}

/// Matches the shared normalization tail rooted at the `Add` at
/// `index`: `centered / sqrt(variance + epsilon) * scale + shift`,
/// with every broadcast a unary node whose axis parameter names the
/// batch axis; the shapes agree by the record-time equal-shape
/// assertions of the binary nodes between them. The interiors are
/// collected by walking the formula — `centered` fans out five ways —
/// and `Catalog::collect` checks the closure.
fn match_tail<E: Element>(index: usize, view: &View<Tensor<E>>) -> Option<Tail> {
    let Some(Op::Add(_)) = view.op(index) else {
        return None;
    };
    // Cheap reject: the output is a rank-2 `[batch, features]` value
    // whose second operand broadcasts a rank-1 shift.
    if view.shape(index).rank() != 2 {
        return None;
    }
    let scaled = view.operand(index, 0);
    let shift_bcast = view.operand(index, 1);
    let Some(Op::BroadcastAlong(shift_along)) = view.op(shift_bcast) else {
        return None;
    };
    let Some(Op::Mul(_)) = view.op(scaled) else {
        return None;
    };
    let shift = view.sole_operand(shift_bcast);
    let normalized = view.operand(scaled, 0);
    let scale_bcast = view.operand(scaled, 1);
    let Some(Op::BroadcastAlong(scale_along)) = view.op(scale_bcast) else {
        return None;
    };
    if shift_along.axis != 0 || scale_along.axis != 0 {
        return None;
    }
    let scale = view.sole_operand(scale_bcast);
    let Some(Op::Div(_)) = view.op(normalized) else {
        return None;
    };
    let centered = view.operand(normalized, 0);
    let dev_bcast = view.operand(normalized, 1);
    let Some(Op::BroadcastAlong(dev_along)) = view.op(dev_bcast) else {
        return None;
    };
    if dev_along.axis != 0 {
        return None;
    }
    let deviation = view.sole_operand(dev_bcast);
    let Some(Op::Map(Map {
        op: MapOperation::Sqrt,
    })) = view.op(deviation)
    else {
        return None;
    };
    let var_plus = view.sole_operand(deviation);
    let Some(Op::Add(_)) = view.op(var_plus) else {
        return None;
    };
    let variance = view.operand(var_plus, 0);
    let eps_bcast = view.operand(var_plus, 1);
    let Some(Op::Broadcast(_)) = view.op(eps_bcast) else {
        return None;
    };
    let epsilon = view.sole_operand(eps_bcast);
    // The raise renders epsilon as the operation's attribute, so it
    // must be a single-value leaf whose payload emission can read.
    let Some(Op::Leaf(_)) = view.op(epsilon) else {
        return None;
    };
    if view.shape(epsilon).volume() != 1 {
        return None;
    }
    let Some(Op::Sub(_)) = view.op(centered) else {
        return None;
    };
    let input = view.operand(centered, 0);
    let mean_bcast = view.operand(centered, 1);
    let Some(Op::BroadcastAlong(mean_along)) = view.op(mean_bcast) else {
        return None;
    };
    if mean_along.axis != 0 {
        return None;
    }
    let mean = view.sole_operand(mean_bcast);
    Some(Tail {
        group: BatchNormalization {
            input,
            scale,
            shift,
            epsilon,
            mean,
            variance,
        },
        interiors: smallvec![
            scaled,
            shift_bcast,
            scale_bcast,
            normalized,
            dev_bcast,
            deviation,
            var_plus,
            eps_bcast,
            centered,
            mean_bcast,
        ],
        centered,
    })
}

/// Returns the reduction behind a recorded `mean_along(0)` at `node` —
/// `Div(SumAlong(source, 0), counted leaf)` — as the source, the sum,
/// and the count leaf. The leaf must certify as `counted` of the
/// reduced shape and the source's batch extent: an unverified divisor
/// would raise a formula that is not a mean.
fn mean_along_of<E: Element>(node: usize, view: &View<Tensor<E>>) -> Option<(usize, usize, usize)> {
    let Some(Op::Div(_)) = view.op(node) else {
        return None;
    };
    let sum = view.operand(node, 0);
    let count = view.operand(node, 1);
    let Some(Op::SumAlong(along)) = view.op(sum) else {
        return None;
    };
    if along.axis != 0 {
        return None;
    }
    let Some(Op::Leaf(leaf)) = view.op(count) else {
        return None;
    };
    let source = view.sole_operand(sum);
    let batch = view.shape(source).axes()[0];
    if !leaf.0.is_counted(view.shape(node), batch) {
        return None;
    }
    Some((source, sum, count))
}

/// Matches the training-mode batch-normalization formula rooted at
/// `index`: the shared tail whose statistics are the batch's own
/// `mean_along` reductions of the input and of the squared centering.
/// The mean and variance are named results; everything else in the
/// diamond is an unnamed interior.
pub(crate) fn match_training<E: Element>(
    index: usize,
    view: &View<Tensor<E>>,
) -> Option<Candidate> {
    let mut tail = match_tail(index, view)?;
    let (mean_source, mean_sum, mean_count) = mean_along_of(tail.group.mean, view)?;
    if mean_source != tail.group.input {
        return None;
    }
    let (squared, var_sum, var_count) = mean_along_of(tail.group.variance, view)?;
    let Some(Op::Mul(_)) = view.op(squared) else {
        return None;
    };
    if view.operand(squared, 0) != tail.centered || view.operand(squared, 1) != tail.centered {
        return None;
    }
    tail.interiors
        .extend_from_slice(&[mean_sum, mean_count, squared, var_sum, var_count]);
    let named = smallvec![tail.group.mean, tail.group.variance];
    Some(Candidate {
        pattern: Pattern::BatchNormTraining(tail.group),
        root: index,
        interiors: tail.interiors,
        named,
    })
}

/// Matches the inference-mode batch-normalization formula rooted at
/// `index`: the shared tail over supplied statistics. It runs after
/// [`match_training`] in catalog order (training is the more specific
/// ending), and a training recording cannot fall through to it
/// anyway: there the centering feeds the variance computation, a
/// consumer outside this tail, so the closure check rejects it.
pub(crate) fn match_inference<E: Element>(
    index: usize,
    view: &View<Tensor<E>>,
) -> Option<Candidate> {
    let tail = match_tail(index, view)?;
    Some(Candidate {
        pattern: Pattern::BatchNormInference(tail.group),
        root: index,
        interiors: tail.interiors,
        named: SmallVec::new(),
    })
}

#[cfg(test)]
#[path = "tests/batch_norm_tests.rs"]
mod tests;
