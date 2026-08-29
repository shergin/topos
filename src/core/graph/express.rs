//! The executable reading of the printable IR: `express` runs one
//! opcode over any recordable payload, `vjp` applies its derivative
//! rule. Together they make the frozen spec replayable from the
//! public surface — over `Tensor` a walk is the interpreter, over
//! `Trace` it re-records, and an out-of-tree payload is a new
//! interpretation of the same spec, with the engine as its oracle.

use crate::op::{
    Add, Broadcast, BroadcastAlong, Div, Fold, Gather, LogSoftmax, LogSumExp, Map, MatMul, Maximum,
    Mul, Narrow, Neg, Operation, Pad, Permute, Powf, Reshape, Scatter, Step, Sub, Sum, SumAlong,
    Unfold,
};
use crate::{MapOperation, Recordable};

use super::Opcode;

impl Opcode {
    /// Computes or records this operation over `operands`, position
    /// for position: the executable reading of the printable IR.
    ///
    /// Walking a spec's nodes in allocation order and expressing each
    /// computed opcode over its operands' results reconstructs the
    /// spec under any [`Recordable`] interpretation: over
    /// [`Tensor`](crate::Tensor) it computes — the interpreter's own
    /// step — and over [`Trace`](crate::Trace) it re-records. A new
    /// interpretation (a dual number for forward mode, a shape
    /// analyzer) plugs in as a payload, never as a fork of the rules.
    ///
    /// # Panics
    /// Panics if this opcode is a source — `Leaf`, `Parameter`, and
    /// `Input` are supplied, not expressed — if `operands.len()`
    /// differs from [`Opcode::arity`], or as the operation's own
    /// shape checks panic: the same checks recording makes.
    pub fn express<R: Recordable>(&self, operands: &[&R]) -> R {
        assert!(
            !self.is_source(),
            "sources are supplied, not expressed; feed {} its payload instead",
            self.name()
        );
        assert_eq!(
            operands.len(),
            self.arity(),
            "{} takes {} operands",
            self.name(),
            self.arity()
        );
        match self {
            Opcode::Leaf | Opcode::Parameter | Opcode::Input => {
                unreachable!("sources are rejected above")
            }
            Opcode::Add => (*operands[0]).clone() + (*operands[1]).clone(),
            Opcode::Sub => (*operands[0]).clone() - (*operands[1]).clone(),
            Opcode::Mul => (*operands[0]).clone() * (*operands[1]).clone(),
            Opcode::Div => (*operands[0]).clone() / (*operands[1]).clone(),
            Opcode::Neg => -(*operands[0]).clone(),
            Opcode::Map { operation } => {
                let operand = operands[0];
                match operation {
                    MapOperation::Exp => operand.exp(),
                    MapOperation::Ln => operand.ln(),
                    MapOperation::Sqrt => operand.sqrt(),
                    MapOperation::Tanh => operand.tanh(),
                    MapOperation::Sin => operand.sin(),
                    MapOperation::Cos => operand.cos(),
                    MapOperation::Log1p => operand.log1p(),
                    MapOperation::Expm1 => operand.expm1(),
                    MapOperation::Erf => operand.erf(),
                    MapOperation::ErfDerivative => operand.erf_derivative(),
                }
            }
            Opcode::Powf => operands[0].powf((*operands[1]).clone()),
            Opcode::Maximum => operands[0].maximum(operands[1]),
            Opcode::Step => operands[0].step(operands[1]),
            Opcode::MatMul => operands[0].matmul(operands[1]),
            Opcode::Sum => operands[0].sum(),
            Opcode::SumAlong { axis } => operands[0].sum_along(*axis),
            Opcode::Broadcast { shape } => operands[0].broadcast(shape.clone()),
            Opcode::BroadcastAlong { axis, extent } => operands[0].broadcast_along(*axis, *extent),
            Opcode::Reshape { shape } => operands[0].reshape(shape.clone()),
            Opcode::Permute { order } => operands[0].permute(order),
            Opcode::Narrow { axis, start, len } => operands[0].narrow(*axis, *start, *len),
            Opcode::Pad {
                axis,
                start,
                full_extent,
            } => operands[0].pad(*axis, *start, *full_extent),
            Opcode::Unfold {
                axis,
                size,
                step,
                dilation,
            } => operands[0].unfold(*axis, *size, *step, *dilation),
            Opcode::Fold {
                axis,
                size,
                step,
                dilation,
                extent,
            } => operands[0].fold(*axis, *size, *step, *dilation, *extent),
            Opcode::Gather => operands[0].gather(operands[1]),
            Opcode::Scatter => operands[0].scatter(operands[1]),
            Opcode::LogSoftmax { axis } => operands[0].log_softmax(*axis),
            Opcode::LogSumExp { axis } => operands[0].logsumexp(*axis),
        }
    }

    /// Applies this operation's reverse-mode rule: the cotangent
    /// handed to each operand, position for position, given the
    /// forward `output` and the incoming `seed`. `None` marks an
    /// operand that is data rather than a differentiable dependency —
    /// a gather's selection — exactly as the engine scan treats it.
    ///
    /// It is the public name of the one rule body: the same
    /// `backward` the engine scan computes with and `differentiate`
    /// records with, under whichever [`Recordable`] interpretation
    /// `R` supplies. The engine scan remains the oracle a scan built
    /// on this surface is graded against.
    ///
    /// # Panics
    /// Panics if this opcode is a source — sources have no rule — or
    /// if `operands.len()` differs from [`Opcode::arity`].
    pub fn vjp<R: Recordable>(&self, operands: &[&R], output: &R, seed: &R) -> Vec<Option<R>> {
        assert!(
            !self.is_source(),
            "sources have no derivative rule; gradients stop at {}",
            self.name()
        );
        assert_eq!(
            operands.len(),
            self.arity(),
            "{} takes {} operands",
            self.name(),
            self.arity()
        );
        let cotangents = match self {
            Opcode::Leaf | Opcode::Parameter | Opcode::Input => {
                unreachable!("sources are rejected above")
            }
            Opcode::Add => Add.backward(operands, output, seed),
            Opcode::Sub => Sub.backward(operands, output, seed),
            Opcode::Mul => Mul.backward(operands, output, seed),
            Opcode::Div => Div.backward(operands, output, seed),
            Opcode::Neg => Neg.backward(operands, output, seed),
            Opcode::Map { operation } => Map { op: *operation }.backward(operands, output, seed),
            Opcode::Powf => Powf.backward(operands, output, seed),
            Opcode::Maximum => Maximum.backward(operands, output, seed),
            Opcode::Step => Step.backward(operands, output, seed),
            Opcode::MatMul => MatMul.backward(operands, output, seed),
            Opcode::Sum => Sum.backward(operands, output, seed),
            Opcode::SumAlong { axis } => SumAlong { axis: *axis }.backward(operands, output, seed),
            Opcode::Broadcast { shape } => Broadcast {
                shape: shape.clone(),
            }
            .backward(operands, output, seed),
            Opcode::BroadcastAlong { axis, extent } => BroadcastAlong {
                axis: *axis,
                extent: *extent,
            }
            .backward(operands, output, seed),
            Opcode::Reshape { shape } => Reshape {
                shape: shape.clone(),
            }
            .backward(operands, output, seed),
            Opcode::Permute { order } => Permute {
                order: order.clone(),
            }
            .backward(operands, output, seed),
            Opcode::Narrow { axis, start, len } => Narrow {
                axis: *axis,
                start: *start,
                len: *len,
            }
            .backward(operands, output, seed),
            Opcode::Pad {
                axis,
                start,
                full_extent,
            } => Pad {
                axis: *axis,
                start: *start,
                full_extent: *full_extent,
            }
            .backward(operands, output, seed),
            Opcode::Unfold {
                axis,
                size,
                step,
                dilation,
            } => Unfold {
                axis: *axis,
                size: *size,
                step: *step,
                dilation: *dilation,
            }
            .backward(operands, output, seed),
            Opcode::Fold {
                axis,
                size,
                step,
                dilation,
                extent,
            } => Fold {
                axis: *axis,
                size: *size,
                step: *step,
                dilation: *dilation,
                extent: *extent,
            }
            .backward(operands, output, seed),
            Opcode::Gather => Gather.backward(operands, output, seed),
            Opcode::Scatter => Scatter.backward(operands, output, seed),
            Opcode::LogSoftmax { axis } => {
                LogSoftmax { axis: *axis }.backward(operands, output, seed)
            }
            Opcode::LogSumExp { axis } => {
                LogSumExp { axis: *axis }.backward(operands, output, seed)
            }
        };
        cotangents.into_vec()
    }
}
