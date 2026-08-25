use super::SlotId;
use crate::graph::Opcode;
use crate::{Element, MapOperation, Shape, Tensor, Tensorial};

use static_assertions::assert_impl_all;

use super::{
    Add, Broadcast, BroadcastAlong, Cotangents, Div, Fold, Gather, Input, Leaf, LogSoftmax,
    LogSumExp, Map, MatMul, Maximum, Mul, Narrow, Neg, Operation, Pad, Parameter, Permute, Powf,
    Reads, Reshape, Scatter, Step, Sub, Sum, SumAlong, Unfold,
};

// Entry-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Op<f64>: Send, Sync);

/// The differentiable operation that produced a value, together with the
/// operation's parameters.
///
/// It is a statically sized closed set: each variant owns exactly its
/// parameters (a leaf's payload, a parameter's slot, a reduction's
/// axis), while the node's operand links live beside the node in the
/// tape's operands column and reach every method as a positional slice.
/// The enum dispatches to the variants with a plain `match`, avoiding
/// boxing and vtables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op<Data> {
    Leaf(Leaf<Data>),
    Parameter(Parameter),
    Input(Input),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Neg(Neg),
    Map(Map),
    MatMul(MatMul),
    Sum(Sum),
    SumAlong(SumAlong),
    Broadcast(Broadcast),
    BroadcastAlong(BroadcastAlong),
    Reshape(Reshape),
    Permute(Permute),
    Narrow(Narrow),
    Pad(Pad),
    Unfold(Unfold),
    Fold(Fold),
    Gather(Gather),
    Scatter(Scatter),
    LogSoftmax(LogSoftmax),
    LogSumExp(LogSumExp),
    Powf(Powf),
    Maximum(Maximum),
    Step(Step),
}

impl<Data> Op<Data> {
    /// Creates a leaf op holding `data`.
    pub(crate) fn leaf(data: Data) -> Self {
        Op::Leaf(Leaf(data))
    }

    /// Creates a parameter op referencing `slot`.
    pub(crate) fn parameter(slot: SlotId) -> Self {
        Op::Parameter(Parameter(slot))
    }

    /// Creates an input op referencing `slot`.
    pub(crate) fn input(slot: SlotId) -> Self {
        Op::Input(Input(slot))
    }

    /// Creates the sum of the `[left, right]` operands.
    pub(crate) fn add() -> Self {
        Op::Add(Add)
    }

    /// Creates the difference of the `[left, right]` operands.
    pub(crate) fn sub() -> Self {
        Op::Sub(Sub)
    }

    /// Creates the product of the `[left, right]` operands.
    pub(crate) fn mul() -> Self {
        Op::Mul(Mul)
    }

    /// Creates the quotient of the `[left, right]` operands.
    pub(crate) fn div() -> Self {
        Op::Div(Div)
    }

    /// Creates the negation of the single operand.
    pub(crate) fn neg() -> Self {
        Op::Neg(Neg)
    }

    /// Creates the unary elementwise transcendental `op` of the single
    /// operand.
    pub(crate) fn map(op: MapOperation) -> Self {
        Op::Map(Map { op })
    }

    /// Creates the matrix product of the `[left, right]` operands.
    pub(crate) fn matmul() -> Self {
        Op::MatMul(MatMul)
    }

    /// Creates the sum of every value in the single operand.
    pub(crate) fn sum() -> Self {
        Op::Sum(Sum)
    }

    /// Creates the sum of the single operand along `axis`.
    pub(crate) fn sum_along(axis: usize) -> Self {
        Op::SumAlong(SumAlong { axis })
    }

    /// Creates the explicit broadcast of the single-value operand
    /// across `shape`.
    pub(crate) fn broadcast(shape: Shape) -> Self {
        Op::Broadcast(Broadcast { shape })
    }

    /// Creates the explicit repetition of the single operand along a
    /// new axis of `extent` inserted at `axis`.
    pub(crate) fn broadcast_along(axis: usize, extent: usize) -> Self {
        Op::BroadcastAlong(BroadcastAlong { axis, extent })
    }

    /// Creates the reshape of the single operand to `shape`.
    pub(crate) fn reshape(shape: Shape) -> Self {
        Op::Reshape(Reshape { shape })
    }

    /// Creates the permutation of the single operand's axes by `order`.
    pub(crate) fn permute(order: impl IntoIterator<Item = usize>) -> Self {
        Op::Permute(Permute {
            order: order.into_iter().collect(),
        })
    }

    /// Creates the window of `len` elements from `start` along `axis` of
    /// the single operand.
    pub(crate) fn narrow(axis: usize, start: usize, len: usize) -> Self {
        Op::Narrow(Narrow { axis, start, len })
    }

    /// Creates the single operand placed at `start ..` along `axis`
    /// inside zeros of `full_extent`.
    pub(crate) fn pad(axis: usize, start: usize, full_extent: usize) -> Self {
        Op::Pad(Pad {
            axis,
            start,
            full_extent,
        })
    }

    /// Creates the sliding windows of the single operand along `axis`:
    /// `size` elements every `dilation` steps, one window every `step`.
    pub(crate) fn unfold(axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        Op::Unfold(Unfold {
            axis,
            size,
            step,
            dilation,
        })
    }

    /// Creates the row gather over the `[table, selection]` operands: the
    /// table's rows picked by the one-hot selection.
    pub(crate) fn gather() -> Self {
        Op::Gather(Gather)
    }

    /// Creates the log-softmax of the single operand along `axis`.
    pub(crate) fn log_softmax(axis: usize) -> Self {
        Op::LogSoftmax(LogSoftmax { axis })
    }

    /// Creates the fused log-sum-exp of the operand along `axis`.
    pub(crate) fn log_sum_exp(axis: usize) -> Self {
        Op::LogSumExp(LogSumExp { axis })
    }

    /// Creates the Heaviside step of `[operand, threshold]`.
    pub(crate) fn step() -> Self {
        Op::Step(Step)
    }

    /// Creates the fold of the operand's `(count, size)` pair at
    /// `axis` back onto an axis of `extent`.
    pub(crate) fn fold(
        axis: usize,
        size: usize,
        step: usize,
        dilation: usize,
        extent: usize,
    ) -> Self {
        Op::Fold(Fold {
            axis,
            size,
            step,
            dilation,
            extent,
        })
    }

    /// Creates the scatter-add of `[gradient, selection]` into the
    /// selection's vocabulary rows.
    pub(crate) fn scatter() -> Self {
        Op::Scatter(Scatter)
    }

    /// Creates the elementwise power of the `[base, exponent]` operands.
    pub(crate) fn powf() -> Self {
        Op::Powf(Powf)
    }

    /// Creates the elementwise maximum of the `[left, right]` operands.
    pub(crate) fn maximum() -> Self {
        Op::Maximum(Maximum)
    }

    /// Returns the public opcode of this op: the payload-free
    /// snapshot the IR view prints, one variant per variant here.
    pub(crate) fn opcode(&self) -> Opcode {
        match self {
            Op::Leaf(_) => Opcode::Leaf,
            Op::Parameter(_) => Opcode::Parameter,
            Op::Input(_) => Opcode::Input,
            Op::Add(_) => Opcode::Add,
            Op::Sub(_) => Opcode::Sub,
            Op::Mul(_) => Opcode::Mul,
            Op::Div(_) => Opcode::Div,
            Op::Neg(_) => Opcode::Neg,
            Op::Map(map) => Opcode::Map { operation: map.op },
            Op::MatMul(_) => Opcode::MatMul,
            Op::Sum(_) => Opcode::Sum,
            Op::SumAlong(sum_along) => Opcode::SumAlong {
                axis: sum_along.axis,
            },
            Op::Broadcast(broadcast) => Opcode::Broadcast {
                shape: broadcast.shape.clone(),
            },
            Op::BroadcastAlong(broadcast_along) => Opcode::BroadcastAlong {
                axis: broadcast_along.axis,
                extent: broadcast_along.extent,
            },
            Op::Reshape(reshape) => Opcode::Reshape {
                shape: reshape.shape.clone(),
            },
            Op::Permute(permute) => Opcode::Permute {
                order: permute.order.clone(),
            },
            Op::Narrow(narrow) => Opcode::Narrow {
                axis: narrow.axis,
                start: narrow.start,
                len: narrow.len,
            },
            Op::Pad(pad) => Opcode::Pad {
                axis: pad.axis,
                start: pad.start,
                full_extent: pad.full_extent,
            },
            Op::Unfold(unfold) => Opcode::Unfold {
                axis: unfold.axis,
                size: unfold.size,
                step: unfold.step,
                dilation: unfold.dilation,
            },
            Op::Fold(fold) => Opcode::Fold {
                axis: fold.axis,
                size: fold.size,
                step: fold.step,
                dilation: fold.dilation,
                extent: fold.extent,
            },
            Op::Gather(_) => Opcode::Gather,
            Op::Scatter(_) => Opcode::Scatter,
            Op::LogSoftmax(log_softmax) => Opcode::LogSoftmax {
                axis: log_softmax.axis,
            },
            Op::LogSumExp(log_sum_exp) => Opcode::LogSumExp {
                axis: log_sum_exp.axis,
            },
            Op::Step(_) => Opcode::Step,
            Op::Powf(_) => Opcode::Powf,
            Op::Maximum(_) => Opcode::Maximum,
        }
    }

    /// Returns which payload values this op's derivative rule
    /// reads: the read contract behind training-plan liveness.
    /// Sources have no rule and retain nothing.
    pub(crate) fn reads(&self) -> Reads {
        match self {
            Op::Leaf(_) | Op::Parameter(_) | Op::Input(_) => Reads::NOTHING,
            Op::Add(add) => add.reads(),
            Op::Sub(sub) => sub.reads(),
            Op::Mul(mul) => mul.reads(),
            Op::Div(div) => div.reads(),
            Op::Neg(neg) => neg.reads(),
            Op::Map(map) => map.reads(),
            Op::MatMul(matmul) => matmul.reads(),
            Op::Sum(sum) => sum.reads(),
            Op::SumAlong(sum_along) => sum_along.reads(),
            Op::Broadcast(broadcast) => broadcast.reads(),
            Op::BroadcastAlong(broadcast_along) => broadcast_along.reads(),
            Op::Reshape(reshape) => reshape.reads(),
            Op::Permute(permute) => permute.reads(),
            Op::Narrow(narrow) => narrow.reads(),
            Op::Pad(pad) => pad.reads(),
            Op::Unfold(unfold) => unfold.reads(),
            Op::Gather(gather) => gather.reads(),
            Op::LogSoftmax(log_softmax) => log_softmax.reads(),
            Op::LogSumExp(log_sum_exp) => log_sum_exp.reads(),
            Op::Step(step) => step.reads(),
            Op::Fold(fold) => fold.reads(),
            Op::Scatter(scatter) => scatter.reads(),
            Op::Powf(powf) => powf.reads(),
            Op::Maximum(maximum) => maximum.reads(),
        }
    }

    /// Returns the number of operand links this op expects.
    ///
    /// Sources have none; recording asserts every node's operand list
    /// against this count at the single append site.
    pub(crate) fn arity(&self) -> usize {
        match self {
            Op::Leaf(_) | Op::Parameter(_) | Op::Input(_) => 0,
            Op::Add(add) => add.arity(),
            Op::Sub(sub) => sub.arity(),
            Op::Mul(mul) => mul.arity(),
            Op::Div(div) => div.arity(),
            Op::Neg(neg) => neg.arity(),
            Op::Map(map) => map.arity(),
            Op::MatMul(matmul) => matmul.arity(),
            Op::Sum(sum) => sum.arity(),
            Op::SumAlong(sum_along) => sum_along.arity(),
            Op::Broadcast(broadcast) => broadcast.arity(),
            Op::BroadcastAlong(broadcast_along) => broadcast_along.arity(),
            Op::Reshape(reshape) => reshape.arity(),
            Op::Permute(permute) => permute.arity(),
            Op::Narrow(narrow) => narrow.arity(),
            Op::Pad(pad) => pad.arity(),
            Op::Unfold(unfold) => unfold.arity(),
            Op::Gather(gather) => gather.arity(),
            Op::LogSoftmax(log_softmax) => log_softmax.arity(),
            Op::LogSumExp(log_sum_exp) => log_sum_exp.arity(),
            Op::Step(step) => step.arity(),
            Op::Fold(fold) => fold.arity(),
            Op::Scatter(scatter) => scatter.arity(),
            Op::Powf(powf) => powf.arity(),
            Op::Maximum(maximum) => maximum.arity(),
        }
    }
}

/// It hand-rolls the delegation an enum-dispatch macro would generate: a
/// plain `match` per method. Exhaustiveness makes adding a variant a
/// compile error until every method handles it. Leaves and parameters
/// are supplied here rather than computed: they do not implement
/// `Operation`, whose contract is the derivative rule alone.
/// Forward computes over the one payload the graph has — `Tensor<E>`,
/// through each operation's inherent method — because computing a
/// payload is engine business, not part of any rule.
impl<E: Element> Op<Tensor<E>> {
    /// Computes this node's payload from its `operands`' payloads
    /// (gathered positionally by the engine), or supplies it: a leaf's
    /// embedded payload, a parameter's entry in the run's `parameters`
    /// slots, or an input's entry in the run's `inputs` slots.
    pub(crate) fn forward(
        &self,
        operands: &[&Tensor<E>],
        parameters: &[Tensor<E>],
        inputs: &[Tensor<E>],
    ) -> Tensor<E> {
        match self {
            Op::Leaf(leaf) => leaf.0.clone(),
            Op::Parameter(parameter) => parameters[parameter.0.index()].clone(),
            Op::Input(input) => inputs[input.0.index()].clone(),
            Op::Add(add) => add.forward(operands),
            Op::Sub(sub) => sub.forward(operands),
            Op::Mul(mul) => mul.forward(operands),
            Op::Div(div) => div.forward(operands),
            Op::Neg(neg) => neg.forward(operands),
            Op::Map(map) => map.forward(operands),
            Op::MatMul(matmul) => matmul.forward(operands),
            Op::Sum(sum) => sum.forward(operands),
            Op::SumAlong(sum_along) => sum_along.forward(operands),
            Op::Broadcast(broadcast) => broadcast.forward(operands),
            Op::BroadcastAlong(broadcast_along) => broadcast_along.forward(operands),
            Op::Reshape(reshape) => reshape.forward(operands),
            Op::Permute(permute) => permute.forward(operands),
            Op::Narrow(narrow) => narrow.forward(operands),
            Op::Pad(pad) => pad.forward(operands),
            Op::Unfold(unfold) => unfold.forward(operands),
            Op::Gather(gather) => gather.forward(operands),
            Op::LogSoftmax(log_softmax) => log_softmax.forward(operands),
            Op::LogSumExp(log_sum_exp) => log_sum_exp.forward(operands),
            Op::Step(step) => step.forward(operands),
            Op::Fold(fold) => fold.forward(operands),
            Op::Scatter(scatter) => scatter.forward(operands),
            Op::Powf(powf) => powf.forward(operands),
            Op::Maximum(maximum) => maximum.forward(operands),
        }
    }

    /// Infers the shape of this op's result from its `operands`'
    /// positional shapes, panicking on incompatibility.
    ///
    /// It is the shape-level mirror of `forward`: the same fold over the
    /// tape, run once per node at record time instead of once per run.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        match self {
            Op::Leaf(leaf) => leaf.infer_shape(),
            Op::Parameter(_) => {
                unreachable!("parameter shapes are recorded by `record_parameter`")
            }
            Op::Input(_) => {
                unreachable!("input shapes are recorded by `record_input`")
            }
            Op::Add(add) => add.infer_shape(operands),
            Op::Sub(sub) => sub.infer_shape(operands),
            Op::Mul(mul) => mul.infer_shape(operands),
            Op::Div(div) => div.infer_shape(operands),
            Op::Neg(neg) => neg.infer_shape(operands),
            Op::Map(map) => map.infer_shape(operands),
            Op::MatMul(matmul) => matmul.infer_shape(operands),
            Op::Sum(sum) => sum.infer_shape(operands),
            Op::SumAlong(sum_along) => sum_along.infer_shape(operands),
            Op::Broadcast(broadcast) => broadcast.infer_shape(operands),
            Op::BroadcastAlong(broadcast_along) => broadcast_along.infer_shape(operands),
            Op::Reshape(reshape) => reshape.infer_shape(operands),
            Op::Permute(permute) => permute.infer_shape(operands),
            Op::Narrow(narrow) => narrow.infer_shape(operands),
            Op::Pad(pad) => pad.infer_shape(operands),
            Op::Unfold(unfold) => unfold.infer_shape(operands),
            Op::Gather(gather) => gather.infer_shape(operands),
            Op::LogSoftmax(log_softmax) => log_softmax.infer_shape(operands),
            Op::LogSumExp(log_sum_exp) => log_sum_exp.infer_shape(operands),
            Op::Step(step) => step.infer_shape(operands),
            Op::Fold(fold) => fold.infer_shape(operands),
            Op::Scatter(scatter) => scatter.infer_shape(operands),
            Op::Powf(powf) => powf.infer_shape(operands),
            Op::Maximum(maximum) => maximum.infer_shape(operands),
        }
    }
}

impl<Data> Op<Data> {
    /// Computes one cotangent per operand, given this node's computed
    /// `output` payload and its own `gradient`; empty for leaves,
    /// parameters, and inputs, where gradients stop and get read out.
    /// The engine accumulates the returned cotangents into its gradient
    /// buffer.
    ///
    /// The rule payload is a separate parameter from the tape's own
    /// `Data` because the rules are pure trait polymorphism: the engine
    /// applies them over payload buffers, and `differentiate` applies
    /// the very same rules over recording `Trace`
    /// handles — one source of derivative truth, two interpretations.
    pub(crate) fn backward<Rule: Tensorial>(
        &self,
        operands: &[&Rule],
        output: &Rule,
        gradient: &Rule,
    ) -> Cotangents<Rule> {
        match self {
            Op::Leaf(_) | Op::Parameter(_) | Op::Input(_) => Cotangents::new(),
            Op::Add(add) => add.backward(operands, output, gradient),
            Op::Sub(sub) => sub.backward(operands, output, gradient),
            Op::Mul(mul) => mul.backward(operands, output, gradient),
            Op::Div(div) => div.backward(operands, output, gradient),
            Op::Neg(neg) => neg.backward(operands, output, gradient),
            Op::Map(map) => map.backward(operands, output, gradient),
            Op::MatMul(matmul) => matmul.backward(operands, output, gradient),
            Op::Sum(sum) => sum.backward(operands, output, gradient),
            Op::SumAlong(sum_along) => sum_along.backward(operands, output, gradient),
            Op::Broadcast(broadcast) => broadcast.backward(operands, output, gradient),
            Op::BroadcastAlong(broadcast_along) => {
                broadcast_along.backward(operands, output, gradient)
            }
            Op::Reshape(reshape) => reshape.backward(operands, output, gradient),
            Op::Permute(permute) => permute.backward(operands, output, gradient),
            Op::Narrow(narrow) => narrow.backward(operands, output, gradient),
            Op::Pad(pad) => pad.backward(operands, output, gradient),
            Op::Unfold(unfold) => unfold.backward(operands, output, gradient),
            Op::Gather(gather) => gather.backward(operands, output, gradient),
            Op::LogSoftmax(log_softmax) => log_softmax.backward(operands, output, gradient),
            Op::LogSumExp(log_sum_exp) => log_sum_exp.backward(operands, output, gradient),
            Op::Step(step) => step.backward(operands, output, gradient),
            Op::Fold(fold) => fold.backward(operands, output, gradient),
            Op::Scatter(scatter) => scatter.backward(operands, output, gradient),
            Op::Powf(powf) => powf.backward(operands, output, gradient),
            Op::Maximum(maximum) => maximum.backward(operands, output, gradient),
        }
    }
}
