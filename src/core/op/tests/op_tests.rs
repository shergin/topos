use smallvec::smallvec;

use crate::{MapOperation, Shape, Tensor};

use super::super::{
    Add, Broadcast, BroadcastAlong, Div, Fold, Gather, Input, Leaf, LogSoftmax, LogSumExp, Map,
    MatMul, Maximum, Mul, Narrow, Neg, Pad, Parameter, Permute, Powf, Reshape, Scatter, SlotId,
    Step, Sub, Sum, SumAlong, Unfold,
};
use super::Op;

/// One instance of every variant, so the welds below stay exhaustive:
/// adding a variant breaks this list at compile time.
fn every_op() -> Vec<Op<Tensor<f64>>> {
    vec![
        Op::Leaf(Leaf(Tensor::from(1.0))),
        Op::Parameter(Parameter(SlotId::new(0))),
        Op::Input(Input(SlotId::new(0))),
        Op::Add(Add),
        Op::Sub(Sub),
        Op::Mul(Mul),
        Op::Div(Div),
        Op::Neg(Neg),
        Op::Map(Map {
            op: MapOperation::Exp,
        }),
        Op::Powf(Powf),
        Op::Maximum(Maximum),
        Op::Step(Step),
        Op::MatMul(MatMul),
        Op::Sum(Sum),
        Op::SumAlong(SumAlong { axis: 0 }),
        Op::Broadcast(Broadcast {
            shape: Shape::new([2, 2]),
        }),
        Op::BroadcastAlong(BroadcastAlong { axis: 0, extent: 2 }),
        Op::Reshape(Reshape {
            shape: Shape::new([4]),
        }),
        Op::Permute(Permute {
            order: smallvec![1, 0],
        }),
        Op::Narrow(Narrow {
            axis: 0,
            start: 0,
            len: 1,
        }),
        Op::Pad(Pad {
            axis: 0,
            start: 0,
            full_extent: 2,
        }),
        Op::Unfold(Unfold {
            axis: 0,
            size: 1,
            step: 1,
            dilation: 1,
        }),
        Op::Fold(Fold {
            axis: 0,
            size: 1,
            step: 1,
            dilation: 1,
            extent: 1,
        }),
        Op::Gather(Gather),
        Op::Scatter(Scatter),
        Op::LogSoftmax(LogSoftmax { axis: 0 }),
        Op::LogSumExp(LogSumExp { axis: 0 }),
    ]
}

#[test]
fn the_two_arity_tables_agree() {
    // `Op::arity` and `Opcode::arity` are hand-maintained twins;
    // exhaustiveness catches a missing variant in either, and this
    // weld catches a wrong count. `is_source` rides along: exactly
    // the arity-0 variants are supplied rather than computed.
    for op in every_op() {
        let opcode = op.opcode();
        assert_eq!(op.arity(), opcode.arity(), "{}", opcode.name());
        assert_eq!(opcode.is_source(), op.arity() == 0, "{}", opcode.name());
    }
}

#[test]
fn read_contracts_fit_the_liveness_width() {
    // Training-plan liveness indexes `Reads::operands` by operand
    // position, so the array's width is the vocabulary's maximum
    // arity. A wider op must widen `Reads` the same day; this weld
    // makes that a test failure with a name instead of an index
    // panic inside a caller's compile.
    for op in every_op() {
        let reads = op.reads();
        assert!(
            op.arity() <= reads.operands.len(),
            "{} outgrew the `Reads::operands` width; widen it",
            op.opcode().name()
        );
        for position in op.arity()..reads.operands.len() {
            assert!(
                !reads.operands[position],
                "{} declares a read at position {position}, past its arity",
                op.opcode().name()
            );
        }
    }
}
