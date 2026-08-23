use crate::{Opcode, Shape, Tape, Tensor};

#[test]
fn describe_prints_the_golden_spec() {
    // The dense-layer shape of the note's example: matmul, an
    // axis-wise bias broadcast, an add, and a map.
    let tape: Tape<f64> = Tape::new();
    let weights = tape.parameter(Tensor::filled([10, 32], 0.0_f64));
    let input = tape.input(Tensor::filled([8, 10], 0.0));
    let product = input.matmul(weights);
    let bias = tape.parameter(Tensor::filled([32], 0.0));
    let shifted = product + bias.broadcast_along(0, product);
    let _activated = shifted.tanh();
    let network = tape.into_network();

    assert_eq!(
        network.describe(),
        "   0  Parameter                         [10, 32]\n\
         \x20  1  Input                             [8, 10]\n\
         \x20  2  MatMul         1, 0               [8, 32]\n\
         \x20  3  Parameter                         [32]\n\
         \x20  4  BroadcastAlong 3, 2  axis=0       [8, 32]\n\
         \x20  5  Add            2, 4               [8, 32]\n\
         \x20  6  Tanh           5                  [8, 32]\n\
         network: 7 nodes, 2 parameters, 1 input\n"
    );
}

#[test]
fn nodes_snapshot_the_recording() {
    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(2.0_f64);
    let y = tape.leaf(3.0);
    let product = x * y;
    let (x, y, product) = (x.symbol(), y.symbol(), product.symbol());
    let network = tape.into_network();

    let node = network.node(product);
    assert_eq!(node.symbol(), product);
    assert_eq!(*node.opcode(), Opcode::Mul);
    assert_eq!(node.operands(), [x, y]);
    assert_eq!(*node.shape(), Shape::scalar());
    assert!(!node.is_source());

    let nodes: Vec<_> = network.nodes().collect();
    assert_eq!(nodes.len(), 3);
    assert!(nodes[0].is_source());
    assert_eq!(nodes[0].opcode().arity(), 0);
    assert_eq!(nodes[2].opcode().arity(), 2);
}

#[test]
fn the_open_tape_answers_the_same_view() {
    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(2.0_f64);
    let doubled = x * 2.0;
    let (x, doubled) = (x.symbol(), doubled.symbol());

    // The literal records its own leaf between the parameter and the
    // product, so the open tape holds three nodes.
    assert_eq!(tape.nodes().len(), 3);
    assert_eq!(*tape.node(doubled).opcode(), Opcode::Mul);
    assert!(tape.describe().contains("tape: 3 nodes, 1 parameter"));

    // Payload reads answer sources only.
    assert_eq!(tape.payload(x), Some(2.0.into()));
    assert_eq!(tape.payload(doubled), None);
}

#[test]
fn sealed_payload_reads_answer_sources_only() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.5_f64);
    let x = tape.input(3.0);
    let y = tape.leaf(4.0);
    let loss = w * x + y;
    let (w, x, y, loss) = (w.symbol(), x.symbol(), y.symbol(), loss.symbol());
    let network = tape.into_network();

    assert_eq!(network.payload(w), Some(&Tensor::from(1.5)));
    assert_eq!(network.payload(x), Some(&Tensor::from(3.0)));
    assert_eq!(network.payload(y), Some(&Tensor::from(4.0)));
    assert_eq!(network.payload(loss), None);
}

#[test]
#[should_panic(expected = "different network")]
fn node_rejects_foreign_symbols() {
    let first: Tape<f64> = Tape::new();
    let second: Tape<f64> = Tape::new();
    let foreign = second.parameter(1.0_f64).symbol();
    first.parameter(2.0_f64);
    first.into_network().node(foreign);
}

#[test]
fn plan_results_are_the_declared_order() {
    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(2.0_f64);
    let squared = x * x;
    let cubed = squared * x;
    let (x, squared, cubed) = (x.symbol(), squared.symbol(), cubed.symbol());
    let network = tape.into_network();

    // Roots in request order, then observes; declaration order, not
    // allocation order.
    let plan = network.compile(crate::Request::roots([cubed, squared]).observe([x]));
    assert_eq!(plan.results(), [cubed, squared, x]);
}
