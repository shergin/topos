//! The spec read surface: the frozen IR is executable from the
//! public surface alone.
//!
//! This suite compiles as an external consumer, so every call goes
//! through `topos::` re-exports: walking `Network::nodes` and
//! expressing each opcode reconstructs the spec — over `Tensor` it
//! is the interpreter, over `Trace` it re-records — and a reverse
//! scan written against `Opcode::vjp` reproduces the engine scan's
//! bits. These are the welds the forward-mode example builds on.

use topos::{Detach, Network, Node, Numerics, Opcode, Symbol, Tape, Tensor, Trace};

/// A small mixed spec: matmul, tanh, the log-domain primitive, an
/// elementwise square, and a full reduction.
fn fixture() -> (Network<f64>, [Symbol; 3]) {
    let (network, symbols) = Tape::record(|tape| {
        let weights = tape.parameter(Tensor::new(
            [3, 2],
            (0..6)
                .map(|index| (index as f64 - 2.5) / 4.0)
                .collect::<Vec<_>>(),
        ));
        let inputs = tape.input(Tensor::new(
            [2, 3],
            (0..6)
                .map(|index| (index as f64 - 3.0) / 2.0)
                .collect::<Vec<_>>(),
        ));
        let scores = inputs.matmul(weights).tanh().log_softmax(1);
        let loss = (scores * scores).sum();
        [weights, inputs, loss].detach()
    });
    (network, symbols)
}

/// Replays the spec over tensors: sources take their stored
/// payloads, computed nodes express their opcodes over previously
/// computed operands.
fn replayed(network: &Network<f64>) -> Vec<Tensor<f64>> {
    let mut values: Vec<Tensor<f64>> = Vec::new();
    for node in network.nodes() {
        let value = if node.is_source() {
            network
                .payload(node.symbol())
                .expect("sources hold payloads")
                .clone()
        } else {
            let operands: Vec<&Tensor<f64>> = node
                .operands()
                .iter()
                .map(|symbol| &values[symbol.index()])
                .collect();
            node.opcode().express(&operands)
        };
        values.push(value);
    }
    values
}

fn assert_bitwise(expected: &Tensor<f64>, computed: &Tensor<f64>, subject: &str) {
    let expected = expected.to_vec();
    let computed = computed.to_vec();
    assert_eq!(expected.len(), computed.len(), "{subject}: length differs");
    for (expected, computed) in expected.iter().zip(&computed) {
        assert_eq!(
            expected.to_bits(),
            computed.to_bits(),
            "{subject}: {computed} differs from {expected}"
        );
    }
}

#[test]
fn replay_over_tensors_is_the_interpreter() {
    let (network, _) = fixture();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    // The whole-spec road is exact by construction; the replay's
    // tensor calls consult the ambient posture, so the walk enters
    // the same one.
    let values = Numerics::exactly(|| replayed(&network));
    for node in network.nodes() {
        assert_bitwise(
            run.of(node.symbol()),
            &values[node.symbol().index()],
            node.name(),
        );
    }
}

#[test]
fn replay_over_traces_re_records_the_spec() {
    let (network, _) = fixture();
    let tape: Tape<f64> = Tape::new();
    let mut traces: Vec<Trace<'_, f64>> = Vec::new();
    for node in network.nodes() {
        let trace = if node.is_source() {
            let payload = network
                .payload(node.symbol())
                .expect("sources hold payloads")
                .clone();
            let value = match node.opcode() {
                Opcode::Leaf => tape.leaf(payload),
                Opcode::Parameter => tape.parameter(payload),
                Opcode::Input => tape.input(payload),
                _ => unreachable!("`is_source` names exactly the sources"),
            };
            Trace::of(value)
        } else {
            let operands: Vec<&Trace<'_, f64>> = node
                .operands()
                .iter()
                .map(|symbol| &traces[symbol.index()])
                .collect();
            node.opcode().express(&operands)
        };
        traces.push(trace);
    }
    drop(traces);
    let re_recorded = tape.into_network();
    assert_eq!(re_recorded.describe(), network.describe());
}

#[test]
fn a_public_reverse_scan_matches_the_engine() {
    let (network, [weights, inputs, loss]) = fixture();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let engine = run.backward(loss);

    let nodes: Vec<Node> = network.nodes().collect();
    let (values, cotangents) = Numerics::exactly(|| {
        let values = replayed(&network);
        // The scan mirrors the engine's deliberately: the one-seed,
        // the `Some`-cotangent ancestor propagation, the zero-seeded
        // accumulation in reverse allocation order.
        let mut cotangents: Vec<Option<Tensor<f64>>> = vec![None; values.len()];
        cotangents[loss.index()] = Some(values[loss.index()].one_like());
        for index in (0..values.len()).rev() {
            let Some(seed) = cotangents[index].clone() else {
                continue;
            };
            let node = &nodes[index];
            if node.is_source() {
                continue;
            }
            let operands: Vec<&Tensor<f64>> = node
                .operands()
                .iter()
                .map(|symbol| &values[symbol.index()])
                .collect();
            let recovered = node.opcode().vjp(&operands, &values[index], &seed);
            for (symbol, cotangent) in node.operands().iter().zip(recovered) {
                if let Some(contribution) = cotangent {
                    let slot = symbol.index();
                    let seeded = match cotangents[slot].take() {
                        Some(existing) => existing,
                        None => values[slot].zero_like(),
                    };
                    cotangents[slot] = Some(seeded + contribution);
                }
            }
        }
        (values, cotangents)
    });

    for node in &nodes {
        let slot = node.symbol().index();
        let computed = match &cotangents[slot] {
            Some(gradient) => gradient.clone(),
            None => values[slot].zero_like(),
        };
        assert_bitwise(engine.of(node.symbol()), &computed, node.name());
    }
    // The named parameters are the gradients a training step reads.
    assert!(cotangents[weights.index()].is_some());
    assert!(cotangents[inputs.index()].is_some());
}
