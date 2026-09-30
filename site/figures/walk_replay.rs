{
    use topos::{Numerics, Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum().symbol();
    let network = tape.into_network();
    let oracle = network.forward(&network.parameters(), []);

    // Walk the printed lines: sources take their stored payloads, and
    // every computed node expresses its opcode over operands computed
    // earlier. Under the exact posture, this is the interpreter.
    let values = Numerics::exactly(|| {
        let mut values: Vec<Tensor<f32>> = Vec::new();
        for node in network.nodes() {
            let value = if node.is_source() {
                network.payload(node.symbol()).expect("sources hold payloads").clone()
            } else {
                let operands: Vec<&Tensor<f32>> =
                    node.operands().iter().map(|symbol| &values[symbol.index()]).collect();
                node.opcode().express(&operands)
            };
            values.push(value);
        }
        values
    });
    let by_hand = &values[loss.index()];
    assert_eq!(by_hand.scalar().to_bits(), oracle.of(loss).scalar().to_bits());
    show::lines([
        format!("walked {} lines by hand, outside the crate", values.len()),
        format!("loss by hand:        {by_hand}"),
        format!("loss by interpreter: {}", oracle.of(loss)),
        "same bits: asserted".to_string(),
    ])
}
