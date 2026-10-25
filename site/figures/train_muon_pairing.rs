{
    use topos::{AdamW, Muon, NewtonSchulz, Optimizer, Parameters, Tape, Tensor};

    // A toy language model's parameters: an embedding table, one
    // hidden matrix, and a bias, with three rows looked up.
    let tape: Tape<f64> = Tape::new();
    let embedding = tape.parameter(Tensor::new([4, 2], [1.0, -0.5, 0.25, 2.0, 0.5, 1.5, -1.0, 0.75]));
    let hidden = tape.parameter(Tensor::new([2, 2], [0.5, -1.0, 1.5, 0.25]));
    let bias = tape.parameter(Tensor::new([2], [0.5, -1.0]));
    let rows = tape.leaf(Tensor::selection([2, 0, 3], 4, 1.0));
    let features = rows.matmul(embedding).matmul(hidden);
    let shifted = features + bias.broadcast_along_like(0, features);
    let loss = (shifted * shifted).sum();
    let (embedding, hidden, bias, loss) = (embedding.symbol(), hidden.symbol(), bias.symbol(), loss.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();
    let gradients = network.forward(&parameters, []).backward(loss).parameters(&parameters);

    // Each optimizer steps the whole table at its own rate. Muon
    // orthogonalizes only the hidden matrix; the pairing then selects
    // each parameter's next payload by symbol.
    let quintic = NewtonSchulz::new(
        [Tensor::from(3.4445), Tensor::from(-4.7750), Tensor::from(2.0315)],
        5,
        Tensor::from(1e-7),
    );
    let mut muon = Muon::new(Tensor::from(0.95), quintic);
    let mut adamw = AdamW::new(Tensor::from(0.9), Tensor::from(0.95), Tensor::from(1e-8), Tensor::from(0.1));
    let by_muon = muon.step_where(&parameters, &gradients, &Tensor::from(0.02), |symbol, _| symbol == hidden);
    let by_adamw = adamw.step(&parameters, &gradients, &Tensor::from(3e-4));
    let next: Parameters<f64> = by_muon.step_each(&by_adamw, |symbol, muon, adamw| {
        if symbol == hidden { muon.clone() } else { adamw.clone() }
    });

    let bits = |tensor: &Tensor<f64>| -> Vec<u64> { tensor.to_vec().iter().map(|v| v.to_bits()).collect() };
    assert_eq!(bits(next.of(hidden)), bits(by_muon.of(hidden)));
    assert_eq!(bits(next.of(embedding)), bits(by_adamw.of(embedding)));
    assert_eq!(bits(next.of(bias)), bits(by_adamw.of(bias)));

    // How far each parameter moved: the Euclidean norm of its change.
    let moved = |table: &Parameters<f64>, symbol| {
        let change = table.of(symbol).clone() - parameters.of(symbol).clone();
        (change.clone() * change).sum().sqrt().scalar()
    };
    show::lines([
        "parameter   shape   route   moved by".to_string(),
        format!("embedding   [4, 2]  adamw   {:.4}", moved(&next, embedding)),
        format!("hidden      [2, 2]  muon    {:.4}", moved(&next, hidden)),
        format!("bias        [2]     adamw   {:.4}", moved(&next, bias)),
        String::new(),
        format!(
            "hidden under muon alone {:.4}, under adamw alone {:.4}",
            moved(&by_muon, hidden),
            moved(&by_adamw, hidden)
        ),
        "next == muon's hidden, adamw's embedding and bias, bit for bit: asserted".to_string(),
    ])
}
