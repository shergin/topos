{
    use topos::{Numerics, Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let (weights, hidden, loss) = (weights.symbol(), hidden.symbol(), loss.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();

    // The interpreter over the whole spec is the oracle; its reverse
    // scan applies the derivative rules without recording.
    let oracle = network.forward(&parameters, []);
    let engine_gradient = oracle.backward(loss).parameters(&parameters);

    // A plan under `Exact` must reproduce the oracle's bits; the
    // default `Fast` plan may let a backend reorder a sum.
    let declare = || network.entry(adjoints.roots()).observe([hidden]);
    let exact_run = declare().numerics(Numerics::Exact).lower().forward(&parameters, []);
    let fast_run = declare().lower().forward(&parameters, []);
    let recorded_gradient = exact_run.recorded_gradients(&adjoints);

    let bits = |tensor: &Tensor<f32>| -> Vec<u32> { tensor.to_vec().iter().map(|v| v.to_bits()).collect() };
    assert_eq!(bits(oracle.of(loss)), bits(exact_run.of(loss)));
    assert_eq!(bits(engine_gradient.of(weights)), bits(recorded_gradient.of(weights)));
    let fast_agrees = bits(fast_run.of(loss)) == bits(exact_run.of(loss));
    show::lines([
        "loss:".to_string(),
        format!("  interpreter        {}", oracle.of(loss)),
        format!("  exact plan         {}", exact_run.of(loss)),
        format!("  fast plan          {}", fast_run.of(loss)),
        "d loss / d weights:".to_string(),
        format!("  engine scan        {}", engine_gradient.of(weights)),
        format!("  recorded gradient  {}", recorded_gradient.of(weights)),
        format!("hidden, declared readable: {}", exact_run.of(hidden)),
        String::new(),
        "exact plan == interpreter, bit for bit: asserted".to_string(),
        "recorded gradient == engine scan, bit for bit: asserted".to_string(),
        format!(
            "fast plan == exact plan: {}",
            if fast_agrees { "yes; no backend claimed any of this graph in this build" } else { "no; a backend served part of it under Fast" }
        ),
    ])
}
