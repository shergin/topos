{
    use topos::{Tape, Tensor};

    let tape: Tape<f64> = Tape::new();
    let weights = tape.parameter(Tensor::new([3, 2], [0.5, -0.25, 0.125, 0.75, -0.5, 0.25]));
    let inputs = tape.input(Tensor::new([2, 3], [0.5, -1.0, 0.25, -0.75, 0.5, 1.25]));
    let scores = inputs.matmul(weights).tanh().log_softmax(1);
    let loss = (scores * scores).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let (weights, loss) = (weights.symbol(), loss.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();

    // The engine scan computes the gradient; the recorded nodes, run
    // by the same interpreter, answer it again.
    let run = network.forward(&parameters, []);
    let engine = run.backward(loss).parameters(&parameters);
    let recorded = run.recorded_gradients(&adjoints);
    let bits = |tensor: &Tensor<f64>| -> Vec<u64> { tensor.to_vec().iter().map(|v| v.to_bits()).collect() };
    assert_eq!(bits(engine.of(weights)), bits(recorded.of(weights)));
    show::lines([
        format!("engine scan        {}", engine.of(weights)),
        format!("recorded gradient  {}", recorded.of(weights)),
        "same bits in every element: asserted".to_string(),
    ])
}
