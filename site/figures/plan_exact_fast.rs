{
    use topos::{Numerics, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(init::xavier(3)(&Shape::new([64, 64])));
    let batch = tape.input(init::normal(5, 1.0)(&Shape::new([32, 64])));
    let loss = batch.matmul(weights).tanh().sum().symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let interpreter = network.forward(&parameters, []);
    let exact = network.entry([loss]).numerics(Numerics::Exact).lower().forward(&parameters, []);
    let fast = network.entry([loss]).lower().forward(&parameters, []);
    let bits = |tensor: &Tensor<f32>| tensor.scalar().to_bits();
    assert_eq!(bits(interpreter.of(loss)), bits(exact.of(loss)));
    show::lines([
        format!("interpreter  {}", interpreter.of(loss)),
        format!("exact plan   {}", exact.of(loss)),
        format!("fast plan    {}", fast.of(loss)),
        String::new(),
        "exact == interpreter: asserted, in every build".to_string(),
        format!(
            "fast == exact: {}",
            if bits(fast.of(loss)) == bits(exact.of(loss)) { "yes, no backend is on in this build" } else { "no: a backend summed in its own order" }
        ),
    ])
}
