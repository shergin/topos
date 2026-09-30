{
    use malevich::{Line, Plot};
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[4, 8, 8, 1], Activation::Tanh, init::xavier(3));
    let features: Tensor<f32> = init::normal(5, 1.0)(&Shape::new([16, 4]));
    let targets: Tensor<f32> = init::normal(7, 1.0)(&Shape::new([16, 1]));
    let error = mlp.express(tape.input(features)) - tape.input(targets);
    let loss = (error * error).sum().symbol();
    let network = tape.into_network();

    // A field holds one payload per node; its norms are the shape of
    // the backward pass.
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
    let norms: Vec<f64> = gradients
        .payloads()
        .iter()
        .map(|payload| payload.to_vec().iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt())
        .collect();
    show::plot(
        Plot::new()
            .layer(Line::y(norms).label("gradient norm"))
            .title("the backward pass of a [4, 8, 8, 1] mlp, node by node")
            .x_label("node")
            .y_label("norm"),
    )
}
