{
    use malevich::{Line, LineStyle, Plot};
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[8, 32, 32, 4], Activation::Tanh, init::xavier(3));
    let features: Tensor<f32> = init::normal(5, 1.0)(&Shape::new([64, 8]));
    let predicted = mlp.express(tape.input(features));
    let loss = (predicted * predicted).sum();
    let adjoints = tape.differentiate(loss, mlp.parameters());
    let loss = loss.symbol();
    let network = tape.into_network();

    let training = network.entry(adjoints.roots()).lower();
    let inference = network.entry([loss]).lower();
    let series = |plan: &topos::Plan<f32>| -> Vec<f64> {
        plan.live_series().iter().map(|&elements| elements as f64).collect()
    };
    show::plot(
        Plot::new()
            .layer(Line::y(series(&training)).label("loss + gradients").style(LineStyle::Corners))
            .layer(Line::y(series(&inference)).label("loss only").style(LineStyle::Corners))
            .title("live volume along two plans of one [8, 32, 32, 4] mlp")
            .x_label("scheduled node")
            .y_label("elements"),
    )
}
