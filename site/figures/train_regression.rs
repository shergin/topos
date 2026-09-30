{
    use malevich::{Line, Plot, Points};
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    let features: Tensor<f32> = init::uniform(3, 3.0)(&Shape::new([96, 1]));
    let noise: Tensor<f32> = init::normal(5, 0.1)(&Shape::new([96, 1]));
    let xs = features.to_vec();
    let ys: Vec<f32> = xs.iter().zip(noise.to_vec()).map(|(x, n)| x.sin() + n).collect();

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[1, 16, 1], Activation::Tanh, init::xavier(7));
    let input = tape.input(features);
    let predicted = mlp.express(input);
    let error = predicted - tape.input(Tensor::new([96, 1], ys.clone()));
    let loss = (error * error).sum();
    let adjoints = tape.differentiate(loss, mlp.parameters());
    let (input, predicted) = (input.symbol(), predicted.symbol());
    let network = tape.into_network();

    let plan = network.entry(adjoints.roots()).lower();
    let learning_rate = Tensor::from(0.002_f32);
    let mut parameters = network.parameters();
    for _ in 0..1500 {
        let gradients = plan.forward(&parameters, []).recorded_gradients(&adjoints);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    // The fit: the trained expression fed an even grid instead of the
    // samples. Same nodes, different feed.
    let grid: Vec<f32> = (0..96).map(|i| -3.0 + 6.0 * i as f32 / 95.0).collect();
    let run = network.forward(&parameters, [(input, Tensor::new([96, 1], grid.clone()))]);
    let fit = run.of(predicted).to_vec();
    show::plot(
        Plot::new()
            .layer(Points::xy(xs, ys).label("samples"))
            .layer(Line::function(-3.0..3.0, f64::sin).label("sin x"))
            .layer(Line::xy(grid, fit).label("fit"))
            .title("a tanh mlp fit to noisy sin x")
            .x_label("x"),
    )
}
