{
    use malevich::{Line, Plot};
    use topos::{Activation, Adam, Mlp, Module, Optimizer, Sgd, Shape, Tape, Tensor, init};

    // A tanh MLP fit to sin x over 48 samples, recorded once.
    let features: Tensor<f32> = init::uniform(3, 3.0)(&Shape::new([48, 1]));
    let targets: Vec<f32> = features.to_vec().iter().map(|x| x.sin()).collect();
    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[1, 16, 1], Activation::Tanh, init::xavier(7));
    let error = mlp.express(tape.input(features)) - tape.input(Tensor::new([48, 1], targets));
    let loss = (error * error).sum();
    let adjoints = tape.differentiate(loss, mlp.parameters());
    let loss = loss.symbol();
    let network = tape.into_network();
    let plan = network.entry(adjoints.roots()).lower();

    // The loop owns the state and the learning rate; the optimizer is
    // the strategy it hands the gradients to.
    let mut curves = Vec::new();
    let mut sgd = Sgd;
    let mut adam = Adam::new(Tensor::from(0.9_f32), Tensor::from(0.999), Tensor::from(1e-8));
    let runs: [(&str, &mut dyn Optimizer<f32>, f32); 2] =
        [("sgd, lr 0.003", &mut sgd, 0.003), ("adam, lr 0.005", &mut adam, 0.005)];
    for (label, optimizer, learning_rate) in runs {
        let learning_rate = Tensor::from(learning_rate);
        let mut parameters = network.parameters();
        let mut losses = Vec::new();
        for _ in 0..400 {
            let run = plan.forward(&parameters, []);
            losses.push(run.of(loss).scalar());
            let gradients = run.recorded_gradients(&adjoints);
            parameters = optimizer.step(&parameters, &gradients, &learning_rate);
        }
        curves.push((label, losses));
    }
    let mut plot = Plot::new()
        .title("two optimizers, one recorded gradient")
        .x_label("step")
        .y_label("sum of squared errors")
        .log_y();
    for (label, losses) in curves {
        plot = plot.layer(Line::y(losses).label(label));
    }
    show::plot(plot)
}
