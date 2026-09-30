{
    use malevich::{Line, Plot};
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let loss = loss.symbol();
    let network = tape.into_network();

    // Lowered once, the plan holds no state: each step feeds the
    // current parameters, reads the recorded gradient, and mints the
    // next parameters. The learning rate is spread over each gradient
    // explicitly, because nothing broadcasts by accident.
    let plan = network.entry(adjoints.roots()).lower();
    let learning_rate = Tensor::from(0.05_f32);
    let mut current = network.parameters();
    let mut losses = Vec::new();
    for _ in 0..40 {
        let run = plan.forward(&current, []);
        losses.push(run.of(loss).scalar());
        let gradients = run.recorded_gradients(&adjoints);
        current = current.step(&gradients, |weight, gradient| {
            weight.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    show::plot(
        Plot::new()
            .layer(Line::y(losses).label("loss"))
            .title("forty steps through the compiled plan")
            .x_label("step")
            .y_label("loss"),
    )
}
