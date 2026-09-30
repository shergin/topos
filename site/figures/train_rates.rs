{
    use malevich::{Line, Plot};
    use topos::{Detach, Tape, Tensor};

    let (network, (w, b, loss)) = Tape::record(|tape| {
        let w = tape.parameter(0.0_f64);
        let b = tape.parameter(0.0);
        // Three samples of the line y = 2x + 1, as leaves; the total
        // loss is the sum of the squared errors.
        let mut loss = None;
        for (x, y) in [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)] {
            let error = w * tape.leaf(x) + b - tape.leaf(y);
            let squared = error * error;
            loss = Some(match loss {
                Some(total) => total + squared,
                None => squared,
            });
        }
        (w, b, loss.expect("at least one sample")).detach()
    });
    let initial = network.parameters();

    let mut plot = Plot::new()
        .title("one spec, three learning rates")
        .x_label("step")
        .y_label("loss")
        .log_y();
    for learning_rate in [0.005, 0.02, 0.05] {
        // Cloning the state is all a what-if costs.
        let mut parameters = initial.clone();
        let mut losses = Vec::new();
        for _ in 0..300 {
            let run = network.forward(&parameters, []);
            losses.push(run.of(loss).scalar());
            let gradients = run.backward(loss).parameters(&parameters);
            parameters = parameters.step(&gradients, |parameter, gradient| {
                parameter.clone() - gradient.clone() * Tensor::from(learning_rate)
            });
        }
        let label = format!(
            "lr {learning_rate}: w {:.2}, b {:.2}",
            parameters.of(w).scalar(),
            parameters.of(b).scalar()
        );
        plot = plot.layer(Line::y(losses).label(label));
    }
    show::plot(plot)
}
