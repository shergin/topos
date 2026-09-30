{
    use malevich::{Line, Plot};
    use topos::{Detach, Tape, Tensor};

    let (network, [w, x, y, loss]) = Tape::record(|tape| {
        let w = tape.parameter(0.0_f64);
        let x = tape.input(0.0);
        let y = tape.input(0.0);
        let error = w * x - y;
        [w, x, y, error * error].detach()
    });
    let mut parameters = network.parameters();

    let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
    let mut losses = Vec::new();
    for step in 0..100 {
        let (sample_x, sample_y) = samples[step % samples.len()];
        let run = network.forward(&parameters, [(x, sample_x.into()), (y, sample_y.into())]);
        losses.push(run.of(loss).scalar());
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = parameters.step(&gradients, |w, g| {
            w.clone() - g.clone() * Tensor::from(0.02)
        });
    }
    let learned = parameters.of(w).scalar();
    show::plot(
        Plot::new()
            .layer(Line::y(losses).label(format!("w = {learned:.5}")))
            .title("a hundred steps toward y = 2x")
            .x_label("step")
            .y_label("squared error")
            .log_y(),
    )
}
