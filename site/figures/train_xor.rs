{
    use malevich::{Cells, Plot};
    use topos::{Activation, Mlp, Module, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[2, 4, 1], Activation::Tanh, init::uniform(7, 0.5));
    // The defaults only fix the shapes: two samples of two features.
    let x = tape.input(Tensor::filled([2, 2], 0.0_f32));
    let y = tape.input(Tensor::filled([2, 1], 0.0));
    let predicted = mlp.express(x);
    let error = predicted - y;
    let loss = (error * error).sum();
    let (x, y, loss, predicted) = (x.symbol(), y.symbol(), loss.symbol(), predicted.symbol());
    let network = tape.into_network();
    let mut parameters = network.parameters();

    let minibatches = [
        (Tensor::new([2, 2], [0.0, 0.0, 0.0, 1.0]), Tensor::new([2, 1], [-1.0, 1.0])),
        (Tensor::new([2, 2], [1.0, 0.0, 1.0, 1.0]), Tensor::new([2, 1], [1.0, -1.0])),
    ];
    let learning_rate = Tensor::from(0.05_f32);
    for step in 0..3000 {
        let (batch_x, batch_y) = &minibatches[step % 2];
        let run = network.forward(&parameters, [(x, batch_x.clone()), (y, batch_y.clone())]);
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    // The surface, two grid cells per run through the same expression.
    let (columns, rows) = (24, 12);
    let centers: Vec<(f32, f32)> = (0..rows)
        .flat_map(|row| (0..columns).map(move |column| ((column as f32 + 0.5) / columns as f32, (row as f32 + 0.5) / rows as f32)))
        .collect();
    let mut surface = Vec::new();
    for pair in centers.chunks(2) {
        let feed = Tensor::new([2, 2], [pair[0].0, pair[0].1, pair[1].0, pair[1].1]);
        surface.extend(network.forward(&parameters, [(x, feed)]).of(predicted).to_vec());
    }
    show::plot(
        Plot::new()
            .layer(Cells::matrix(columns, surface).extents((0.0, 1.0), (0.0, 1.0)))
            .colorbar()
            .title("the learned xor surface"),
    )
}
