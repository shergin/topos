{
    use std::f32::consts::PI;

    use malevich::{Cells, Plot, Points};
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    // Two interleaved half-moons of a hundred noisy points each.
    let noise: Tensor<f32> = init::normal(5, 0.1)(&Shape::new([200, 2]));
    let noise = noise.to_vec();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut labels = Vec::new();
    for moon in 0..2 {
        for index in 0..100 {
            let angle = PI * index as f32 / 99.0;
            let (x, y) = if moon == 0 {
                (angle.cos(), angle.sin())
            } else {
                (1.0 - angle.cos(), 0.5 - angle.sin())
            };
            let at = 2 * (moon * 100 + index);
            xs.push(x + noise[at]);
            ys.push(y + noise[at + 1]);
            labels.push(if moon == 0 { 1.0_f32 } else { -1.0 });
        }
    }
    let features: Vec<f32> = xs.iter().zip(&ys).flat_map(|(x, y)| [*x, *y]).collect();

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[2, 16, 16, 1], Activation::Tanh, init::xavier(7));
    let predicted = mlp.express(tape.input(Tensor::new([200, 2], features)));
    let error = predicted - tape.input(Tensor::new([200, 1], labels));
    let loss = (error * error).sum();
    let adjoints = tape.differentiate(loss, mlp.parameters());

    // A grid-shaped twin of the same parameters rasterizes the surface.
    let (columns, rows) = (48, 16);
    let (x_span, y_span) = ((-1.5_f32, 2.5_f32), (-1.0_f32, 1.5_f32));
    let centers: Vec<f32> = (0..rows)
        .flat_map(|row| {
            (0..columns).flat_map(move |column| {
                [
                    x_span.0 + (column as f32 + 0.5) / columns as f32 * (x_span.1 - x_span.0),
                    y_span.0 + (row as f32 + 0.5) / rows as f32 * (y_span.1 - y_span.0),
                ]
            })
        })
        .collect();
    let surface = mlp.express(tape.input(Tensor::new([columns * rows, 2], centers))).symbol();
    let network = tape.into_network();

    let plan = network.entry(adjoints.roots()).lower();
    let learning_rate = Tensor::from(0.0003_f32);
    let mut parameters = network.parameters();
    for _ in 0..2000 {
        let gradients = plan.forward(&parameters, []).recorded_gradients(&adjoints);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    let values = network.forward(&parameters, []).of(surface).to_vec();
    let (upper_x, lower_x) = xs.split_at(100);
    let (upper_y, lower_y) = ys.split_at(100);
    show::plot(
        Plot::new()
            .layer(Cells::matrix(columns, values).extents(
                (f64::from(x_span.0), f64::from(x_span.1)),
                (f64::from(y_span.0), f64::from(y_span.1)),
            ))
            .layer(Points::xy(upper_x.to_vec(), upper_y.to_vec()).label("class +1"))
            .layer(Points::xy(lower_x.to_vec(), lower_y.to_vec()).label("class -1"))
            .colorbar()
            .title("the learned decision surface"),
    )
}
