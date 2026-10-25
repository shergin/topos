{
    use malevich::{Line, Plot};
    use topos::{NewtonSchulz, Tensor};

    // A diagonal matrix is the clearest input: its entries are its
    // singular values, the stretch factors along its axes, and the
    // iteration keeps it diagonal, so each entry's path is read off
    // the diagonal. Any matrix with these singular values moves them
    // along the same paths; only its directions differ, and the
    // iteration leaves those alone.
    let stretches = [3.0_f64, 0.5, 0.1, 0.02];
    let mut entries = vec![0.0_f64; 16];
    for (index, stretch) in stretches.iter().enumerate() {
        entries[index * 5] = *stretch;
    }
    let matrix = Tensor::new([4, 4], entries);

    // The reference quintic, written here: the polynomial is the
    // caller's, not the facade's. Iteration zero is the input divided
    // by its Frobenius norm.
    let quintic = |steps| {
        NewtonSchulz::new(
            [Tensor::from(3.4445_f64), Tensor::from(-4.7750), Tensor::from(2.0315)],
            steps,
            Tensor::from(1e-7),
        )
    };
    let iterations: Vec<f64> = (0..=5).map(|step| step as f64).collect();
    let mut plot = Plot::new()
        .title("newton-schulz on four singular values")
        .x_label("iteration")
        .y_label("singular value")
        .log_y();
    for (index, stretch) in stretches.iter().enumerate() {
        let path: Vec<f64> = (0..=5)
            .map(|steps| quintic(steps).orthogonalize(&matrix).to_vec()[index * 5])
            .collect();
        plot = plot.layer(Line::xy(iterations.clone(), path).label(format!("from {stretch}")));
    }
    show::plot(plot)
}
