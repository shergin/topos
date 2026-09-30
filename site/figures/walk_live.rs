{
    use malevich::{Line, LineStyle, Plot, Rule};
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let hidden = hidden.symbol();
    let network = tape.into_network();

    let plan = network.entry(adjoints.roots()).observe([hidden]).lower();
    let live: Vec<f64> = plan.live_series().iter().map(|&elements| elements as f64).collect();
    let peak = live.iter().copied().fold(0.0, f64::max);
    show::plot(
        Plot::new()
            .layer(Line::y(live).label("live elements").style(LineStyle::Corners))
            .layer(Rule::h(peak).label(format!("peak {peak}")))
            .title("live volume along the schedule")
            .x_label("scheduled node")
            .y_label("elements"),
    )
}
