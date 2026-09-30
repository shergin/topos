{
    use malevich::Theme;
    use topos::{Numerics, Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let hidden = hidden.symbol();
    let network = tape.into_network();
    let entry = network.entry(adjoints.roots()).observe([hidden]).numerics(Numerics::Exact);
    show::card(entry.to_html(Theme::DARK))
}
