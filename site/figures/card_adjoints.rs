{
    use malevich::Theme;
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let bias = tape.parameter(Tensor::new([2], [0.0_f32, 0.0]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let loss = (input.matmul(weights) + bias.broadcast_along(0, 1)).tanh().sum();
    let adjoints = tape.differentiate(loss, [weights, bias]);
    show::card(adjoints.to_html(Theme::DARK))
}
