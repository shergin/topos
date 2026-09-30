{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();

    let spec_nodes = tape.len();
    let adjoints = tape.differentiate(loss, [weights]);
    let appended = tape.len() - spec_nodes;
    let gradient = adjoints.of(weights.symbol()).index();
    show::spec_from(
        tape.describe(),
        spec_nodes,
        &format!("{appended} nodes appended by differentiate; node {gradient} is d loss / d weights"),
    )
}
