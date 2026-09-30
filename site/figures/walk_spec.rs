{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let _loss = (hidden * hidden).sum();
    show::spec(tape.describe())
}
