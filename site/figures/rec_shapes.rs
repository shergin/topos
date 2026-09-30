{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let batch = tape.input(Tensor::filled([3, 2], 0.0_f32));
    let weights = tape.parameter(Tensor::filled([2, 4], 0.1));
    let bias = tape.parameter(Tensor::filled([4], 0.0));
    let affine = batch.matmul(weights) + bias.broadcast_along(0, 3);
    let _per_sample = affine.tanh().sum_along(1);
    show::spec(tape.describe())
}
