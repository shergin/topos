{
    use topos::{Shape, Tape, Tensor, conv2d, init};

    let tape: Tape<f32> = Tape::new();
    let image = tape.input(init::normal(1, 1.0)(&Shape::new([1, 1, 6, 6])));
    let filters = tape.parameter(init::normal(2, 0.3)(&Shape::new([2, 1, 3, 3])));
    let bias = tape.parameter(Tensor::filled([2], 0.0_f32));
    let _convolved = conv2d(image, filters, bias, 1, 1);
    show::spec(tape.describe())
}
