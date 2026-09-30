{
    use topos::{Linear, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let layer = Linear::new(
        &tape,
        init::xavier(7)(&Shape::new([3, 2])),
        Tensor::filled([2], 0.0_f32),
    );
    let batch = tape.input(Tensor::filled([4, 3], 0.0_f32));
    let _output = layer.express(batch);
    show::spec(tape.describe())
}
