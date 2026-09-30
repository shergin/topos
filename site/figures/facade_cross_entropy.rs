{
    use topos::{Tape, Tensor, cross_entropy};

    let tape: Tape<f32> = Tape::new();
    let logits = tape.input(Tensor::filled([4, 3], 0.0_f32));
    // One-hot targets travel as a selection: the class index per row.
    let targets = tape.input(Tensor::selection([0, 2, 1, 2], 3, 1.0_f32));
    let _loss = cross_entropy(logits, targets);
    show::spec(tape.describe())
}
