{
    use topos::{Linear, Module, Shape, Tape, Tensor, cross_entropy, init};

    let tape: Tape<f32> = Tape::new();
    let head = Linear::new(&tape, init::xavier(7)(&Shape::new([4, 3])), Tensor::filled([3], 0.0_f32));
    let features = tape.input(Tensor::filled([2, 4], 0.0_f32));
    let logits = head.express(features);
    // Two readings of one graph: a loss to train on, and probabilities
    // to sample from. They share the projection and nothing else.
    let targets = tape.input(Tensor::selection([0, 2], 3, 1.0_f32));
    let loss = cross_entropy(logits, targets).symbol();
    let probabilities = logits.softmax(1).symbol();
    let network = tape.into_network();

    let train = network.entry([loss]).lower();
    let sample = network.entry([probabilities]).lower();
    show::panels([
        ("train", show::spec(train.describe())),
        ("sample", show::spec(sample.describe())),
    ])
}
