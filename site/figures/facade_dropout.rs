{
    use topos::{Dropout, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let dropout = Dropout::new(&tape, [1, 6]);
    let activations = tape.input(Tensor::new([1, 6], [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let masked = dropout.express(activations).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    // Unfed, the mask is its all-ones default: inference.
    let inference = network.forward(&parameters, []);
    // Training draws a seeded mask on the host and feeds it like a batch.
    let mask = init::dropout(7, 0.5)(&Shape::new([1, 6]));
    let training = network.forward(&parameters, [(dropout.mask(), mask.clone())]);
    show::lines([
        format!("unfed  {}", inference.of(masked)),
        format!("mask   {mask}   (0 drops, 1 / keep rescales)"),
        format!("fed    {}", training.of(masked)),
    ])
}
