{
    use topos::{Linear, Module, Shape, Tape, Tensor, init};

    let weights: Tensor<f32> = init::xavier(7)(&Shape::new([3, 2]));
    let bias = Tensor::filled([2], 0.0_f32);

    // The facade.
    let facade: Tape<f32> = Tape::new();
    let layer = Linear::new(&facade, weights.clone(), bias.clone());
    let _output = layer.express(facade.input(Tensor::filled([4, 3], 0.0_f32)));

    // By hand, through the public operations alone.
    let by_hand: Tape<f32> = Tape::new();
    let weights = by_hand.parameter(weights);
    let bias = by_hand.parameter(bias);
    let batch = by_hand.input(Tensor::filled([4, 3], 0.0_f32));
    let _output = batch.matmul(weights) + bias.broadcast_along(0, 4);

    assert_eq!(facade.describe(), by_hand.describe());
    show::panels([
        ("Linear::express", show::spec(facade.describe())),
        ("by hand", show::spec(by_hand.describe())),
    ])
}
