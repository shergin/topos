{
    use topos::{Bf16, Element, Tape, Tensor};

    // One recording function, generic over the element.
    fn loss_of<E: Element + From<f32>>() -> (String, Tensor<E>) {
        let tape: Tape<E> = Tape::new();
        let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4].map(E::from)));
        let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0].map(E::from)));
        let loss = input.matmul(weights).tanh().exp().sum().symbol();
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        (network.describe(), run.of(loss).clone())
    }
    let (spec_f64, loss_f64) = loss_of::<f64>();
    let (spec_f32, loss_f32) = loss_of::<f32>();
    let (spec_bf16, loss_bf16) = loss_of::<Bf16>();
    assert_eq!(spec_f64, spec_f32);
    assert_eq!(spec_f64, spec_bf16);
    show::lines([
        format!("the spec over f64, f32, and Bf16 is the same text: {}", spec_f64 == spec_f32 && spec_f32 == spec_bf16),
        format!("loss over f64   {loss_f64}"),
        format!("loss over f32   {loss_f32}"),
        format!("loss over Bf16  {loss_bf16}"),
    ])
}
