{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let hidden = hidden.symbol();
    let network = tape.into_network();

    // The entry declares what a run may read: the loss and the
    // gradient as roots, plus `hidden` as an observed interior.
    let plan = network.entry(adjoints.roots()).observe([hidden]).lower();
    let results = plan.results().iter().map(|symbol| symbol.index());
    show::spec_marked(plan.describe(), results)
}
