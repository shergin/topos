{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum().symbol();
    let weights = weights.symbol();
    let network = tape.into_network();

    // `backward()` is a memory posture: runs keep what the engine scan
    // will reread, so `Run::backward` still works on a plan's run.
    let plan = network.entry([loss]).backward().lower();
    let run = plan.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
    let mut text = plan.describe();
    text.push_str(&format!("\nd loss / d weights through the plan's run: {}\n", gradients.of(weights)));
    show::spec(text)
}
