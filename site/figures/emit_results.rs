{
    use topos::{Tape, Tensor};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let (weights, hidden, loss) = (weights.symbol(), hidden.symbol(), loss.symbol());
    let network = tape.into_network();

    let plan = network.entry(adjoints.roots()).observe([hidden]).lower();
    let module = plan.emit_stablehlo().expect("every operation lowers");
    let signature = module.lines().find(|line| line.contains("func.func")).unwrap_or("").trim();
    let returned = module.lines().find(|line| line.trim().starts_with("return")).unwrap_or("").trim();
    let results: Vec<String> = plan.results().iter().map(|symbol| symbol.index().to_string()).collect();
    show::lines([
        format!("entry roots:    {} (the loss), {} (its gradient)", loss.index(), adjoints.of(weights).index()),
        format!("entry observes: {} (hidden)", hidden.index()),
        format!("plan.results(): [{}]", results.join(", ")),
        String::new(),
        signature.to_string(),
        format!("    {returned}"),
    ])
}
