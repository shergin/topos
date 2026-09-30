{
    use topos::{Backend, Shape, Tape, init};

    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(init::xavier(3)(&Shape::new([64, 64])));
    let batch = tape.input(init::normal(5, 1.0)(&Shape::new([32, 64])));
    let loss = batch.matmul(weights).tanh().sum();
    let adjoints = tape.differentiate(loss, [weights]);
    let network = tape.into_network();
    let plan = network.entry(adjoints.roots()).lower();

    // Wrap any region: the tally says which backend served each formula.
    let (_, services) = Backend::tallied(|| plan.forward(&network.parameters(), []));
    let lines: Vec<String> = services
        .iter()
        .map(|service| {
            let server = match service.backend {
                Some(backend) => format!("{backend:?}"),
                None => "reference".to_string(),
            };
            format!("{:?} {:?} x{}: {server}", service.formula, service.precision, service.count)
        })
        .collect();
    show::text(lines.join("\n"))
}
