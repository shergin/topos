{
    use topos::{Shape, Tape, Tensor, conv2d, init, max_pool};

    let tape: Tape<f32> = Tape::new();
    let image = tape.input(init::normal(1, 1.0)(&Shape::new([1, 1, 6, 6])));
    let filters = tape.parameter(init::normal(2, 0.3)(&Shape::new([2, 1, 3, 3])));
    let bias = tape.parameter(Tensor::filled([2], 0.0_f32));
    let convolved = conv2d(image, filters, bias, 1, 0).relu();
    let pooled = max_pool(convolved, 2, 2);
    let loss = pooled.sum().symbol();
    let network = tape.into_network();

    let plan = network.entry([loss]).lower();
    let mut elected: Vec<String> = plan
        .patterns()
        .iter()
        .map(|pattern| {
            format!(
                "{:?} rooted at node {}, covering {} nodes, served as {:?}",
                pattern.kind(),
                pattern.root().index(),
                pattern.nodes().len(),
                pattern.formula()
            )
        })
        .collect();
    elected.insert(0, format!("{} patterns elected:", elected.len()));
    show::panels([
        ("plan", show::spec(plan.describe())),
        ("elected", show::text(elected.join("\n"))),
    ])
}
