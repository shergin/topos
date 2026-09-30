{
    use topos::{Tape, Tensor};

    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(Tensor::new([3], [0.5, -0.25, 1.5]));
    let value = (x * x * x).sum();

    // The first derivative is recorded as ordinary nodes...
    let first = tape.differentiate(value, [x]);
    let gradient = first.of(x.symbol());

    // ...so the second one is a VJP over the recorded gradient, seeded
    // with a direction: the Hessian-vector product `H v`.
    let second_starts = tape.len();
    let direction = tape.leaf(Tensor::new([3], [0.25, 1.0, -0.5]));
    let second = tape.vjp(gradient, direction, [x]);
    let hvp = second.of(x.symbol());

    let x = x.symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    show::panels([
        (
            "values",
            show::lines([
                format!("x                 = {}", run.of(x)),
                format!("gradient  3 x^2   = {}", run.of(gradient)),
                "direction v       = [0.25, 1, -0.5]".to_string(),
                format!("H v       6 x . v = {}", run.of(hvp)),
            ]),
        ),
        (
            "spec",
            show::spec_from(network.describe(), second_starts, "the second derivative, recorded over the first"),
        ),
    ])
}
