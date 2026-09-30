{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(0.0);
    let x = tape.input(0.0);
    let y = tape.input(0.0);
    let error = w * x - y;
    let loss = error * error;

    let spec_nodes = tape.len();
    let adjoints = tape.differentiate(loss, [w]);
    let gradient = adjoints.of(w.symbol()).index();
    show::spec_from(
        tape.describe(),
        spec_nodes,
        &format!("recorded by differentiate; node {gradient} is d loss / d w"),
    )
}
