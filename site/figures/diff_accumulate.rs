{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(0.0);
    let x = tape.input(0.0);
    let y = tape.input(0.0);
    let error = w * x - y;
    let loss = error * error;

    let spec_nodes = tape.len();
    tape.differentiate(loss, [w]);
    show::graph(Graph::from_nodes(tape.nodes()).classify_from(spec_nodes, "derived"))
}
