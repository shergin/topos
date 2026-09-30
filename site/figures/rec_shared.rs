{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let a = tape.leaf(2.0);
    let b = tape.leaf(3.0);
    let c = tape.leaf(4.0);
    let sum = a + b;
    let product = sum * c;
    let _expression = -product + a * c;
    show::graph(Graph::from_nodes(tape.nodes()))
}
