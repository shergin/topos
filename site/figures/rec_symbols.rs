{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(0.25);
    let doubled = w * 2.0;
    // Values borrow the tape; symbols are the detached names.
    let (w, doubled) = (w.symbol(), doubled.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    show::lines([
        format!("w.index()          = {}", w.index()),
        format!("doubled.index()    = {}", doubled.index()),
        format!("parameters.of(w)   = {}", parameters.of(w)),
        format!("run.of(doubled)    = {}", run.of(doubled)),
        format!("network.node(w)    = {}", network.node(w).to_string().trim()),
    ])
}
