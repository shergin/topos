{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.5);
    let square = (w * w).symbol();
    let w = w.symbol();
    let network = tape.into_network();
    let sealed = network.len();

    // Reopening consumes the network; the old symbols still resolve.
    let tape = network.into_tape();
    let _cube = tape.resolve(square) * tape.resolve(w);
    show::spec_from(tape.describe(), sealed, "appended after the reopen")
}
