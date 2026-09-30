{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let constant = tape.leaf(3.0);
    let weight = tape.parameter(0.5);
    let sample = tape.input(1.0);
    let _activation = (weight * sample + constant).tanh();
    show::spec(tape.describe())
}
