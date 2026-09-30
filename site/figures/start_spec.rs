{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(0.0);
    let x = tape.input(0.0);
    let y = tape.input(0.0);
    let error = w * x - y;
    let _loss = error * error;
    show::spec(tape.describe())
}
