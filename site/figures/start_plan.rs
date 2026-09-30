{
    use topos::{Detach, Tape};

    let (network, [loss]) = Tape::record(|tape| {
        let w = tape.parameter(0.0_f64);
        let x = tape.input(0.0);
        let y = tape.input(0.0);
        let error = w * x - y;
        [error * error].detach()
    });
    let plan = network.entry([loss]).lower();
    show::spec(plan.describe())
}
