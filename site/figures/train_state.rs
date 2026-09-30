{
    use topos::{Detach, Tape, Tensor};

    let (network, [w, x, y, loss]) = Tape::record(|tape| {
        let w = tape.parameter(0.0_f64);
        let x = tape.input(0.0);
        let y = tape.input(0.0);
        let error = w * x - y;
        [w, x, y, error * error].detach()
    });
    let before = network.describe();
    let mut parameters = network.parameters();
    let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
    for step in 0..100 {
        let (sample_x, sample_y) = samples[step % samples.len()];
        let run = network.forward(&parameters, [(x, sample_x.into()), (y, sample_y.into())]);
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = parameters.step(&gradients, |w, g| {
            w.clone() - g.clone() * Tensor::from(0.02)
        });
    }
    assert_eq!(network.describe(), before);
    show::lines([
        format!("network.describe() unchanged after 100 steps: {}", network.describe() == before),
        format!("network.parameters().of(w)   {}   (the recorded initial, materialized fresh)", network.parameters().of(w)),
        format!("parameters.of(w)             {}   (the table the loop stepped)", parameters.of(w)),
        format!("parameters.clone()           a what-if, costing {} payloads", parameters.len()),
    ])
}
