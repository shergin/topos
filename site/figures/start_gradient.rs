{
    use topos::{Detach, Tape};

    let (network, [w, x, y, loss]) = Tape::record(|tape| {
        let w = tape.parameter(0.5_f64);
        let x = tape.input(2.0);
        let y = tape.input(4.0);
        let error = w * x - y;
        [w, x, y, error * error].detach()
    });
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
    show::lines([
        format!("loss = (w x - y)^2           = {}", run.of(loss)),
        format!("d loss / d w = 2 (w x - y) x = {}", gradients.of(w)),
        format!("d loss / d x = 2 (w x - y) w = {}", gradients.of(x)),
        format!("d loss / d y = -2 (w x - y)  = {}", gradients.of(y)),
    ])
}
