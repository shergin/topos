{
    use topos::Tape;

    let tape: Tape<f64> = Tape::new();
    let a = tape.leaf(2.0);
    let b = tape.leaf(3.0);
    let c = tape.leaf(4.0);
    let expression = -((a + b) * c) + a * c;
    let (a, b, c, expression) = (a.symbol(), b.symbol(), c.symbol(), expression.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(expression);
    show::lines([
        format!("-((a + b) * c) + a * c = {}", run.of(expression)),
        format!("d/da = -c + c = {}", gradients.of(a)),
        format!("d/db = -c     = {}", gradients.of(b)),
        format!("d/dc = -(a + b) + a = {}", gradients.of(c)),
    ])
}
