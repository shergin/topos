{
    use topos::{Tape, Tensor};

    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(Tensor::new([3], [0.5, -0.25, 1.5]));
    let cubes = x * x * x;

    // A gradient needs a scalar target; a VJP takes any target and an
    // explicit seed of the same shape. This seed picks the first row
    // of the Jacobian.
    let seed = tape.leaf(Tensor::new([3], [1.0, 0.0, 0.0]));
    let adjoints = tape.vjp(cubes, seed, [x]);
    let (x, cubes) = (x.symbol(), cubes.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    show::lines([
        format!("x           = {}", run.of(x)),
        format!("x^3         = {}", run.of(cubes)),
        "seed        = [1, 0, 0]".to_string(),
        format!("J^T seed    = {}   (3 x^2 in the seeded row, zero elsewhere)", run.of(adjoints.of(x))),
    ])
}
