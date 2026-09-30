{
    use malevich::Theme;
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[8, 16, 4], Activation::Tanh, init::xavier(3));
    let features: Tensor<f32> = init::normal(5, 1.0)(&Shape::new([32, 8]));
    let loss = (mlp.express(tape.input(features))).sum();
    let adjoints = tape.differentiate(loss, mlp.parameters());
    let network = tape.into_network();
    let plan = network.entry(adjoints.roots()).lower();
    show::card(plan.to_html(Theme::DARK))
}
