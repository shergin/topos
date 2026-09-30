{
    use malevich::Theme;
    use topos::{Activation, Mlp, Module, Shape, Tape, Tensor, init};

    let tape: Tape<f32> = Tape::new();
    let mlp = Mlp::new(&tape, &[4, 8, 8, 1], Activation::Tanh, init::xavier(3));
    let features: Tensor<f32> = init::normal(5, 1.0)(&Shape::new([16, 4]));
    let loss = (mlp.express(tape.input(features))).sum().symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(loss);
    show::card(gradients.to_html(Theme::DARK))
}
