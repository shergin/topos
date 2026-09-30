{
    use malevich::Theme;
    use topos::{Activation, Mlp, Tape, init};

    let tape: Tape<f32> = Tape::new();
    let _mlp = Mlp::new(&tape, &[3, 4, 2], Activation::Tanh, init::xavier(11));
    let _rate = tape.parameter(0.01_f32);
    let network = tape.into_network();
    show::card(network.parameters().to_html(Theme::DARK))
}
