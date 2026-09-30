{
    use malevich::Theme;
    use topos::{Shape, Tensor, init};

    let weights: Tensor<f32> = init::xavier(7)(&Shape::new([4, 6]));
    show::card(weights.to_html(Theme::DARK))
}
