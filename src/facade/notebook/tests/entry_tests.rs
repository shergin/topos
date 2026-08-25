use crate::{Numerics, Tape};
use malevich::Theme;

#[test]
fn the_card_names_roots_observes_and_posture() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.0);
    let hidden = w * w;
    let loss = hidden.sum();
    let (hidden, loss) = (hidden.symbol(), loss.symbol());
    let network = tape.into_network();
    let html = network
        .entry([loss])
        .observe([hidden])
        .numerics(Numerics::Exact)
        .to_html(Theme::DARK);
    assert!(html.contains("entry"));
    assert!(html.contains("1 roots"));
    assert!(html.contains("1 observes"));
    assert!(html.contains("Exact"));
    assert!(html.contains("forward-only"));
    assert!(html.contains(&format!("#{}", loss.id.index())));
}

#[test]
fn a_backward_entry_names_the_retain_posture() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.0);
    let loss = (w * w).sum().symbol();
    let network = tape.into_network();
    let html = network.entry([loss]).backward().to_html(Theme::DARK);
    assert!(html.contains("retain for backward"));
}
