use crate::Tape;
use malevich::Theme;

#[test]
fn the_card_lists_each_pair_in_wrt_order() {
    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let loss = (a * b).sum();
    let adjoints = tape.differentiate(loss, [a, b]);
    let html = adjoints.to_html(Theme::DARK);
    assert!(html.contains("adjoints"));
    assert!(html.contains("2 pairs"));
    assert!(html.contains(&format!(
        "#{}  →  #{}",
        a.symbol().id.index(),
        adjoints.of(a.symbol()).id.index()
    )));
    assert!(html.contains("target"));
}

#[test]
fn rendering_is_deterministic() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.0);
    let loss = (w * w).sum();
    let adjoints = tape.differentiate(loss, [w]);
    assert_eq!(adjoints.to_html(Theme::DARK), adjoints.to_html(Theme::DARK));
}
