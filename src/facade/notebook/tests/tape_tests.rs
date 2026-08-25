use crate::Tape;
use malevich::Theme;

#[test]
fn the_card_renders_the_ir_dump() {
    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let _sum = a + b;
    let html = tape.to_html(Theme::DARK);
    for line in tape.describe().lines() {
        assert!(
            html.contains(line.trim_end()),
            "card is missing the spec line {line:?}"
        );
    }
    assert!(html.contains("3 nodes"));
}

#[test]
fn an_empty_tape_still_has_a_summary() {
    let tape: Tape<f64> = Tape::new();
    assert!(tape.to_html(Theme::DARK).contains("0 nodes"));
}

#[test]
fn rendering_is_deterministic_and_theme_aware() {
    let tape: Tape<f64> = Tape::new();
    let _leaf = tape.parameter(1.0);
    assert_eq!(tape.to_html(Theme::DARK), tape.to_html(Theme::DARK));
    assert!(tape.to_html(Theme::DARK).contains("#0d1117"));
    assert!(tape.to_html(Theme::LIGHT).contains("#ffffff"));
}
