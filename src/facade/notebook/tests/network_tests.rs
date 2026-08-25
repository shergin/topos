use super::*;
use crate::Tape;

#[test]
fn the_card_renders_the_ir_dump() {
    let empty: Network<f64> = Tape::new().into_network();
    assert!(empty.to_html(Theme::DARK).contains("0 nodes"));

    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let _sum = a + b;
    let network = tape.into_network();
    let html = network.to_html(Theme::DARK);
    // The card is the same text `describe` answers, line for line.
    for line in network.describe().lines() {
        assert!(
            html.contains(line.trim_end()),
            "card is missing the spec line {line:?}"
        );
    }
    assert!(html.contains("3 nodes"));
}

#[test]
fn one_node_reads_in_the_singular() {
    let tape: Tape<f64> = Tape::new();
    let _only = tape.parameter(1.0);
    let network = tape.into_network();
    assert!(network.to_html(Theme::DARK).contains("1 node"));
}

#[test]
fn a_long_spec_elides_the_middle_of_the_dump() {
    let tape: Tape<f64> = Tape::new();
    for _ in 0..90 {
        let _ = tape.parameter(0.0);
    }
    let network = tape.into_network();
    let html = network.to_html(Theme::DARK);
    assert!(html.contains("more lines"));
    assert!(html.contains("90 nodes"));
}

#[test]
fn rendering_is_deterministic_and_theme_aware() {
    let tape: Tape<f64> = Tape::new();
    let _leaf = tape.parameter(1.0);
    let network = tape.into_network();
    assert_eq!(network.to_html(Theme::DARK), network.to_html(Theme::DARK));
    assert!(network.to_html(Theme::DARK).contains("#0d1117"));
    assert!(network.to_html(Theme::LIGHT).contains("#ffffff"));
}
