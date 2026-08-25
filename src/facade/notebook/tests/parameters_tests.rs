use crate::{Tape, Tensor};
use malevich::Theme;

#[test]
fn the_card_lists_each_slot_shape() {
    let tape: Tape<f64> = Tape::new();
    let _bias = tape.parameter(0.5);
    let _weights = tape.parameter(Tensor::new([2, 2], vec![1.0, 0.0, 0.0, 1.0]));
    let parameters = tape.into_network().parameters();
    let html = parameters.to_html(Theme::DARK);
    assert!(html.contains("2 slots"));
    assert!(html.contains("[]"));
    assert!(html.contains("0.5"));
    assert!(html.contains("[2, 2]"));
}

#[test]
fn an_empty_table_says_so() {
    let parameters = Tape::<f64>::new().into_network().parameters();
    assert!(parameters.to_html(Theme::DARK).contains("no parameters"));
}

#[test]
fn a_long_table_elides_the_middle() {
    let tape: Tape<f64> = Tape::new();
    for _ in 0..40 {
        let _ = tape.parameter(0.0);
    }
    let html = tape.into_network().parameters().to_html(Theme::DARK);
    assert!(html.contains("40 slots"));
    assert!(html.contains("more slots"));
}
