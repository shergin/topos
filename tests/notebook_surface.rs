//! The notebook weld: every datum a notebook card displays is
//! derivable from the public surface alone.
//!
//! The facade gate (`facade_surface.rs`) lints source text for
//! private module paths, but a `pub(crate)` method call names no
//! module and slips through — the shape by which the notebook tier
//! once held privileged reads. This suite compiles as an external
//! consumer, so it can only name public items: it rebuilds each
//! card's data from the public readers and asserts the rendered
//! card agrees. A weld, not a proof — it pins the current cards,
//! and a new card datum needs a new line here.
#![cfg(feature = "evcxr")]

use malevich::Theme;
use topos::{Detach, Numerics, Tape};

#[test]
fn the_entry_card_data_is_public() {
    let (network, [weight, loss]) = Tape::record(|tape| {
        let weight = tape.parameter(1.0_f64);
        [weight, weight * weight].detach()
    });

    let bound = network
        .entry([loss])
        .observe([weight])
        .backward()
        .numerics(Numerics::Exact);
    assert_eq!(bound.entry().roots, [loss]);

    let entry = bound.into_entry();
    assert_eq!(entry.roots, [loss]);
    assert_eq!(entry.observe, [weight]);
    assert!(entry.backward);
    assert_eq!(entry.numerics, Numerics::Exact);

    let card = entry.to_html(Theme::DARK);
    assert!(card.contains(&format!("#{}", loss.index())));
    assert!(card.contains(&format!("#{}", weight.index())));
    assert!(card.contains("retain for backward"));
    assert!(card.contains("Exact"));
}

#[test]
fn the_position_a_symbol_answers_is_the_describe_number() {
    let (network, [weight, loss]) = Tape::record(|tape| {
        let weight = tape.parameter(1.0_f64);
        [weight, weight * weight].detach()
    });

    assert_eq!(weight.index(), 0);
    assert_eq!(loss.index(), 1);

    // One describe line per node in allocation order, so the line at
    // a symbol's position describes that node.
    let described = network.describe();
    let loss_line = described
        .lines()
        .nth(loss.index())
        .expect("describe covers every recorded node");
    assert!(loss_line.contains("Mul"));
}

#[test]
fn the_adjoints_card_data_is_public() {
    let tape = Tape::new();
    let weight = tape.parameter(1.0_f64);
    let loss = weight * weight;
    let adjoints = tape.differentiate(loss, [weight]);

    let card = adjoints.to_html(Theme::DARK);
    assert!(card.contains(&format!("#{}", adjoints.target().index())));
    for &(wrt, gradient) in adjoints.pairs() {
        assert!(card.contains(&format!("#{}", wrt.index())));
        assert!(card.contains(&format!("#{}", gradient.index())));
    }
}

#[test]
fn the_parameters_card_data_is_public() {
    let (network, [weight]) = Tape::record(|tape| {
        let weight = tape.parameter(2.5_f64);
        [weight].detach()
    });
    let parameters = network.parameters();
    assert_eq!(parameters.payloads().len(), parameters.len());

    // The scalar slot's displayed value is readable through the
    // constant-fill fast path or the element iterator, exactly as
    // the card computes it.
    let payload = parameters.of(weight);
    let value = match payload.as_constant() {
        Some(element) => *element,
        None => payload.iter().next().expect("a scalar holds one element"),
    };
    assert_eq!(value, 2.5);

    let card = parameters.to_html(Theme::DARK);
    assert!(card.contains("2.5"));
}
