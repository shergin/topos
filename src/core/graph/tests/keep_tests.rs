use crate::{Keep, Symbol, Tape};

#[test]
fn record_returns_the_detached_keep_set() {
    let (network, [w, x, loss]) = Tape::record(|tape| {
        let w = tape.parameter(0.0_f64);
        let x = tape.input(1.0);
        let loss = (w * x) * (w * x);
        [w, x, loss].keep()
    });
    let parameters = network.parameters();
    assert_eq!(parameters.of(w).scalar(), 0.0);
    let run = network.forward(&parameters, []);
    assert_eq!(run.of(loss).scalar(), 0.0);
    let _: Symbol = x;
}

#[test]
fn a_bare_value_detaches_to_one_symbol() {
    let (network, w) = Tape::record(|tape| tape.parameter(2.0_f64).keep());
    let _: Symbol = w;
    assert_eq!(network.parameters().of(w).scalar(), 2.0);
}

#[test]
fn mixed_tuples_detach_member_by_member() {
    let (network, (w, gradients, kept)) = Tape::record(|tape| {
        let w = tape.parameter(3.0_f64);
        let loss = w * w;
        let adjoints = tape.differentiate(loss, [w]);
        (w, adjoints, vec![loss]).keep()
    });
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(kept[0]).scalar(), 9.0);
    assert_eq!(run.of(gradients.of(w)).scalar(), 6.0);
}

#[test]
fn an_empty_keep_set_still_seals() {
    let (network, ()) = Tape::record(|tape: &Tape<f64>| {
        tape.parameter(1.0_f64);
    });
    assert_eq!(network.len(), 1);
}

#[test]
fn a_keep_call_is_the_whole_ritual() {
    // The one call replaces the N-way `.symbol()` destructure, on an
    // open tape as well.
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.0_f64);
    let doubled = w * 2.0;
    let [w, doubled] = [w, doubled].keep();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(doubled).scalar(), 2.0);
    assert_eq!(network.parameters().of(w).scalar(), 1.0);
}
