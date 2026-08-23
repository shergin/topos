use crate::{Adjoints, Symbol, Tape, Tensor};

/// Records a two-parameter loss and differentiates it, answering the
/// carrier beside the symbols the assertions name.
fn differentiated() -> (Adjoints, Symbol, Symbol, Symbol) {
    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(Tensor::new([2], [1.0_f64, -2.0]));
    let b = tape.parameter(Tensor::new([2], [0.5_f64, 3.0]));
    let loss = (a * b).sum();
    let (a, b, loss) = (a.symbol(), b.symbol(), loss.symbol());
    let adjoints = tape.differentiate(loss, [a, b]);
    (adjoints, loss, a, b)
}

#[test]
fn pairs_follow_wrt_order() {
    let (adjoints, _, a, b) = differentiated();
    assert_eq!(adjoints.wrt().collect::<Vec<_>>(), vec![a, b]);
    let pairs = adjoints.pairs();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, a);
    assert_eq!(pairs[1].0, b);
    assert_eq!(
        adjoints.gradients().collect::<Vec<_>>(),
        vec![pairs[0].1, pairs[1].1]
    );
}

#[test]
fn of_answers_each_entry() {
    let (adjoints, _, a, b) = differentiated();
    assert_eq!(adjoints.of(a), adjoints.pairs()[0].1);
    assert_eq!(adjoints.of(b), adjoints.pairs()[1].1);
}

#[test]
#[should_panic(expected = "not a `wrt` entry")]
fn of_rejects_a_symbol_that_was_not_differentiated() {
    let (adjoints, loss, _, _) = differentiated();
    adjoints.of(loss);
}

#[test]
fn roots_lead_with_the_target() {
    let (adjoints, loss, a, b) = differentiated();
    assert_eq!(adjoints.target(), loss);
    assert_eq!(
        adjoints.roots().collect::<Vec<_>>(),
        vec![loss, adjoints.of(a), adjoints.of(b)]
    );
}

#[test]
fn map_gradients_rewrites_gradients_and_keeps_the_pairing() {
    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(Tensor::new([2], [1.5_f64, -0.75]));
    let loss = (a * a).sum();
    let (a, loss) = (a.symbol(), loss.symbol());
    let adjoints = tape.differentiate(loss, [a]);
    // The emission consumers' aliasing move: a same-shape reshape per
    // gradient, pairing and target carried through the rewrite.
    let aliased = adjoints.map_gradients(|gradient| {
        let value = tape.resolve(gradient);
        value.reshape(value.shape()).symbol()
    });
    assert_eq!(aliased.target(), loss);
    assert_eq!(aliased.wrt().collect::<Vec<_>>(), vec![a]);
    assert_ne!(aliased.of(a), adjoints.of(a));

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(
        run.of(aliased.of(a)).to_vec(),
        run.of(adjoints.of(a)).to_vec()
    );
}
