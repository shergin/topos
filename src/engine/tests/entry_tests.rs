use crate::{Entry, Tape};

#[test]
fn interpret_evaluates_the_declared_closure_and_nothing_else() {
    // Two twins on one tape: interpreting the training entry skips the
    // sampling twin entirely, and a read of the dropped twin is loud.
    let tape: Tape<f64> = Tape::new();
    let weight = tape.parameter(2.0_f64);
    let training = (weight * weight).symbol();
    let sampling = (weight * 3.0).symbol();
    let weight = weight.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let run = network.entry([training]).interpret(&parameters, []);
    assert_eq!(run.of(training).scalar(), 4.0);

    let full = network.forward(&parameters, []);
    assert_eq!(full.of(sampling).scalar(), 6.0);
    assert_eq!(run.of(training).scalar(), full.of(training).scalar());
    assert_eq!(run.backward(training).of(weight).scalar(), 4.0);
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn reads_outside_the_entry_are_rejected() {
    let tape: Tape<f64> = Tape::new();
    let weight = tape.parameter(2.0_f64);
    let training = (weight * weight).symbol();
    let sampling = (weight * 3.0).symbol();
    let network = tape.into_network();
    let run = network
        .entry([training])
        .interpret(&network.parameters(), []);
    run.of(sampling);
}

#[test]
fn lower_matches_the_detached_compile() {
    let tape: Tape<f64> = Tape::new();
    let weight = tape.parameter(1.5_f64);
    let loss = (weight * weight).sum().symbol();
    let weight = weight.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let bound = network.entry([loss]).backward().lower();
    let detached = network.compile(Entry::roots([loss]).backward());
    let bound_run = bound.forward(&parameters, []);
    let detached_run = detached.forward(&parameters, []);
    assert_eq!(bound_run.of(loss).scalar(), detached_run.of(loss).scalar());
    assert_eq!(
        bound_run.backward(loss).of(weight).scalar(),
        detached_run.backward(loss).of(weight).scalar()
    );
}

#[test]
fn a_detached_entry_survives_a_reopen() {
    let tape: Tape<f64> = Tape::new();
    let weight = tape.parameter(1.0_f64);
    let loss = (weight * weight).symbol();
    let network = tape.into_network();
    let entry = network.entry([loss]).into_entry();

    // Reopen, extend, reseal: the stored entry still lowers over the
    // grown spec's prefix.
    let tape = network.into_tape();
    let _late = tape.resolve(loss) * 2.0;
    let network = tape.into_network();
    let plan = network.compile(entry);
    let run = plan.forward(&network.parameters(), []);
    assert_eq!(run.of(loss).scalar(), 1.0);
}
