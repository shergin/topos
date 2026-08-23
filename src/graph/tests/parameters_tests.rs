use crate::{Tape, Tensor};

#[test]
fn parameters_materialize_the_record_site_initials() {
    let tape = Tape::new();
    let weight = tape.parameter(1.5_f64).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    assert_eq!(parameters.len(), 1);
    assert_eq!(*parameters.of(weight), 1.5);

    // Every call answers an independent state: stepping one leaves a
    // fresh materialization at the initials.
    let run = network.forward(&parameters, []);
    let gradients = run.backward(weight).parameters(&parameters);
    let stepped = parameters.step(&gradients, |parameter, gradient| parameter - gradient);
    assert_eq!(*stepped.of(weight), 0.5);
    assert_eq!(*network.parameters().of(weight), 1.5);
}

#[test]
fn step_each_passes_the_parameter_symbol() {
    let tape = Tape::new();
    let first = tape.parameter(1.0_f64).symbol();
    let second = tape.parameter(2.0).symbol();
    let loss = (tape.resolve(first) * tape.resolve(second)).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss).parameters(&parameters);
    let stepped = parameters.step_each(&gradients, |symbol, current, direction| {
        if symbol == first {
            current - direction
        } else {
            *current
        }
    });
    assert_eq!(*stepped.of(first), -1.0);
    assert_eq!(*stepped.of(second), 2.0);
}

#[test]
fn cloned_states_diverge_independently() {
    let tape = Tape::new();
    let weight = tape.parameter(0.0_f64).symbol();
    let loss = (tape.resolve(weight) * tape.resolve(weight)).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    let fast = parameters.clone();
    let run = network.forward(&fast, []);
    let gradients = run.backward(loss).parameters(&fast);
    let fast = fast.step(&gradients, |parameter, _| parameter + 1.0);

    assert_eq!(*fast.of(weight), 1.0);
    assert_eq!(*parameters.of(weight), 0.0);
}

#[test]
#[should_panic(expected = "symbol belongs to a different network")]
fn of_rejects_foreign_symbols() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let parameters = network.parameters();

    let foreign = Tape::new().parameter(1.0_f64).symbol();
    parameters.of(foreign);
}

#[test]
#[should_panic(expected = "does not name a parameter")]
fn of_rejects_non_parameter_symbols() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let constant = tape.leaf(2.0).symbol();
    let network = tape.into_network();
    network.parameters().of(constant);
}

#[test]
#[should_panic(expected = "direction belongs to a different network")]
fn step_rejects_foreign_directions() {
    let first = Tape::new();
    let weight = first.parameter(1.0_f64).symbol();
    let first = first.into_network();
    let state = first.parameters();
    let direction = first
        .forward(&state, [])
        .backward(weight)
        .parameters(&state);

    let second = Tape::new();
    second.parameter(1.0_f64);
    let second = second.into_network();
    second
        .parameters()
        .step(&direction, |parameter, _| *parameter);
}

#[test]
#[should_panic(expected = "direction covers different parameter slots")]
fn step_rejects_directions_over_different_slots() {
    let tape = Tape::new();
    let weight = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let direction = network
        .forward(&parameters, [])
        .backward(weight)
        .parameters(&parameters);

    // Reopen and record one more parameter: the old direction does not
    // cover the new slot, and stepping the carried state with it must
    // be loud.
    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    let carried = parameters.carried(&network);
    carried.step(&direction, |parameter, _| *parameter);
}

#[test]
fn carried_keeps_payloads_and_seeds_new_slots() {
    let tape = Tape::new();
    let old = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let initial = network.parameters();
    let run = network.forward(&initial, []);
    let gradients = run.backward(old).parameters(&initial);
    let parameters = initial.step(&gradients, |parameter, gradient| parameter + gradient);
    assert_eq!(*parameters.of(old), 2.0);

    let tape = network.into_tape();
    let new = tape.parameter(7.0).symbol();
    let network = tape.into_network();

    let carried = parameters.carried(&network);
    assert_eq!(carried.len(), 2);
    assert_eq!(*carried.of(old), 2.0);
    assert_eq!(*carried.of(new), 7.0);
}

#[test]
#[should_panic(expected = "parameters do not cover this network")]
fn forward_rejects_uncarried_parameters() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let parameters = network.parameters();

    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    network.forward(&parameters, []);
}

#[test]
fn with_payloads_replaces_named_parameters() {
    let tape = Tape::new();
    let kept = tape.parameter(Tensor::new([2], [1.0_f64, 2.0])).symbol();
    let replaced = tape.parameter(Tensor::new([2], [3.0, 4.0])).symbol();
    let network = tape.into_network();

    let parameters = network
        .parameters()
        .with_payloads([(replaced, Tensor::new([2], [9.0, 9.0]))]);
    assert_eq!(parameters.of(kept).to_vec(), &[1.0, 2.0]);
    assert_eq!(parameters.of(replaced).to_vec(), &[9.0, 9.0]);
}

#[test]
#[should_panic(expected = "must preserve the parameter's shape")]
fn with_payloads_rejects_shape_changes() {
    let tape = Tape::new();
    let weight = tape.parameter(Tensor::new([2], [1.0_f64, 2.0])).symbol();
    let network = tape.into_network();
    network
        .parameters()
        .with_payloads([(weight, Tensor::new([3], [1.0, 2.0, 3.0]))]);
}

#[test]
fn algebra_combines_tables_entry_by_entry() {
    let tape = Tape::new();
    let first = tape.parameter(2.0_f64).symbol();
    let second = tape.parameter(3.0).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let doubled = parameters.map(|value| value * 2.0);
    assert_eq!(*doubled.of(first), 4.0);
    assert_eq!(*doubled.of(second), 6.0);

    let summed = &parameters + &doubled;
    assert_eq!(*summed.of(first), 6.0);
    assert_eq!(*summed.of(second), 9.0);

    let scaled = parameters.scale(&0.5);
    assert_eq!(*scaled.of(first), 1.0);
    assert_eq!(*scaled.of(second), 1.5);

    let least = parameters.zip(&doubled, |left, right| left.min(*right));
    assert_eq!(*least.of(first), 2.0);
    assert_eq!(*least.of(second), 3.0);
}

#[test]
#[should_panic(expected = "parameter tables belong to different networks")]
fn zip_rejects_foreign_tables() {
    let first = Tape::new();
    first.parameter(1.0_f64);
    let first = first.into_network().parameters();

    let second = Tape::new();
    second.parameter(1.0_f64);
    let second = second.into_network().parameters();

    first.zip(&second, |left, _| *left);
}

#[test]
#[should_panic(expected = "parameter tables cover different slots")]
fn zip_rejects_tables_over_different_slots() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let narrow = network.parameters();

    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    let wide = network.parameters();

    narrow.zip(&wide, |left, _| *left);
}

#[test]
fn projection_reads_every_parameter_out_of_a_field() {
    let tape = Tape::new();
    let weights = tape.parameter(Tensor::new([2], [1.0_f64, -2.0]));
    let bias = tape.parameter(Tensor::new([2], [0.5_f64, 3.0]));
    let loss = ((weights * bias).sum() * (weights * bias).sum()).symbol();
    let (weights, bias) = (weights.symbol(), bias.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();

    let field = network.forward(&parameters, []).backward(loss);
    let projected = field.parameters(&parameters);
    assert_eq!(projected.len(), parameters.len());
    for symbol in [weights, bias] {
        assert_eq!(projected.of(symbol).to_vec(), field.of(symbol).to_vec());
    }
}

#[test]
#[should_panic(expected = "field belongs to a different network")]
fn projection_rejects_foreign_fields() {
    let first = Tape::new();
    let weight = first.parameter(1.0_f64).symbol();
    let first = first.into_network();
    let field = first.forward(&first.parameters(), []).backward(weight);

    let second = Tape::new();
    second.parameter(1.0_f64);
    let second = second.into_network();
    field.parameters(&second.parameters());
}

#[test]
#[should_panic(expected = "field is stale")]
fn projection_rejects_stale_fields_after_extension() {
    let tape = Tape::new();
    let weight = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let field = network.forward(&parameters, []).backward(weight);

    // Reopen and record one more parameter: the old field does not
    // cover the new slot, so projecting onto the carried table must
    // be loud.
    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    let carried = parameters.carried(&network);
    field.parameters(&carried);
}
