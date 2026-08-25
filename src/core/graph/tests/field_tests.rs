use crate::{Tape, Tensor};

#[test]
fn algebra_combines_elementwise() {
    let tape = Tape::new();
    let a = tape.leaf(2.0_f64);
    let b = tape.leaf(3.0);
    let product = a * b;
    let sum = a + b;
    let (a, b, product, sum) = (a.symbol(), b.symbol(), product.symbol(), sum.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let d_product = run.backward(product);
    let d_sum = run.backward(sum);

    let combined = &d_product + &d_sum;
    assert_eq!(combined.of(a).scalar(), 4.0);
    assert_eq!(combined.of(b).scalar(), 3.0);

    let result = combined.scale(&2.0.into());
    assert_eq!(result.of(a).scalar(), 8.0);

    let squared = d_product.zip(&d_product, |left, right| left.clone() * right.clone());
    assert_eq!(squared.of(a).scalar(), 9.0);

    let shifted = d_sum.map(|value| value.clone() + Tensor::from(1.0));
    assert_eq!(shifted.of(b).scalar(), 2.0);
}

#[test]
#[should_panic(expected = "fields belong to different networks")]
fn combination_rejects_foreign_networks() {
    let first = Tape::new();
    let second = Tape::new();
    let a = first.leaf(1.0_f64).symbol();
    let b = second.leaf(1.0).symbol();

    let first = first.into_network();
    let second = second.into_network();
    let field_first = first.forward(&first.parameters(), []).backward(a);
    let field_second = second.forward(&second.parameters(), []).backward(b);

    let _ = &field_first + &field_second;
}

#[test]
fn projected_fields_step_later_parameter_states() {
    let tape = Tape::new();
    let w = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let run = network.forward(&parameters, []);
    let gradients = run.backward(w);
    assert_eq!(gradients.of(w).scalar(), 1.0);

    // A projected direction is detached state: it still steps
    // parameter states minted after the run it came from.
    let direction = gradients.parameters(&parameters);
    let stepped = parameters.step(&direction, |parameter, direction| {
        parameter.clone() - direction.clone()
    });
    assert_eq!(stepped.of(w).scalar(), 0.0);
    let again = stepped.step(&direction, |parameter, direction| {
        parameter.clone() - direction.clone()
    });
    assert_eq!(again.of(w).scalar(), -1.0);
}
