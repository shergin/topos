{
    use topos::{Shape, Tape, Tensor, causal_mask, init, scaled_dot_product};

    let tape: Tape<f32> = Tape::new();
    let query = tape.input(init::normal(1, 1.0)(&Shape::new([4, 8])));
    let key = tape.input(init::normal(2, 1.0)(&Shape::new([4, 8])));
    let value = tape.input(init::normal(3, 1.0)(&Shape::new([4, 8])));
    // The mask and the scale are caller-supplied values: a facade
    // never chooses a float constant.
    let mask = tape.leaf(causal_mask(4, f32::NEG_INFINITY));
    let scale = tape.leaf(Tensor::from(1.0 / 8f32.sqrt()));
    let _head = scaled_dot_product(query, key, value, mask, scale);
    show::spec(tape.describe())
}
