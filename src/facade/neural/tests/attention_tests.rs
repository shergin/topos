use crate::{Detach, Tape, Tensor, causal_mask, scaled_dot_product};

/// A small dense payload with distinct dyadic values.
fn dense(shape: impl Into<crate::Shape>, seed: usize) -> Tensor<f32> {
    let shape = shape.into();
    let volume = shape.volume();
    Tensor::new(
        shape,
        (0..volume)
            .map(|index| ((index * 7 + seed * 3) % 16) as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn the_facade_matches_the_hand_rolled_spelling_bitwise() {
    let build = |facade: bool| {
        let (network, [output]) = Tape::record(|tape| {
            let query = tape.leaf(dense([2, 3], 1));
            let key = tape.leaf(dense([4, 3], 2));
            let value = tape.leaf(dense([4, 3], 3));
            let mask = tape.leaf(Tensor::filled([2, 4], 0.0_f32));
            let scale = tape.leaf(Tensor::filled([], 0.5_f32));
            let output = if facade {
                scaled_dot_product(query, key, value, mask, scale)
            } else {
                let scores = query.matmul(key.transpose());
                let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
                weights.matmul(value)
            };
            [output].detach()
        });
        let run = network.forward(&network.parameters(), []);
        run.of(output).to_vec()
    };

    let facade = build(true);
    let hand_rolled = build(false);
    assert_eq!(facade.len(), hand_rolled.len());
    for (facade, hand_rolled) in facade.iter().zip(&hand_rolled) {
        assert_eq!(facade.to_bits(), hand_rolled.to_bits());
    }
}

#[test]
fn the_causal_mask_is_zero_at_and_below_the_diagonal() {
    let mask = causal_mask::<f32>(3, f32::NEG_INFINITY);
    assert_eq!(mask.shape(), crate::Shape::new([3, 3]));
    let expected = [
        0.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        0.0,
    ];
    for (element, expected) in mask.iter().zip(expected) {
        assert_eq!(element.to_bits(), expected.to_bits());
    }
}

#[test]
#[should_panic(expected = "equal shapes")]
fn a_mask_of_the_wrong_shape_panics_at_the_recording() {
    let tape: Tape<f32> = Tape::new();
    let query = tape.leaf(dense([2, 3], 1));
    let key = tape.leaf(dense([4, 3], 2));
    let value = tape.leaf(dense([4, 3], 3));
    let mask = tape.leaf(Tensor::filled([3, 3], 0.0_f32));
    let scale = tape.leaf(Tensor::filled([], 0.5_f32));
    scaled_dot_product(query, key, value, mask, scale);
}
