use crate::{Detach, Embedding, Module, Tape, Tensor, named_parameters};

/// A one-hot `[count, vocab]` selection payload.
fn one_hot(rows: &[usize], vocabulary: usize) -> Tensor<f32> {
    Tensor::selection(rows.to_vec(), vocabulary, 1.0)
}

fn table() -> Tensor<f32> {
    Tensor::new(
        [4, 3],
        (0..12).map(|index| index as f32 / 4.0).collect::<Vec<_>>(),
    )
}

#[test]
fn the_lookup_matches_the_hand_rolled_gather_bitwise() {
    let selection = one_hot(&[2, 0, 2], 4);
    let build = |facade: bool| {
        let (network, [rows]) = Tape::record(|tape| {
            let rows = if facade {
                let embedding = Embedding::new(tape, table());
                embedding.express(tape.leaf(selection.clone()))
            } else {
                let table = tape.parameter(table());
                table.gather(tape.leaf(selection.clone()))
            };
            [rows].detach()
        });
        let run = network.forward(&network.parameters(), []);
        run.of(rows).to_vec()
    };

    let facade = build(true);
    let hand_rolled = build(false);
    assert_eq!(facade.len(), hand_rolled.len());
    for (facade, hand_rolled) in facade.iter().zip(&hand_rolled) {
        assert_eq!(facade.to_bits(), hand_rolled.to_bits());
    }
}

#[test]
fn the_table_is_visited_as_weights() {
    let tape: Tape<f32> = Tape::new();
    let embedding = Embedding::new(&tape, table());
    let named = named_parameters(&embedding);
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].0.to_string(), "weights");
    assert_eq!(named[0].1, embedding.table());
    assert_eq!(
        embedding.parameters().collect::<Vec<_>>(),
        vec![embedding.table()]
    );
}

#[test]
fn tying_reads_through_the_exposed_symbol() {
    // A tied head is a matmul with the resolved table, not a second
    // module: the symbol is the tie.
    let (network, [logits]) = Tape::record(|tape| {
        let embedding = Embedding::new(tape, table());
        let rows = embedding.express(tape.leaf(one_hot(&[1, 3], 4)));
        let logits = rows.matmul(tape.resolve(embedding.table()).transpose());
        [logits].detach()
    });
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(logits).shape(), crate::Shape::new([2, 4]));
}

#[test]
#[should_panic(expected = "must be rank 2")]
fn a_rank_one_table_panics() {
    let tape: Tape<f32> = Tape::new();
    Embedding::new(&tape, Tensor::filled([4], 0.0_f32));
}
