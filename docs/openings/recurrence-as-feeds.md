# Recurrence as feeds

A transformer can generate the next token without re-running the
whole window. Each layer keeps last step's keys and values, and
only computes the new row. That cache has to live somewhere.

A natural place would be inside the network. That is the place
we refused. The surprise is that the cache still has somewhere
to go.

## The decision

If the graph also holds the live weights, every training step
makes a new graph. Then a name has to say which graph it
belongs to.

The fix is in [spec and state](../principles/spec-and-state.md).
The graph is the architecture. It does not change when you
train. The weights are a table you own. A training step
rewrites that table and leaves the graph alone.

## What it opened

Once weights live in a table you pass in, any per-run data can
live that way. A cache is not special. It is an extra input.
This run writes it; the next run reads it.

So decode is a second formula on the same graph:

- in: one token, plus last step's keys and values
- out: the next-token scores, plus the updated keys and values

The new rows are written with `scatter`, the adjoint of the
`gather` embeddings already use. There is no cache instruction and no hidden
buffer. The "run the whole window every time" formula stays on
the tape as a check. At a toy size the two formulas match bit
for bit. On a real model they produce the same text.

Decode was not why we split spec from state. It became easy
because the carry already had a place to sit: the same place a
batch sits.

Trying two learning rates is the same pattern. Copy the weight
table. Do not copy the graph.

## Spelled today

`Network` cannot be cloned. `Parameters::step` builds the next
weight table. `parameters.clone()` is a what-if.

The language-model examples record two formulas on one tape:
the full window, and one-token decode. Each layer's keys and
values are ordinary inputs. A step feeds them in and reads
them back.

This section may rot; the rest must not.

## Not this

- A new opcode for caches. The carry is an input, like a batch.
- Storing batch-norm running averages in the graph. That
  policy belongs to the caller, like every other extra table.
- Cloning the graph to try a learning rate. Clone the weights.

See [Spec and state](../principles/spec-and-state.md) for the
rule, and [Several exports, one spec](several-exports.md) for
how two formulas share one graph.
