# Several exports, one spec

The same model can train on a batch and generate a token.
Those two jobs share weights. They should not compute the
same nodes. Train does not need the sampling head. Decode
does not need the full window.

The list of values this run may read is what makes that
split real.

## The decision

[Observability is a license](../principles/observability.md):
say what you want to read, once. Everything derived — the
slow run, the plan, fusion, emission — honors that list. A
read off the list fails loudly. A name you asked to see,
sitting inside a chain that could have been fused, blocks
the fusion — unless the match names it as a result it
writes back. Optimization knows what must survive and what
it may drop.

## What it opened

One graph can export several functions. You record them
together. Each run names which export it is.

The names you keep from recording are the ones later code
may mention. They are not "compute all of these every
time." The training function can skip the sampling head.
The decode function can skip the full window.

Printing the plan for another compiler uses the same list,
in the same order. The plan is already a closed function
with those results. Emission does not invent an output
convention. It writes the list down.

[Recurrence as feeds](recurrence-as-feeds.md) is how a
decode function carries last step's keys and values. This
article is why decode is its own function, not a mode of
the full-window one.

## Spelled today

`Entry` is that list: results, extra values to observe,
memory and numerics choices. `network.entry([loss])` binds
it. `interpret` runs the slow path over just those
ancestors. `lower` builds the plan. `Plan::results` is the
list `emit_stablehlo` returns. `Run::of` panics on a name
this run did not compute. `Network::forward` still runs
the whole graph. That is the debug path, not training.

This section may rot; the rest must not.

## Not this

- Guessing the list from buffers that happened to survive.
  If you did not ask to read it, it is not readable.
- One implied function per graph.
- Treating every name you detached at record time as
  something this run must compute.

See [Observability is a license](../principles/observability.md)
for the rule, and [Fusion raises](fusion-raises.md) for the
barrier that list is.
