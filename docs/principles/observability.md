# Observability is a license

What you may read is declared. Everything derived runs on that
declaration, not on the whole tape.

## Why

A bag of nodes with a late request reconstructs the function
after the fact. The interpreter then evaluates unused twins and
hopes you only look at what you meant. A plan that infers
readability from liveness lets an interior leak because a buffer
happened to survive. A fusion that ignores a declared interior
erases a value the caller named.

Without a declaration, optimization has no license: it cannot
know what must survive, and it cannot know what is fair to drop.

## The idea

A sealed spec is a compilation unit. It may export several
functions that share weights — train and sample, full context and
decode. Each export is a declared reading: the results that must
be computed, the extra interiors the caller will also read, and
the postures (memory, numerics) the run is under.

Say it once. Every executor honors that keep-set. The interpreter
is the oracle over the ancestor closure; a plan is a derived
schedule of the same closure. A read outside the declaration
fails loudly. An interior that liveness happened to retain is
still unreadable — the contract does not depend on the
optimizer's choices.

The observed must survive any derived artifact. The unobserved is
fair game: skip it, free it, fuse it away. A declared interior
inside a would-be fused chain is a barrier. Fusion is an offer
over the unobserved; it is not a rewrite of something the caller
named.

Construction declares something too, and it is a different
thing. The names you detach from recording are the ones later
phases can mention. They are not "the only entry." Twins
record together; later declarations pick which export to run.
Rule 2 at record time means "names you will mention later," not
"compute everything you named on every run."

## Consequences

- Observability is never inferred from what a pass still holds.
- Adding a name to the declaration can refuse a fusion; that is
  the license working, not a regression.
- Result order is declared. Emission returns exactly those
  values in exactly that order.
- Whole-spec evaluation is a debug path for when you really do
  want every node. It is not the training API.
- The two declarations stay distinct: construction detaches the
  vocabulary of later mentions; an entry names what this run may
  read. Only the second is a keep-set.

## Not this

- Inferring the keep-set from live buffers.
- A fusion that claims a node the caller asked to observe.
- One implied function per spec. Twins are several exports, one
  architecture.
- Making the detached names the execution keep-set. You may
  name a sampling head you do not compute on the training entry.

See [Vision](../vision.md) rule 2, [Names](names.md) for how names
leave construction, and [Spec and state](spec-and-state.md) for
why several entries can share one frozen architecture.

## Spelled today

`Entry` is the declared reading: roots, observes, memory posture,
numerics. `BoundEntry::interpret` and `BoundEntry::lower` are the
two executors. `Run::of` panics outside the mask.
`Tape::record`'s return is `Detach`. `Network::forward` still
evaluates the whole spec. An unnamed keep-set node inside a
pattern is a fusion barrier; a named result (batch-norm's
statistics) is written back instead. This section may rot; the
rest must not.

See [Several exports, one spec](../openings/several-exports.md)
for the twins and the emission ABI.
