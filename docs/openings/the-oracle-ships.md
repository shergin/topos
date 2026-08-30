# The oracle ships

There is a slow, obvious way to run the graph. We keep it.
Everything faster is a claim about those bits. A backend
that cannot match them is declined, not trusted.

We kept the slow path because it is the spec. The surprise
is what that spec is for once you hand the plan to someone
else.

## The decision

The interpreter's bits are the truth. Reordering float math
is a named choice (`Numerics::Fast`), never a silent effect
of turning a feature on. `Exact` restores the reference
bits in the same process, in every build, including ones
that compiled a GPU kernel.

Two identical runs cannot differ. There is no `rand` and
no clock in the graph. A rerun is a checksum.

## What it opened

A plan is already a closed function. `Plan::emit_stablehlo`
writes it as text. An industrial runtime — XLA today,
outside this crate — takes the text from there. We do not
link their compiler. Printing the plan is the same kind of
thing as `describe`.

The slow run still checks the other side of that boundary.
Where float physics forbids bit-identity, the check is an
envelope, and a miss is reported, not skipped.

On a language model the in-crate run, the in-crate plan,
and XLA on CPU produce the same text. The same printed
module through a vendor GPU plugin can run faster and be
wrong. The stack is small enough to read, and strict
enough to put the disagreement on that plugin.

Catching a backend was not why we kept the interpreter.
It is what a spec is for once the plan leaves the crate.

## Spelled today

`Network::forward` always uses `Numerics::Exact`.
`interpret` uses the posture the entry asked for. A plan
matches the interpreter bit for bit wherever only exact
kernels serve. When
`TOPOS_STABLEHLO_VALIDATOR` / `TOPOS_STABLEHLO_EVALUATOR`
point at a toolchain, emitted modules are parsed and
checked against the plan. The language-model example
documents a vendor GPU row as wrong rather than hiding
it.

This section may rot; the rest must not.

## Not this

- Deleting the interpreter once a backend is fast.
- A feature flag that silently reorders sums.
- Treating emission as a second compiler that is allowed
  to disagree without a name.

See [Vision](../vision.md) rule 3, [Acceleration](../acceleration.md)
for `Exact` / `Fast`, and [Fusion raises](fusion-raises.md) for
why the printed module still has industrial ops to run.
