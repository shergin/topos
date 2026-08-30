# What earns an instruction

An instruction sits in the vocabulary iff something real speaks it
and no composition of the rest reproduces its bits.

## Why

A compiler that will record reverse mode cannot treat the op set
as a convenience menu. Every derivative rule expands into
instructions that themselves have rules. If that expansion leaves
the vocabulary, the transform has nowhere to write. If it stays
inside by duplicating an instruction that composition already
expresses in bits, the spec grows a synonym, dumps get noisier,
and two spellings of the same math start to drift.

The other failure is the kernel zoo: a faster or stabler form
lands as a new instruction because a facade wanted a name. Speed
and accuracy then hide in the IR instead of living where they
belong — accuracy in the vocabulary, speed in the catalog.

## The idea

One membership test, two clauses, both required:

- A real consumer records it, or a derivative rule speaks it.
- No composition of the remaining instructions reproduces its
  bits.

Bit-identity makes the second clause decidable. The first keeps
the set from growing "in case." Together they are the closed
core: small because redundant spellings are refused, complete
because adjoints and float-breaking formulas are not.

Adjoint pairing is the organizing structure. Reductions pair with
broadcasts, windows with their folds, gathers with scatters;
some instructions are self-adjoint; a one-way mask can terminate
the closure. Recorded reverse mode is possible because that
pairing is a theorem of the set, not a hope about the engine.

Composition is the default spelling. A formula moves down a tier
and earns an instruction only when floating point breaks the
composed form. A fused or native form that *preserves* bits is
not a new instruction — it is an offer the catalog may elect.
Accuracy lives in the vocabulary. Speed lives in the catalog.

A documented exception is allowed when the bit-test would make
the spec unreadable and no later tier recovers the cost. The
exception is the judgment, written down; it is not a second test.

## Consequences

- Adding an instruction is a breaking change of the IR, judged by
  the test, never by convenience.
- Changing a derivative rule that speaks a new instruction means
  the instruction earns its seat the same day, or the rule is
  restated in the existing set.
- Silently replacing a stable primitive with a composition that
  does not preserve bits is a different spec, not a rewrite.
- Unary elementwise transcendentals are one instruction kind
  parameterized by the map, not a new instruction per function.
- Facades compose through the public surface. Their names are not
  a reason to grow the vocabulary.

## Not this

- A kernel zoo. Faster spellings elect; they do not mint opcodes.
- Retiring an instruction that still has a consumer or a rule.
- Treating "we might need it" as clause (a).
- Opening the set so a dependent can register an instruction.
  New primitives land in the crate, under this test.

See [Recorded reverse mode](recorded-reverse.md) for why the set
must close under adjoints, and [Vision](../vision.md) for composition
as the default.

## Spelled today

`Opcode` / `Op` is the closed set. `Map` carries
`MapOperation`. `log_softmax` and `log_sum_exp` earned their
seats on bits; `relu` is `maximum` against a counted zero;
`Sub` is the documented exception on the bit clause (`Add` of
`Neg` is bit-exact). `Powf` is the exception on the consumer
clause: kept ahead of its consumers by decision — its
composition is neither bit-faithful nor defined on negative
bases, and the expected consumers (learned pooling exponents,
fractional robust-loss powers) are on file in the op-set audit.
This section may rot; the rest must not.

See [Fusion raises](../openings/fusion-raises.md) for the
catalog's second life as an idiom raiser.
