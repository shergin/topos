# Vision

Topos is an autodiff compiler stack. Record a graph, inspect
it, differentiate it, compile it, emit it. The spec is an immutable
value and the state is yours. The network never moves; a training
step is a pure data transform of caller-owned parameters.

The goal is a complete compiler whose first-principles design makes
it the easiest place to build the next thing. Better a few strict
rules than lots of features: that is how the next idea stays easy.
Once the graph is written, its sizes do not change.

That means three commitments:

- **The whole modern ML-compiler stack, small enough to read.**
  One spec, a short list of named interpretations of it. Every
  stage is visible and printable, none of it magic.
- **Every result provable.** The plain interpreter is the
  executable spec; anything faster must match it, bit for bit by
  default. A claim is one assert away from proof.
- **Built for learning, research, and production.** New ideas
  plug in at named seams, with the oracle as ground truth. The
  core stays closed and simple on purpose. Completeness is the
  point: the stack aims at the whole compiler.

A typical compiler rewrites the program until it becomes something
that can run. Topos does not. The tape is written once; everything
after it is a named way of reading the same spec. That list is the
compiler — small enough to read because there is nothing else.

- Recording writes the tape and infers every shape at the
  expression that records it. Sealing freezes that as the spec.
- The interpreter runs it. Those bits are the truth.
- Reverse mode is the same scan backward. The same derivative
  rules can also record themselves as ordinary nodes, so compiled
  training is a forward plan over that extra tape — fusion and
  liveness then apply to the chain rule itself.
- A plan is a schedule of what to run, what to keep, and what to
  free. It is derived from the spec and holds no state.
- Fusion is an offer a run may elect, never a rewrite of the tape.
- Emission writes the plan as text for an industrial backend. It
  is a sibling of `describe`, not a second compiler.

That list is also where new work lands. A new element type, a new
fusion, a new backend, a new emission target, a new AD mode —
each plugs in at a named seam and is proven against the
interpreter. The idea does not need a fork, and a hand-rolled
equivalent has the same standing as a built-in.

Adoption may follow; it is never chased. No benchmark races, no
coverage races, no plugin bazaar.

## The rules

Five rules, one axis each: what a program means, what may be
read, what is true, how a faster form earns its spelling, what
the core owns.

1. **The tape is the spec.** Recording writes it, sealing fixes
   it. Everything downstream — derivatives, plans, fusions,
   backends, emission — is a derived artifact: free to change the
   spelling, forbidden to change the meaning. Shapes are static
   and tapes are cheap — one tape per shape bucket, re-record
   rather than generalize. A plan holds no state, so it survives
   every training step and every reopen.
2. **Say what you want to read.** Observability is declared —
   roots and observes — never inferred, and a read outside the
   declaration fails loudly. The declaration is the license every
   optimization runs on: the observed must survive any rewrite;
   the unobserved is fair game to fuse away.
3. **The interpreter's bits are the truth.** The plain interpreter
   is the executable spec: whole-spec evaluation always walks the
   reference paths, so its bits are the same in every build, on
   every platform. Every plan, backend, and emitted module must
   reproduce those bits under the exact posture; a compiled
   backend serves only under the fast posture a plan or an entry
   declares. Seeded runs replay exactly; anything that reorders
   float math is that labeled option, never a silent one.
4. **Composition is the default.** A fused or native form buys
   its way in with a measured reason — float behavior or real
   cost. The primitives compose; specialized spelling is earned.
5. **The core is enough; facades prove it.** The core owns
   exactly what the crate exists to do. Every facade — layers,
   optimizers, losses — composes through the public surface
   alone, so a hand-rolled equivalent behaves identically and
   each facade stands as proof that the primitives suffice.

## Principles

Constraints the vision names without arguing. One file per
principle; the type names in each "Spelled today" section may
rot, the rest must not.

- [Recorded reverse mode](principles/recorded-reverse.md)
- [Spec and state](principles/spec-and-state.md)
- [Names](principles/names.md)
- [What earns an instruction](principles/vocabulary.md)
- [The element is the seam](principles/element.md)
- [Observability is a license](principles/observability.md)

## Openings

Decisions and what they paid for, in
[openings](openings/). Not constraints — consequences. An
opening may wait on a seam that is designed but not yet
collected.

- [AD as a named reading](openings/ad-as-a-reading.md)
- [Recurrence as feeds](openings/recurrence-as-feeds.md)
- [Several exports, one spec](openings/several-exports.md)
- [Fusion raises](openings/fusion-raises.md)
- [The oracle ships](openings/the-oracle-ships.md)
- [Dropout as a feed](openings/dropout-as-a-feed.md)
