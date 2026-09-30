# Entries and plans

## Say what you want to read

A sealed spec is a compilation unit, and it may export several functions that share weights: train and sample, full context and decode. An `Entry` is one such export, declared: the **roots** a run must compute, the extra interior values it will also **observe**, the **memory posture**, and the **numerics**.

```rust
let entry = network.entry([loss]).observe([hidden]).numerics(Numerics::Exact);
let run = entry.interpret(&parameters, []);   // the oracle over just those ancestors
let plan = entry.lower();                     // the compiled schedule
```

Say it once. Every executor honors that keep-set: the interpreter evaluates exactly the ancestors of the declared results; a plan is a derived schedule of the same closure. A read outside the declaration panics — an interior that liveness happened to retain is still unreadable, because the contract does not depend on the optimizer's choices. `Network::forward`, the whole-spec oracle, is the debug road for when you really do want every node.

## Lowering

`lower` derives a `Plan`: what to run, what to keep, what to free, which patterns to elect. It freezes a prefix of the spec and takes `Parameters` per call, so it holds no state, survives every training step, and outlives a reopen. `describe` prints one line per scheduled node with its liveness, then the summary.

{{figure plan_forward}}

## Observability is a license

The declaration is what every optimization runs on. The observed must survive any derived artifact; the unobserved is fair game to skip, free, or fuse away. Add one name to the declaration and one line changes.

{{figure plan_observe}}

Nodes an entry does not reach are not scheduled at all. When the gradient of the walkthrough is recorded with respect to the weights, the rules also produce the gradient with respect to the input — four nodes nobody declared, so a run never computes them and the emitted text never mentions them.

{{figure walk_skipped nocode}}

## Live volume

Every `freed after N` in a plan is a buffer release, and the summary's peak is the most elements alive at once. `Plan::live_series` is that accounting as a curve. A training plan's curve climbs while the forward values are still needed by the chain rule and falls as the backward half consumes them; an inference plan over the same graph stays flat and low.

{{figure plan_live}}

The numbers are the plan's own accounting in elements — constants and placeholders count as zero — not allocator truth. They are what the analysis licenses, which is exactly the number a design decision needs. The [notebook card](../notebooks/) for a plan draws this curve under the schedule.

## The backward posture

`Entry::backward()` declares runs that answer `Run::backward`: buffers retain what the engine reverse scan reads. It is a memory posture — the retain-all one, which the graded consumers preferred over freeing or rematerializing mid-run — not a second compiler. The column then says what the analysis *could* release, and the summary reports the retain-all total a run actually holds beside the release floor.

{{figure plan_backward}}

For compiled training, prefer recording the derivative and compiling a forward-only plan over `adjoints.roots()`: fusion and liveness then apply to the chain rule itself. `backward`'s place is the oracle, and quick procedural use.

## Patterns are offers

The catalog matches structures over the frozen columns — a windowed product, a window reduction, the two batch-norm formulas — and a plan elects the matches it can use. A match never rewrites the tape: at home it calls a fused kernel, and a region nobody elects just runs the original operations. A value you declared readable inside a match blocks it, unless the match names that value as a result and writes it back — which is how observing batch-norm's statistics survives the fusion.

{{figure plan_patterns}}

`Plan::patterns` is the elections as data. The same match has a second job abroad: when the plan is emitted, it becomes the other compiler's named op. [Fusion raises](../../openings/fusion-raises/) tells that story; [emission](../emission/) shows the text.

## Exact and Fast

Results are a function of the payloads, the compiled features, the numerics posture, and the machine. `Network::forward` is exact by construction: its bits are the same in every build, on every platform. A plan under `Numerics::Exact` computes those same bits in the same process, whatever backend features are compiled in. Under `Fast` — an entry's default — a backend may serve large products with sums in its own order; that is a labeled choice on the entry, never a silent effect of a feature flag.

{{figure plan_exact_fast}}

Which backend actually served is run-time data: wrap any region in `Backend::tallied` and read the per-formula tally. [Acceleration](../acceleration/) has the ladder and the numbers.

## Where next

- [Emission](../emission/) — the plan as text.
- [Observability is a license](../../principles/observability/) — the rule, and [several exports, one spec](../../openings/several-exports/), the twins it makes real.
