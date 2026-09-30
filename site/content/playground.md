# Playground

Write a small graph and read it six ways. The engine on this page is the crate compiled to WebAssembly: your program records onto a real `Tape`, and every tab is the reading the library answers for it — the spec, the gradient `differentiate` appends, the plan an entry lowers to, the StableHLO it emits, the values a run computes, and the picture. Change a line and every reading remakes itself.

{{playground}}

## The tape language

A program is a list of assignments, one per line. Each line records one or more nodes on the tape; a name is the symbol of the node it names. The language exists only on this page: it is a thin spelling of the calls `Tape` and `Value` already offer, so every reading it produces is one the crate produces.

| line | records |
|---|---|
| `w = parameter [2, 2]` | a trainable parameter of that shape, seeded deterministically |
| `x = input [1, 2]` | an input with a seeded default, fed per run |
| `c = leaf [3]` | a constant |
| `b = parameter 0.5` | a rank-0 parameter holding that value; `input 1.5` and `leaf 2.0` likewise |
| `y = a + b`, `a - b`, `a * b`, `a / b`, `-a` | elementwise arithmetic; a rank-0 operand is broadcast explicitly, and the spec shows it |
| `y = a @ b` | the matrix product |
| `tanh(a)`, `exp(a)`, `ln(a)`, `sqrt(a)`, `sin(a)`, `cos(a)`, `relu(a)` | maps |
| `sum(a)` | the sum of every element, rank 0 |
| `sum_along(a, 1)`, `log_softmax(a, 1)`, `logsumexp(a, 1)`, `softmax(a, 1)`, `mean_along(a, 1)` | one axis |
| `transpose(a)` | the two axes of a matrix, swapped |
| `broadcast(a, [3, 4])`, `reshape(a, [12])` | shape changes |
| `2.0` | a numeric literal is a rank-0 leaf |
| `# ...` | a comment |

Choose which scalar to differentiate and which interior to observe from the controls; the parameters are the `wrt` set, and the differentiate target must be rank 0 — `sum` it first if it is not. A shape that does not fit is refused before the crate sees it, with the two shapes in the message, the same way the recording line panics in Rust.

## What to try

- Record the README's loss, `w * x - y` squared, and watch the derivative append four Leaf-then-Add pairs: one gradient started at zero, one consumer's share added at a time. That is accumulation, stated once in the engine.
- Add `h = tanh(x @ w)` as a second line and observe it. One line of the plan turns from `freed` to `kept`; nothing else moves.
- Turn on exact numerics. On this page no backend is ever on, so the plan's values do not change, and that is the point: `Exact` is the interpreter's bits in every build.
- Write `log_softmax(x @ w, 1)` and read the StableHLO: the stable composition an industrial compiler receives, one fused primitive and the plain arithmetic around it.
