# Names

A name, a structural witness, and the right to extend the
recording are three jobs. They are not one identity.

## Why

It is tempting to stuff lineage, the whole structural map, a node
index, and who currently holds the pen into a single value that
"handles identity." Every handle then carries a chain it does not
read. Every training step pays for a path nobody walks. Equality
starts meaning too many things: same node, same family, same
prefix, same writer.

The fat name also fights the one property a node handle must have.
Names are copied constantly — into keep-sets, into entries, across
threads, across notebook cells. They have to stay cheap. A witness
of graph structure does not.

## The idea

Three jobs:

- **A name** answers "this node." It is detached, copyable, and
  thin: a family token plus a position. Nodes never move, so the
  position is stable for the life of that family. Equality of
  names is equality of node identity, nothing else.
- **A structural witness** answers "same family, and the same
  map of nodes, over this length?" Fields, plans, and runs ask
  this when they meet a name or another table. It is a read-only
  agreement over a prefix, not a name, and not whole-chain
  equality. Prefix agreement is the check; `Eq` is the wrong
  algebra.
- **Writer rights** answer "who may append?" That is mutation of
  the recording phase, not a property of a node. Once spec and
  state are split, the question almost stops being askable:
  there is one live recording or one sealed spec per family, by
  ownership, and training never appends. The right to extend is
  possessing the recording phase, not holding a richer id.

Construction handles are a fourth, narrower thing: a name plus a
borrow of the recording phase, so a proxy cannot outlive the
seal. They are the spelling of expressions. They are not the
currency of anything after.

## Consequences

- Names stay copyable and payload-free. They do not grow a chain,
  a generation, or a lock.
- A name from another family, or past the length a table covers,
  fails loudly. That is kinship as a check, not as a type the
  caller has to hold.
- Plans and runs do not borrow the spec. They carry enough to
  reject a foreign name and to serve their prefix after the
  spec reopens.
- Linear extension never moves a node, so a name taken before a
  reopen still names that node afterwards.
- Mixing two live recordings in one expression is a recording
  error. Lifetimes do not brand topologies; the check stays at
  the expression.

## Not this

- Merging family, chain, and index into one identity that also
  handles forking. Wrong granularity; fights cheap names; puts
  the writer protocol on the thing that means "node."
- Putting the structural map on every name.
- Treating name equality as "same graph prefix" or "same writer."
- A post-seal proxy. After the freeze, everything speaks names.
  The sealed spec has no proxy type; that absence is the design.

See [Spec and state](spec-and-state.md) for why writer rights
leave the public surface, and [Vision](../vision.md) for the tape
as spec.

## Spelled today

`Symbol` is origin plus index, `Copy`. `Value` borrows the
`Tape` and dies at `into_network`. `Origin` is crate-private;
dependents see `"belongs to a different network"` and coverage
panics. `Keep` is how names leave construction. This section
may rot; the rest must not.
