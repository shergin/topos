#!/bin/sh
# Records the gallery: what each example prints into a pipe, verbatim, into
# site/gallery/NAME.txt. The site shows these plates beside the sources. Run
# it after `cargo build --release --examples`; the language models need the
# accelerate build and their cached checkpoints, so they come last.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
out="$root/site/gallery"
bin="$root/target/release/examples"
mkdir -p "$out"
cd "$root"
export NO_COLOR=1 COLUMNS=96 LINES=66

record() {
  name=$1
  shift
  started=$(date +%s)
  if "$bin/$name" "$@" > "$out/$name.txt" 2> "$out/$name.err"; then
    rm -f "$out/$name.err"
    echo "$name: $(( $(date +%s) - started ))s" >&2
  else
    echo "$name: FAILED (see $out/$name.err)" >&2
  fi
}

cargo build --release --examples -q
record chain
record walkthrough
record gradient_descent
record dual
record forward_mode
record element_seam
record regression
record mlp_xor
record moons
record throughput
record makemore_bigram
record makemore_mlp
record makemore_mlp_facade
record makemore_mlp_compiled
record makemore_mlp_adam
record makemore_mlp_parallel
record makemore_embedding_map
record makemore_mlp_batchnorm
record makemore_transformer
record mnist
record cifar10

# The language models: the accelerate build, and the checkpoints cached under
# ~/.cache/topos by an earlier run.
if [ "${TOPOS_RECORD_MODELS:-1}" = 1 ]; then
  cargo build --release --features accelerate --example gpt2 --example llama -q
  record gpt2 "Once upon a time" 40
  record llama "Once upon a time" 24
fi
echo "recorded into $out" >&2
