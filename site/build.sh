#!/bin/sh
# Build the site into site/dist: the generator first (pages and plates), then
# the wasm the playground uses. Serve with:
#   python3 -m http.server 4173 --directory site/dist
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
out="$root/site/dist"
cd "$root"
cargo run --release -q -p topos-site -- --out "$out"
wasm-pack build site/wasm --target web --out-dir "$out/wasm" --release
rm -f "$out/wasm/.gitignore" "$out/wasm/README.md" "$out/wasm/.npmignore"
# The glue is ESM; GitHub Pages must not treat it as CommonJS.
printf '%s\n' '{"type":"module"}' > "$out/wasm/package.json"
echo "built $out"
