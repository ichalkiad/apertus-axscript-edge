#!/usr/bin/env bash
# Assemble a model_axscript tree with the Apertus port applied.
#
#   scripts/assemble_tree.sh <voyager-sdk checkout> <dest>
#
# Copies <sdk>/model_axscript to <dest>/model_axscript, copies this repository's
# model_axscript/ over it, and applies patches/register-apertus.patch (the two registry
# tuples the SDK requires for a new model). <dest> can then go on PYTHONPATH.
# The SDK checkout must be at the tag the patch was made against (v1.9.0).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <voyager-sdk checkout> <dest>" >&2
  exit 2
fi
sdk=$(cd "$1" && pwd)
dest=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -d "$sdk/model_axscript" ]]; then
  echo "error: $sdk has no model_axscript/ (expected a voyager-sdk checkout)" >&2
  exit 1
fi
if [[ -e "$dest/model_axscript" ]]; then
  echo "error: $dest/model_axscript already exists; remove it first" >&2
  exit 1
fi

mkdir -p "$dest"
cp -r "$sdk/model_axscript" "$dest/model_axscript"
cp -r "$here/model_axscript/." "$dest/model_axscript/"
(cd "$dest" && patch -p1 --forward --quiet < "$here/patches/register-apertus.patch")
find "$dest/model_axscript" -name __pycache__ -prune -exec rm -rf {} +
echo "assembled $dest/model_axscript (sdk: $sdk)"
