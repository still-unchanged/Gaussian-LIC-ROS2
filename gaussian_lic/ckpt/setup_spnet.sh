#!/usr/bin/env bash
set -eo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$repo/env.sh" "$@"
