#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${REPO_ROOT}/docs/_static"

generate-schema-doc \
    --config template_name=js \
    --config expand_buttons=True \
    --config footer_show_time=False \
    "${REPO_ROOT}/schemas/v1/annotation.json" \
    "${REPO_ROOT}/docs/_static/schema_v1.html"
