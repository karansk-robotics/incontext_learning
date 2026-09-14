#!/usr/bin/env bash
# Build cyclo_py inside the ROS 2 Jazzy container, where cyclo_control's own
# dependencies resolve. On the bare host, ldd on libcyclo_motion_controller_core.so
# reports 6 unresolved libraries (pinocchio, coal, urdfdom, osqp, OsqpEigen), so
# this will not build outside the container.
#
#   ./container.sh bash -lc 'aiworker_icrt/cyclo_bindings/build.sh'
#
# Produces cyclo_py*.so next to this script; add the directory to PYTHONPATH.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [ ! -d /opt/ros ]; then
    echo "error: no /opt/ros -- run this inside the container, not on the host." >&2
    exit 1
fi

python3 -c 'import pybind11' 2>/dev/null || pip install --no-cache-dir pybind11

cmake -S "$HERE" -B "$HERE/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCYCLO_INSTALL="$ROOT/third_party/cyclo_control/install" \
    -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build "$HERE/build" -j"$(nproc)"
cp "$HERE"/build/cyclo_py*.so "$HERE/"

echo
echo "built: $(ls "$HERE"/cyclo_py*.so)"
PYTHONPATH="$HERE" python3 -c "
import cyclo_py as cy
print('import OK')
print('  KinematicsSolver:', [m for m in dir(cy.KinematicsSolver) if not m.startswith('_')][:6], '...')
print('  controller      :', [m for m in dir(cy.AIWorkerBimanualMoveLController) if not m.startswith('_')][:4], '...')
"
