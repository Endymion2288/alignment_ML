#!/usr/bin/env bash
# Compile the independent leave-target-out helper against the existing
# Calypso/Athena build.  Does not modify Calypso CMake or call
# KalmanFitterTool.fit.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
src_dir="$project_root/alignment/leave_target_out_dump"
out_dir="$project_root/build/leave_target_out_dump"
calypso_root="${CALYPSO_ROOT:-$project_root/../calypso}"
compile_json="$calypso_root/build/compile_commands.json"
link_txt="$calypso_root/build/PhysicsAnalysis/NtupleDumper/CMakeFiles/NtupleDumper.dir/link.txt"
merge_script="$calypso_root/run/cmake/modules/scripts/mergeConfdb2.py"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "setup_environment.sh is absent" >&2
  exit 2
fi
if [[ ! -f "$compile_json" ]]; then
  echo "Calypso compile_commands.json is absent: $compile_json" >&2
  exit 2
fi
if [[ ! -f "$link_txt" ]]; then
  echo "NtupleDumper link.txt is absent: $link_txt" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" calypso
set -u

mkdir -p "$out_dir/obj" "$out_dir/genConf"
export LD_LIBRARY_PATH="$out_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - "$compile_json" "$link_txt" "$src_dir" "$out_dir" "$project_root" "$calypso_root" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

compile_json, link_txt, src_dir, out_dir, project_root, calypso_root = map(Path, sys.argv[1:])
commands = json.loads(compile_json.read_text())
ntd = next(
    item for item in commands
    if str(item.get("file", "")).endswith("NtupleDumperAlg.cxx")
)
parts = shlex.split(ntd["command"])
filtered = []
skip_next = False
for part in parts:
    if skip_next:
        skip_next = False
        continue
    if part in {"-c", "-o"}:
        skip_next = True
        continue
    if part.endswith("NtupleDumperAlg.cxx") or part.endswith(".o"):
        continue
    if part.startswith("-DNtupleDumper_EXPORTS") or part.startswith("-DATLAS_PACKAGE_NAME"):
        continue
    if part.startswith("-fplugin="):
        continue
    filtered.append(part)
compiler = filtered[0]
flags = filtered[1:]
flags += [
    '-DATLAS_PACKAGE_NAME="CkfLeaveTargetOutDump"',
    "-DCkfLeaveTargetOutDump_EXPORTS",
    f"-I{src_dir}",
    f"-I{project_root}",
    f"-I{calypso_root}/Tracking/Acts/FaserActsKalmanFilter",
    f"-I{calypso_root}/Tracking/Acts/FaserActsKalmanFilter/FaserActsKalmanFilter",
    f"-I{calypso_root}/Tracking/Acts/FaserActsKalmanFilter/src",
    f"-I{calypso_root}/Tracking/Acts/FaserActsGeometry",
]
sources = [
    src_dir / "CkfLeaveTargetOutDumpAlg.cxx",
    src_dir / "CkfLeaveTargetOutDump_entries.cxx",
]
objects = []
for source in sources:
    obj = out_dir / "obj" / (source.stem + ".o")
    objects.append(obj)
    cmd = [compiler, *flags, "-c", str(source), "-o", str(obj)]
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

link_line = link_txt.read_text().strip().splitlines()[0]
link_parts = shlex.split(link_line)
gxx = None
libs = []
for part in link_parts:
    if part.endswith("/g++") or part.endswith("/g++.bin") or part == "g++":
        gxx = part
        continue
    if part.endswith("libNtupleDumper.so"):
        continue
    if part.endswith(".o"):
        continue
    if part.startswith("-fplugin="):
        continue
    if part.startswith("-o"):
        continue
    if Path(part).name == "atlas_build_run.sh":
        continue
    if gxx is None and not part.startswith("-") and "g++" not in part:
        continue
    if part == gxx:
        continue
    libs.append(part)
if gxx is None:
    raise SystemExit("g++ was not found in the NtupleDumper link line")
so = out_dir / "libCkfLeaveTargetOutDump.so"
link_cmd = [gxx, "-shared", "-o", str(so), *[str(obj) for obj in objects], *libs]
print("+", shlex.join(link_cmd), flush=True)
subprocess.run(link_cmd, check=True)
print(f"linked {so}", flush=True)
PY

cat > "$out_dir/libCkfLeaveTargetOutDump.components" <<'EOF'
v2::libCkfLeaveTargetOutDump.so:CkfLeaveTargetOutDumpAlg
EOF

genconf_bin="$(command -v genconf || true)"
if [[ -z "$genconf_bin" ]]; then
  echo "genconf is not on PATH after Calypso setup" >&2
  exit 2
fi
ext_lib="/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/24.0.41/InstallArea/x86_64-el9-gcc13-opt/lib"
export LD_LIBRARY_PATH="$out_dir:$ext_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! (
  cd "$out_dir"
  "$genconf_bin" -o "$out_dir/genConf" -p CkfLeaveTargetOutDump --no-init \
    -i libCkfLeaveTargetOutDump.so
); then
  if [[ -f "$out_dir/libCkfLeaveTargetOutDump.so" &&
        -f "$out_dir/libCkfLeaveTargetOutDump.components" &&
        -f "$out_dir/CkfLeaveTargetOutDump.confdb2" ]]; then
    echo "genconf failed; keeping the existing configurable database" >&2
  else
    echo "genconf failed and no existing configurable database is present" >&2
    exit 2
  fi
fi
part="$(find "$out_dir/genConf" -name '*.confdb2_part' | head -n 1)"
if [[ -z "$part" ]]; then
  echo "genconf did not write a confdb2_part" >&2
  exit 2
fi
parts_file="$out_dir/genConf/parts.txt"
printf '%s\n' "$part" > "$parts_file"
python "$merge_script" "$out_dir/CkfLeaveTargetOutDump.confdb2" "$parts_file"
printf 'Built independent LTO helper in %s\n' "$out_dir"
ls -l "$out_dir/libCkfLeaveTargetOutDump.so" \
  "$out_dir/libCkfLeaveTargetOutDump.components" \
  "$out_dir/CkfLeaveTargetOutDump.confdb2"
