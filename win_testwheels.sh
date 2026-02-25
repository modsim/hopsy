#!/usr/bin/env bash
set -euo pipefail

ENVS=(py3.10 py3.11 py3.12 py3.13 py3.14)

mkdir -p logs

run_one () {
  local env="$1"
  local log="logs/${env}.log"

  {
    echo "=============================="
    echo "ENV: $env"
    echo "=============================="
    prefix="$(micromamba env list | awk -v e="$env" '$1==e {print $NF; exit}')"
    py="$prefix/python.exe"
    "$py" --version
    "$py" -c "import hopsy; print(hopsy.__version__, getattr(hopsy,'__build__', None))"
    "$py" -m unittest tests
    #micromamba run -n "$env" python --version
    #micromamba run -n "$env" python -c "import hopsy; print(hopsy.__version__, getattr(hopsy,'__build__', None))"
    #micromamba run -n "$env" python -m unittest tests
    echo "OK: $env"
  } >"$log" 2>&1
}

# Kick off all envs in parallel
pids=()
for env in "${ENVS[@]}"; do
  # Skip envs that don't exist
  if ! micromamba env list | awk '{print $1}' | grep -qx "$env"; then
    echo "SKIP (missing env): $env"
    continue
  fi

  echo "Starting tests in $env (log: logs/${env}.log)"
  run_one "$env" &
  pids+=($!)
done

# Wait and collect statuses
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo
echo "=============================="
echo "Test summary"
echo "=============================="

# Print per-env result by scanning logs
for env in "${ENVS[@]}"; do
  log="logs/${env}.log"
  if [ -f "$log" ]; then
    if grep -q "^OK: $env" "$log"; then
      echo "PASS: $env"
    else
      echo "FAIL: $env  (see $log)"
      # show a small tail to help quickly
      tail -n 40 "$log" || true
      echo
    fi
  fi
done

exit "$fail"
