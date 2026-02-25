set -euo pipefail

PKG="hopsy"

build_one () {
  local env="$1"
  echo "=============================="
  echo "Building in env: $env"
  echo "=============================="

  # uninstall any existing install in that env (ignore errors)
  micromamba run -n "$env" python -m pip uninstall --yes "$PKG" || true

  # dev requirements needed for build/test
  micromamba run -n "$env" python -m pip install -r requirements-dev.txt

  # (optional but recommended) clean previous build artifacts
  rm -rf build/ *.egg-info

  # build wheel
  micromamba run -n "$env" python setup.py bdist_wheel

  # pick the newest wheel produced
  local wheel
  wheel="$(ls -1t dist/*.whl | head -n 1)"
  echo "Installing wheel: $wheel"

  # install wheel for user, no deps, force reinstall
  micromamba run -n "$env" python -m pip install --no-input --user "$wheel" --force-reinstall --no-deps

  # verify import + version/build
  micromamba run -n "$env" python -c "import $PKG; print($PKG.__version__, getattr($PKG, '__build__', None))"
}

build_one py3.10
build_one py3.11
build_one py3.12
build_one py3.13
build_one py3.14
