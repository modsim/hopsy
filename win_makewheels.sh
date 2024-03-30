conda activate py3.8
python -m pip uninstall --yes hopsy && python setup.py bdist_wheel
python -m pip install --no-input  dist/hopsy-1.5.0-cp38-cp38-win_amd64.whl --force-reinstall
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda activate py3.9 && python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input dist/hopsy-1.5.0-cp39-cp39-win_amd64.whl --force-reinstall
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda activate py3.10
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input dist/hopsy-1.5.0-cp310-cp310-win_amd64.whl --force-reinstall
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda activate py3.11
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input  dist/hopsy-1.5.0-cp311-cp311-win_amd64.whl --force-reinstall
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda activate py3.12
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input  dist/hopsy-1.5.0-cp311-cp311-win_amd64.whl --force-reinstall
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
