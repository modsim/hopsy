conda activate py3.8
python -m pip install -r requirements-dev.txt
python -m pip uninstall --yes hopsy && python setup.py bdist_wheel
python -m pip install --no-input --user dist/hopsy-1.5.3-cp38-cp38-win_amd64.whl --force-reinstall --no-deps
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
conda activate py3.9 && python -m pip uninstall --yes hopsy
python -m pip install -r requirements-dev.txt
python setup.py bdist_wheel
python -m pip install --no-input --user dist/hopsy1.5.3-cp39-cp39-win_amd64.whl --force-reinstall  --no-deps
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
conda activate py3.10
python -m pip install -r requirements-dev.txt
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input --user dist/hopsy-1.5.3-cp310-cp310-win_amd64.whl --force-reinstall  --no-deps
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
conda activate py3.11
python -m pip install -r requirements-dev.txt
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input --user dist/hopsy-1.5.3-cp311-cp311-win_amd64.whl --force-reinstall  --no-deps
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
conda activate py3.12
python -m pip install -r requirements-dev.txt
python -m pip uninstall --yes hopsy
python setup.py bdist_wheel
python -m pip install --no-input --user dist/hopsy-1.5.3-cp312-cp312-win_amd64.whl --force-reinstall  --no-deps
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
