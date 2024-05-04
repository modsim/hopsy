conda activate py3.8
python --version
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
python -m unittest tests
conda deactivate
conda activate py3.9
python --version
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
conda deactivate
conda activate py3.10
python --version
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
python -m unittest tests
conda deactivate
conda activate py3.11
python --version
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
python -m unittest tests
conda deactivate
conda activate py3.12
python --version
python -c "import hopsy; print(hopsy.__version__, hopsy.__build__)"
python -m unittest tests
conda deactivate
