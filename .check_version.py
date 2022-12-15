from packaging import version as v

with open('.version', 'r') as f:
    version = f.readline().split('\n')[0]

conform = str(v.parse(version))

if conform != version:
    raise RuntimeError('Version number ' + version + ' is not PEP conform, please change to ' + conform)

