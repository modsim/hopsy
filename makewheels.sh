#!/bin/bash

docker run -d -t --name hopsy_manylinux_build_env -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64

docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "/opt/python/cp310-cp310/bin/pip wheel /io -w /io/dist/"
docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "/opt/python/cp36-cp36m/bin/pip wheel /io -w /io/dist/" 
docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "/opt/python/cp37-cp37m/bin/pip wheel /io -w /io/dist/" 
docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "/opt/python/cp38-cp38/bin/pip wheel /io -w /io/dist/" 
docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "/opt/python/cp39-cp39/bin/pip wheel /io -w /io/dist/" 
docker exec hopsy_manylinux_build_env /bin/bash -c  \
    "auditwheel repair /io/dist/*whl -w /io/dist"

docker stop hopsy_manylinux_build_env
docker rm hopsy_manylinux_build_env
