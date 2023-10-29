#!/bin/bash

git rev-parse --short HEAD > .commit

docker ps -a | grep hopsy_manylinux_build_env && (docker stop hopsy_manylinux_build_env; docker rm hopsy_manylinux_build_env)

docker run -d -t --name hopsy_manylinux_build_env -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64

for py in "cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311";
do
#    docker exec hopsy_manylinux_build_env /bin/bash -c  \
#        "/opt/python/"$py"/bin/pip install -r /io/requirements.txt"
    docker exec hopsy_manylinux_build_env /bin/bash -c  \
        "/opt/python/"$py"/bin/pip wheel --no-deps /io -w /io/dist/"
    for whl in dist/*$py-linux*.whl;
    do
        docker exec hopsy_manylinux_build_env /bin/bash -c  \
            "auditwheel repair /io/"$whl" -w /io/dist"
    done
done

docker stop hopsy_manylinux_build_env
docker rm hopsy_manylinux_build_env

# upload with
#      python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
