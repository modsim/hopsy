#!/bin/bash

docker run -d -t --name hopsy_manylinux_build_env -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64

for py in "cp310-cp310" "cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39";
do 
    docker exec hopsy_manylinux_build_env /bin/bash -c  \
        "/opt/python/"$py"/bin/pip wheel /io -w /io/dist/"
    docker exec hopsy_manylinux_build_env /bin/bash -c  \
        "auditwheel repair /io/dist/*"$py"*.whl -w /io/dist"
done

docker stop hopsy_manylinux_build_env
docker rm hopsy_manylinux_build_env



