.PHONY: build

all:
	make clean
	make build
	make install
	make test

build:
	pip3 wheel --no-deps . -w dist/

install:
	pip3 install --force-reinstall --no-deps dist/hopsy-*-linux*.whl

test:
	python3 -m unittest -v tests

clean:
	rm -rf build/
	rm -rf dist/

manylinux:
	make clean
	make manylinux-build
	make manylinux-install
	make test

manylinux-build:
	./makewheels.sh

manylinux-install:
	pip3 install --force-reinstall --no-deps dist/hopsy-*-manylinux*.whl

