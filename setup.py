# -*- coding: utf-8 -*-
import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    'win32': 'Win32',
    'win-amd64': 'x64',
    'win-arm32': 'ARM',
    'win-arm64': 'ARM64',
}

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('.version', 'r') as fh:
    version = fh.read().split('\n')[0]

with open('.commit', 'r') as fh:
    commit = fh.read().split('\n')[0]


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary 'native' libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = 'Debug' if self.debug else 'Release'

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get('CMAKE_GENERATOR', '')

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(extdir),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
            '-DHOPSY_VERSION_INFO={}'.format(self.distribution.get_version()),
            '-DHOPSY_BUILD_INFO={}'.format(commit),
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], cwd=self.build_temp
            ).decode('utf-8').split('\n')[0]
            print("Build id:", commit_hash)
            cmake_args.append('-DHOPSY_BUILD_ID={}'.format(commit_hash))
        except Exception as e:
            print("ERROR retrieving commit hash. No build ID will be set.")

        if self.compiler.compiler_type != 'msvc':
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                cmake_args += ['-GNinja']

        else:

            # Single config generators are handled 'normally'
            single_config = any(x in cmake_generator for x in {'NMake', 'Ninja'})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {'ARM', 'Win64'})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ['-A', PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)
                ]
                build_args += ['--config', cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, 'parallel') and self.parallel:
                # CMake 3.12+ only.
                build_args += ['-j{}'.format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name='hopsy',
    version=version,
    author='Richard D. Paul',
    author_email='r.paul@fz-juelich.de',
    description='A python interface for hops, the highly optimized polytope sampling toolbox.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        'Documentation': 'https://modsim.github.io/hopsy/',
        'GitHub': 'https://github.com/modsim/hopsy/',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
    ext_modules=[CMakeExtension('hopsy/core')],
    #ext_modules=[Extension('hopsy._hopsy', sources=['src/hopsy/hopsy.cpp'])],
    cmdclass={'build_ext': CMakeBuild},
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'anyio==3.6.1',
        'appdirs==1.4.4',
        'arviz==0.12.1',
        'certifi==2021.10.8',
        'cftime==1.6.0',
        'charset-normalizer==2.0.12',
        'cobra==0.25.0',
        'commonmark==0.9.1',
        'cycler==0.11.0',
        'depinfo==1.7.0',
        'diskcache==5.4.0',
        'fonttools==4.33.3',
        'future==0.18.2',
        'h11==0.12.0',
        'h5py==3.6.0',
        'hopsy==1.0.0b2',
        'httpcore==0.14.7',
        'httpx==0.22.0',
        'idna==3.3',
        'importlib-resources==5.7.1',
        'joblib==1.1.0',
        'kiwisolver==1.4.2',
        'matplotlib==3.5.2',
        'mpmath==1.2.1',
        'netCDF4==1.5.8',
        'numexpr==2.8.1',
        'numpy==1.22.3',
        'optlang==1.5.2',
        'packaging==21.3',
        'pandas==1.4.2',
        'Pillow==9.1.0',
        'PolyRound==0.1.8',
        'pydantic==1.9.0',
        'Pygments==2.12.0',
        'pyparsing==3.0.9',
        'python-dateutil==2.8.2',
        'python-libsbml==5.19.5',
        'pytz==2022.1',
        'rfc3986==1.5.0',
        'rich==12.4.1',
        'ruamel.yaml==0.17.21',
        'ruamel.yaml.clib==0.2.6',
        'scikit-learn==1.1.0',
        'scipy==1.8.0',
        'six==1.16.0',
        'sniffio==1.2.0',
        'swiglpk==5.0.5',
        'sympy==1.10.1',
        'tables==3.7.0',
        'threadpoolctl==3.1.0',
        'tqdm==4.64.0',
        'typing-extensions==4.2.0',
        'xarray==2022.3.0',
        'xarray-einstats==0.2.2',
        'zipp==3.8.0',
    ],
    zip_safe=False,
)
