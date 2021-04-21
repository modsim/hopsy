FROM ubuntu:20.04
LABEL Maintainer="Richard D. Paul <r.paul@fz-juelich.de>"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential software-properties-common cmake libeigen3-dev \
    liblpsolve55-dev lp-solve libxerces-c-dev libhdf5-dev doxygen libncurses5-dev libncursesw5-dev \
    libsbml5-dev mpich libmpich-dev git
RUN apt-get install -y bzip2 libbz2-dev
RUN apt-get install -y coinor-clp coinor-libclp-dev
RUN apt-get install -y git
RUN apt-get install -y python3-distutils
RUN apt-get update -y
RUN apt-get install -y python3-pip

WORKDIR /root

#ADD . /root/hopsy

RUN mkdir /root/.ssh/
ADD ssh/known_hosts /root/.ssh/known_hosts
ADD ssh/id_rsa /root/.ssh/id_rsa

RUN pip3 install numpy
RUN git clone --recursive git@jugit.fz-juelich.de:fluxomics/hopsy.git

RUN pip3 install ./hopsy
RUN python3 hopsy/tests/test.py

#RUN mkdir hopsy/cmake-build

#WORKDIR /home/hopsy/cmake-build
#RUN cmake .. 
#RUN make -j4
