#!/bin/sh

SUDO="sudo -H"

# install most dependencies via apt-get
${SUDO} apt-get -y update
${SUDO} apt-get -y upgrade
# We explicitly set the C++ compiler to g++, the default GNU g++ compiler. This is
# needed because we depend on system-installed libraries built with g++ and linked
# against libstdc++. In case `c++` corresponds to `clang++`, code will not build, even
# if we would pass the flag `-stdlib=libstdc++` to `clang++`.
${SUDO} apt-get -y install g++ cmake pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev
export CXX=g++
export MAKEFLAGS="-j `nproc`"

${SUDO} apt-get -y install python3-dev python3-pip
${SUDO} apt-get -y install libboost-all-dev
# install additional python dependencies via pip
${SUDO} pip3 install  pyplusplus
${SUDO} pip3 install  pygccxml==2.2.1
# install castxml
${SUDO} apt-get -y install castxml
${SUDO} apt-get -y install libboost-python-dev
${SUDO} apt-get -y install libboost-numpy-dev python${PYTHONV}-numpy
${SUDO} pip3 install -vU pygccxml pyplusplus
#install pypy3
${SUDO} add-apt-repository ppa:pypy/ppa
${SUDO} apt-get update
${SUDO} apt-get -y install pypy3

# # #install ompl
# git clone https://github.com/ompl/ompl.git
# cd ompl
# # 改用1.5.2版本就好了
# git checkout tags/1.5.2
# mkdir -p build/Release
# cd build/Release
# cmake ../..
# make -j 20 update_bindings
# make -j 20
# sudo make install
git clone -b 1.5.2 https://github.com/ompl/ompl.git
cd ompl
mkdir -p build/Release
cd build/Release
cmake ../..
make -j 20 update_bindings
make -j 20
sudo make install
# cp -r /usr/lib/python3.8/site-packages/* /usr/lib/python3/dist-packages/

# change py-bindings/generate_bindings.py line 194
# try:
#     self.ompl_ns.class_(f'SpecificParam< std::string >').rename('SpecificParamString')
# except:
#     self.ompl_ns.class_(f'SpecificParam< std::basic_string< char > >').rename('SpecificParamString')

# pip3 install pygccxml==2.2.1

