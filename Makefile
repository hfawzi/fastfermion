PYTHON = python3

# Python stuff
# (`python3 -m site --user-site` returns the site-packages directory)
PYTHON_EXTENSION_SUFFIX := $(shell ${PYTHON}-config --extension-suffix)
PYBIND11_INCLUDES := $(shell ${PYTHON} -m pybind11 --includes)

$(info PYTHON_EXTENSION_SUFFIX = $(PYTHON_EXTENSION_SUFFIX))
$(info PYBIND11_INCLUDES = $(PYBIND11_INCLUDES))

# Compilation flags for the Python module
OPTFLAGS = -O3 -DNDEBUG
CCFLAGS = -Wall -shared -std=c++2a -fPIC $(PYBIND11_INCLUDES)
ifeq ($(shell uname -s), Darwin)
	CCFLAGS += -undefined dynamic_lookup
endif

ffcore: src/*.h src/python_bind.cpp
	g++ $(OPTFLAGS) $(CCFLAGS) -DFF_VERSION=\"dev$(shell date '+%Y-%m-%d.%H-%M-%S')\" src/python_bind.cpp -o fastfermion/ffcore$(PYTHON_EXTENSION_SUFFIX)

pytest:
	pytest ./test

cpptest: src/*.h cpptest/*.h cpptest/run_test.cpp
	g++ -O3 -std=c++2a cpptest/run_test.cpp -o cpptest/run_test

# Uses gcovr
testcov:
	(rm ./fastfermion/*.gcda && \
	g++ $(CCFLAGS) -DFF_VERSION=\"devcov\" --coverage -g src/python_bind.cpp -o fastfermion/ffcore$(PYTHON_EXTENSION_SUFFIX) && \
	pytest ./test && \
	gcovr ./fastfermion/)

# Invokes setup.py
# Will create wheel and store it in dist folder
build-wheel:
	python3 -m build --wheel

install:
	pip3 uninstall fastfermion
	pip3 install --user $(shell ls -rt ./dist/fastfermion-*.whl | tail -n 1)