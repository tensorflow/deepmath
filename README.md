# Deepmath

The Deepmath project seeks to improve automated theorem proving using deep
learning and other machine learning techniques.  Deepmath is a collaboration
between [Google Research](https://research.google.com) and several universities.

## DISCLAIMER:

The source code in this repository is not an official Google product, but
is a research collaboration with external research teams.

For now the repository contains only a C++ implementation of the [HOL Light
kernel](https://www.cl.cam.ac.uk/~jrh13/hol-light), which we have released
early in order to faciliate existing collaborations.  More to come soon,
including neural network models.

## Directories

* `hol`: A C++ implementation of the HOL Light kernel.

## Installation

Deepmath depends on TensorFlow, which is included as a submodule.  To use,
first install all TensorFlow dependencies and run `configure` to initialize
the build system, as [described
here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#installing-from-sources):

    # Install TensorFlow dependencies if necessary
    cd deepmath/tensorflow
    ./configure  # and answer all questions

You can then build `hol` via

    cd deepmath/hol
    bazel build ...
