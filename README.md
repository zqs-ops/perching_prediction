# svm_for_perching_prediction

## Content

- Overview
- System Requirements
- Installation Guide
- Demo
- Results

## Overview

This software package is an efficient machine learning framework for predicting the success of vertical wall perching of bionic flying robots with spines, overcoming the inefficiency of traditional methods. Through the training of mixed sample data, a data-driven model was established to predict the success or failure of any perching event. This high-precision prediction optimizes the control and structural parameters of the robot and ensures stable perching .

## System Requirements

### Hardware requirements

The `svm_for_perching_prediction` software package only requires a standard computer that supports Python3.7 and has sufficient RAM to support memory operations.

### Software requirements

#### OS Requirements

This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:

- Windows 11: Family Chinese Version (23H2)
- Linux: Ubuntu 18.04

#### Python Dependencies

`svm_for_perching_prediction` mainly depends on the Python scientific stack.

```
libsvm
numpy
matplotlib
sklearn
random
pylab
```

## Installation Guide

### Install python 3.7

- Visit Python 's official website: [Download Python | Python.org](https://www.python.org/downloads/)
- Download Python3.7 that suits your operating system.
- Run the downloaded installation program and follow the wizard to install it.
- During the installation process, make sure the "Add Python to PATH" option is checked so that Python can be used directly in the command line.

### Install library

```
python -m pip install --upgrade pip

pip install libsvm
pip install numpy
pip install matplotlib
pip install sklearn
pip install random
pip install pylab
```

### Download the code from github

```
git clonehttps://github.com/zqs-ops/svm_for_perching_prediction.git
```

## Examples

当构件完成代码运行环境后，你只需运行