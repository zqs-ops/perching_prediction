# perching_prediction

## Contents

- Overview
- System requirements
- Installation guide
- Demo
- Instructions for use

## Overview

The software package contains three machine learning methods, namely `mlp`, `svm`, and `random forest`, which are used to predict the success of vertical wall perching of bionic flying robots with spines. Through the training of mixed sample data, a data-driven model was established to predict the success or failure of any perching event. This high-precision prediction optimizes the control and structural parameters of the robot and ensures stable perching .

## System requirements

### Hardware requirements

The `perching_prediction` software package only requires a standard computer that supports Python3.7 and has sufficient RAM to support memory operations.

### Software requirements

#### OS requirements

This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:

- Windows 11: Family Chinese Version (23H2)
- Linux: Ubuntu 18.04

#### Python dependencies

`perching_prediction` mainly depends on the Python scientific stack.

```
libsvm
numpy
matplotlib
sklearn
random
pylab
pandas
tensorflow
```

## Installation guide

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
pip install pandas
pip install tensorflow
```

### Typical install time

- 1 hour.

### Download the code from github

```
git clonehttps://github.com/zqs-ops/svm_for_perching_prediction.git
```

## Demo

### Execution method

- mlp: mlp/demo.py
- random_forest: random_forest/demo.py
- svm: svm/demo.py

### Expected output

- mlp: “demo_training_history.txt” and “demo_decision_boundary_data.txt”.
- random_forest: “demo_estimator.xlsx”.
- svm: “demo_predictions_best_model.txt” and “demo_svm_decision_boundary_comparison.png”.

### Expected run time

- No more than 3 minutes.

### Instructions for use

For different machine learning methods, just run the corresponding programs below:

- mlp: mlp/mlp.py
- random_forest: random_forest/random_forest.py
- svm: svm/svm.py

Note: There are two sets of original input data provided, and readers need to switch between them.