# GP-GOMEA with GPU-Based Fitness Evaluations: Design and Performance Analysis

This repository contains the source code and experimental results for the paper ["GP-GOMEA with GPU-Based Fitness Evaluations: Design and Performance Analysis"](https://arxiv.org/abs/2605.30954) accepted at PPSN 2026 and the master thesis titled ["GP-GOMEA with GPU-Based Fitness Evaluations"](https://resolver.tudelft.nl/uuid:bc05593f-34ac-4dc2-9cce-73ba4a201999).

> [!NOTE]
> This is the code for the paper and thesis, a more user-friendly and complete (GP-)GOMEA library containing the contributions of this repository is currently under development.

## Installation

This project supports Linux (and macOS without GPU support) and requires a recent C++ compiler with C++23 support. For GPU support, a CUDA compiler is needed (tested with g++ 15 and nvcc 13.2.86). With the toolchain in place, the Python bindings (supporting Python >=3.7) can be installed using:

```bash
pip install git+https://github.com/JRPost11/ga-goblin.git@dev#egg=pygom
```

Note that the CUDA kernels are opt-in and only included if CUDA is detected by CMake. If the Python package was compiled without GPU support, the example below will raise a corresponding error.

### Installation From Source

To build from source, the Python package manager [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is additionally required:

```bash
# get the code
git clone --branch dev --depth 1 --single-branch https://github.com/JRPost11/ga-goblin.git
cd ga-goblin
# autogenerate C++ bindings and install the python package
# requires CXX, and optionally CUDACXX and CUDA_TOOLKIT_ROOT_DIR to be set
make bindings
```

Note that building from source is only recommended if the underlying C++ code is changed as well.

## Usage

Once installed, the Python bindings in the `pygom` package provide a SKLearn compatible Python interface for symbolic regression:

```python
import numpy as np
from pygom import KernelVersion
import pygom.gp as gp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def sr_example(gpu: bool = False):
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    est = gp.SymbolicRegressor(
        budget_kwargs=dict(
            max_time_seconds=5,
        ),
        ims_kwargs=dict(initial_population_size=1024, max_num_populations=1),
        discrete_model_kwargs=dict(
            metric="node_proximity"
        ),
        kernel_version = KernelVersion.single_block if gpu else None
    )

    est.fit(X_train, y_train)

    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    print("GPU" if gpu else "CPU")
    print("Best expression:", est.model)
    print("R2 train:", r2_train)
    print("R2 test:", r2_test)

if __name__ == "__main__":
    sr_example() # cpu
    sr_example(gpu=True) # gpu
```

## Paper Experiments & Results

The experiment source code and results for both the paper and thesis can be found at `experiments/python`.

## License

This work is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/

## Citation

If you find this work useful, please cite the published version, the [master thesis for the extended work not in the paper](https://resolver.tudelft.nl/uuid:bc05593f-34ac-4dc2-9cce-73ba4a201999) or the arXiv preprint:

```
@misc{https://doi.org/10.48550/arxiv.2605.30954,
  doi = {10.48550/ARXIV.2605.30954},
  url = {https://arxiv.org/abs/2605.30954},
  author = {Post,  Jasper and Koch,  Johannes and Bouter,  Anton and Alderliesten,  Tanja and Bosman,  Peter A. N.},
  keywords = {Neural and Evolutionary Computing (cs.NE),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {GP-GOMEA with GPU-Based Fitness Evaluations: Design and Performance Analysis},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```

