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
