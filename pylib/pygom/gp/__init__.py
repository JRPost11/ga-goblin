import re

import numpy as np
import sympy as sym
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import pygom


_GPU_AVAILABLE = pygom.has_gpu_support()


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel_version=None,
        **kwargs,
    ):
        self.kernel_version = kernel_version
        self.kwargs = kwargs
        self.imputer = None
        self.front = []
        self.model = None

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs

    def __lambdify_expression(self, e):
        """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
        e = str(e)

        symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
        expr = sym.sympify(e, locals=symbols)
        f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

        def fn(X: np.ndarray):
            try:
                return f(*[X[:, int(s[1:])] for s in symbols.keys()])
            except Exception as e:
                return np.repeat(float("nan"), X.shape[0])

        return fn

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy().astype(np.float64)

        if not isinstance(X, np.ndarray):
            y = y.to_numpy().astype(np.float64)

        # impute if needed
        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=self.kwargs.get("random_state", None),
            sample_posterior=True,
        )
        if np.isnan(X).any():
            X = self.imputer.fit_transform(X)
        else:
            self.imputer.fit(X)

        budget_kwargs = self.kwargs.get("budget_kwargs", {})
        # if "termination_callback" not in budget_kwargs:
        #     budget_kwargs["termination_callback"] = pygom.default_termination_callback
        budget = pygom.Budget(**budget_kwargs)
        template = pygom.Template()
        for branching_factor, depth in self.kwargs.get("outputs", [(2, 4)]):
            template.add_output(pygom.TemplateNode.full_nary(branching_factor, depth))
        for branching_factor, depth in self.kwargs.get("subtrees", []):
            template.add_subtree(pygom.TemplateNode.full_nary(branching_factor, depth))
        str2op = {
            "+": pygom.OpAdd(),
            "-": pygom.OpSubGPU(),
            "*": pygom.OpMul(),
            "/": pygom.OpDiv(),
            "sin": pygom.OpSin(),
            "cos": pygom.OpCos(),
            "exp": pygom.OpExp(),
            "log": pygom.OpLog(),
            "square": pygom.OpSquare(),
            "sqrt": pygom.OpSqrt(),
            "pow": pygom.OpPow(),
            "abs": pygom.OpAbs(),
            "min": pygom.OpMin(),
            "max": pygom.OpMax(),
        }
        ctx = pygom.GPContext(
            num_inputs=int(X.shape[1]),
            expression_template=template,
            operators=[
                str2op[op] for op in self.kwargs.get("operators", "+,-,*,/").split(",")
            ],
            constant_representation=self.kwargs.get("constant_representation", "ercs"),
            enable_subfunctions=True,
        )

        Y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        init = vars(pygom)[self.kwargs.get("init", "HalfHalfInit")]()

        temp = float(np.nanmax(np.abs(Y)))
        constant_init_lower_bound = -temp
        constant_init_upper_bound = temp

        if self.kernel_version is not None and not _GPU_AVAILABLE:
            raise RuntimeError(
                "kernel_version requires a CUDA-enabled build of goblin (GOBLIN_HAS_CUDA)"
            )

        problem = pygom.SRProblem(
            ctx,
            X,
            Y,
            objectives=self.kwargs.get("objectives", "mse"),
            linear_scaling=self.kwargs.get("linear_scaling", False),
            init=init,
            constant_init_lower_bound=self.kwargs.get(
                "constant_init_lower_bound", constant_init_lower_bound
            ),
            constant_init_upper_bound=self.kwargs.get(
                "constant_init_upper_bound", constant_init_upper_bound
            ),
            target_objectives=self.kwargs.get("target_objectives", None),
            kernel_version=self.kernel_version,
        )

        if "algorithm" in self.kwargs:
            alg = vars(pygom)[self.kwargs["algorithm"]](
                **self.kwargs.get("algorithm_kwargs", {})
            )
        else:
            # Check for modular tree -> use custom LT-FOS
            if ctx.subtree_roots:
                # to use node proximity instead of MI
                node_proximity = np.array(ctx.normalized_node_proximity().tolist())

                # create a separate LT FOS per modular tree that is restricted to the corresponding tree indices
                linkage_trees = []
                for subtree in ctx.subtree_roots:
                    indices = ctx.nodes[subtree]
                    linkage_trees.append(
                        pygom.LinkageTreeFOS(
                            custom_similarity=node_proximity[np.ix_(indices, indices)],
                            subset=pygom.Subset(discrete=indices),
                            filter_root=False,  # according to Joe's paper subtree LTs should contain the full subtree
                        )
                    )
                for output in ctx.output_roots:
                    indices = ctx.nodes[output]
                    linkage_trees.append(
                        pygom.LinkageTreeFOS(
                            custom_similarity=node_proximity[np.ix_(indices, indices)],
                            subset=pygom.Subset(discrete=indices),
                        )
                    )

                combined_fos = pygom.CombinedFOS([])
                for lt in linkage_trees:
                    combined_fos.add_model(lt)

                alg = pygom.MixedGOMEA(
                    population_options=pygom.PopulationOptions(
                        **self.kwargs.get("population_kwargs", {})
                    ),
                    ims_options=pygom.IMSOptions(**self.kwargs.get("ims_kwargs", {})),
                    rv_options=pygom.RvOptions(enabled = False), # disabled since this implementation is not yet working correctly **self.kwargs.get("rv_kwargs", {})),
                    discrete_model=combined_fos,
                    continuous_model=vars(pygom)[
                        self.kwargs.get("continuous_model", "FullFOS")
                    ](**self.kwargs.get("continuous_model_kwargs", {})),
                    sampling_model=vars(pygom)[
                        self.kwargs.get("sampling_model", "AMaLGaMSamplingModel")
                    ](**self.kwargs.get("sampling_model_kwargs", {})),
                    repr=self.kwargs.get("repr", "aos"),
                )

            else:
                discrete_model_kwargs = {**self.kwargs.get("discrete_model_kwargs", {})}
                if (
                    discrete_model_kwargs.get("metric", "node_proximity")
                    == "node_proximity"
                ):
                    discrete_model_kwargs["custom_similarity"] = np.array(
                        ctx.normalized_node_proximity().tolist()
                    )

                alg = pygom.MixedGOMEA(
                    population_options=pygom.PopulationOptions(
                        **self.kwargs.get("population_kwargs", {})
                    ),
                    ims_options=pygom.IMSOptions(**self.kwargs.get("ims_kwargs", {})),
                    rv_options=pygom.RvOptions(**self.kwargs.get("rv_kwargs", {})),
                    discrete_model=vars(pygom)[
                        self.kwargs.get("discrete_model", "LinkageTreeFOS")
                    ](**discrete_model_kwargs),
                    continuous_model=vars(pygom)[
                        self.kwargs.get("continuous_model", "FullFOS")
                    ](**self.kwargs.get("continuous_model_kwargs", {})),
                    sampling_model=vars(pygom)[
                        self.kwargs.get("sampling_model", "AMaLGaMSamplingModel")
                    ](**self.kwargs.get("sampling_model_kwargs", {})),
                    repr=self.kwargs.get("repr", "aos"),
                )

        seed = self.kwargs.get("random_state", self.kwargs.get("seed"))

        tracking_kwargs = self.kwargs.get("tracking_kwargs", {})
        if "logpath" in tracking_kwargs:
            tracking_kwargs["logpath"] = str(tracking_kwargs["logpath"])
            archive, _status = pygom.Tracked.run(
                problem,
                alg,
                budget,
                pygom.TrackingOptions(**tracking_kwargs),
                seed=seed,
            )
        else:
            archive, _status = alg.run(problem, budget, seed=seed)

        def str2expr(e: str):
            # format is expr1,expr2,...
            return str(e).split(",")[0]

        # use best in first objective by default
        self.model = str2expr(problem.format_solution(archive.so_solution(0)))

        self.front = [
            str2expr(problem.format_solution(archive[i])) for i in range(archive.size())
        ]

    def predict(self, X: np.ndarray):
        assert self.model is not None

        if not isinstance(X, np.ndarray):
            X = X.to_numpy().astype(np.float64)

        # impute if needed
        if np.isnan(X).any():
            X = self.imputer.transform(X)

        y_pred = np.asarray(self.__lambdify_expression(self.model)(X))

        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred
