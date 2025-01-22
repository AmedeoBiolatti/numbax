import jax

import numba
from numbax.numpize import numpize_jaxpr


def jit(fn, *, fast=False, parallel=False):
    return JitWrapped(fn, fast=fast, parallel=parallel)


def get_specs(arg):
    return jax.dtypes.result_type(arg)


class JitWrapped:
    def __init__(self, fn, *, fast=False, parallel=False):
        self.fn = fn
        self.fast = fast
        self.parallel = parallel

        self._lowered = {}
        self._out_trees = {}

    def _lower(self, specs, *args, **kwargs):
        # input shape
        jaxpr = jax.make_jaxpr(self.fn)(*args, **kwargs)
        # output shape
        out_shape = jax.eval_shape(self.fn, *args, **kwargs)
        _, self._out_trees[specs] = jax.tree.flatten(out_shape)

        self._last_fn_numpized = numpize_jaxpr(jaxpr)
        fn_numba = numba.njit(self._last_fn_numpized, parallel=self.parallel, fastmath=self.fast)
        self._lowered[specs] = fn_numba

    def get_in_specs(self, *args, **kwargs):
        flattened_args, in_tree = jax.tree.flatten((args, kwargs))
        in_dtypes = tuple([get_specs(arg) for arg in flattened_args])
        return in_tree, in_dtypes

    def __call__(self, *args, **kwargs):
        specs = self.get_in_specs(*args, **kwargs)
        if specs not in self._lowered:
            self._lower(specs, *args, **kwargs)

        flattened_args, _ = jax.tree.flatten((args, kwargs))
        flattened_res = self._lowered[specs](*flattened_args)
        if not isinstance(flattened_res, tuple):
            flattened_res = flattened_res,
        res = self._out_trees[specs].unflatten(flattened_res)
        return res
