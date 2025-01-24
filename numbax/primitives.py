import jax
import numpy as np
from numba import prange
from scipy.stats import kappa4

from numbax import utils

_NP = 'np'


class Primitive:
    def __init__(self, valid_kwargs=None, n_args=None):
        if isinstance(n_args, int):
            self.n_args_min = n_args
            self.n_args_max = n_args
        elif isinstance(n_args, tuple):
            self.n_args_min, self.n_args_max = n_args
        elif isinstance(n_args, None):
            self.n_args_min, self.n_args_max = None, None
        else:
            raise TypeError
        self.valid_kwargs = valid_kwargs or ()

    def _validate(self, *args, **kwargs):
        if self.n_args_min:
            assert len(args) >= self.n_args_min
        if self.n_args_max:
            assert len(args) <= self.n_args_max
        if not all([k in self.valid_kwargs for k in kwargs.keys()]):
            missing_kwargs = [k for k in kwargs.keys() if k not in self.valid_kwargs]
            missing_kwargs_str = ", ".join([("'" + str(k) + "'(=" + str(kwargs[k]) + ")") for k in missing_kwargs])
            raise ValueError(f"{missing_kwargs_str} not in kwargs for primitive {self.__class__.__name__}")

    def _call(self, *args: jax.core.Var | jax.core.Literal, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        raise NotImplemented

    def __call__(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        self._validate(*args, **kwargs)
        return self._call(*args, **kwargs)

    def varnames(self, *args: jax.core.Var | jax.core.Literal) -> tuple[str]:
        return tuple(self.varname(arg) for arg in args)

    def varname(self, arg: jax.core.Var | jax.core.Literal) -> str:
        name: str = utils.jaxpr_var_name(arg)
        return name


class AddPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=2)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"np.add({self.varname(args[0])}, {self.varname(args[1])})"], []


class SubPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=2)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"np.subtract({self.varname(args[0])}, {self.varname(args[1])})"], []


class DivPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=2)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"np.divide({self.varname(args[0])}, {self.varname(args[1])})"], []


class MulPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=2)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"np.multiply({self.varname(args[0])}, {self.varname(args[1])})"], []


class IntergerPow(Primitive):
    def __init__(self):
        super().__init__(n_args=1, valid_kwargs=("y",))

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"{self.varname(args[0])} ** {kwargs['y']}"], []


class Pow(Primitive):
    def __init__(self):
        super().__init__(n_args=2)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"{self.varname(args[0])} ** {self.varname(args[1])}"], []


class NumpyUnaryFn(Primitive):
    def __init__(self, fn_name: str):
        super().__init__(n_args=1)
        self.fn_name = fn_name

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [f"{_NP}.{self.fn_name}({self.varname(args[0])})"], []


class ConvertElementType(Primitive):
    def __init__(self):
        super().__init__(valid_kwargs=['new_dtype', 'weak_type', 'sharding'], n_args=1)

    def _call(self, *args: jax.core.Var | jax.core.Literal, **params) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        # TODO: what to do with 'weak_type' and 'sharding'?
        if 'sharding' in params:
            assert params['sharding'] is None

        a = args[0]
        if len(a.aval.shape) == 0:
            return [f"{_NP}.{params['new_dtype']}({self.varname(args[0])})"], []
        return [f"({self.varname(args[0])}).astype({_NP}.{params['new_dtype']})"], []


class Iota(Primitive):
    def __init__(self):
        super().__init__(valid_kwargs=['dtype', 'shape', 'dimension', 'sharding'], n_args=0)

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        assert kwargs['dimension'] == 0
        assert len(kwargs['shape']) == 1
        assert kwargs['sharding'] is None

        start = 0
        stop = kwargs['shape'][0]
        step = 1
        dtype = kwargs['dtype']

        return [f"{_NP}.arange({start}, {stop}, {step}, {_NP}.{dtype})"], []


class Scan(Primitive):
    def __init__(self):
        super().__init__(
            valid_kwargs=['reverse', 'length', 'jaxpr', 'num_consts', 'num_carry', 'linear', 'unroll',
                          '_split_transpose'],
            n_args=(2, 100)
        )

    def _call(self, *args, **params) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        # assert not params['reverse']  # TODO
        assert params['num_consts'] == 0  # TODO
        assert params['num_carry'] >= 1
        assert isinstance(params['linear'], tuple)
        assert all([not li for li in params['linear']])
        assert params['unroll'] == 1
        assert not params['_split_transpose']

        num_carry = params['num_carry']
        length = params['length']

        varnames = self.varnames(*args)
        if num_carry + 1 == len(varnames):
            init, xs = varnames[:-1], varnames[-1]
        elif num_carry == len(varnames):
            init = varnames
            xs = f"range({params['length']})"
        else:
            raise ValueError

        if isinstance(init, list | tuple):
            init = ", ".join(init)

        aux_jaxpr: jax.jaxpr = params['jaxpr']

        lines = []
        lines.append(f"_carry = {init}")
        if len(aux_jaxpr.jaxpr.invars) == num_carry + 1:
            lines.append("_ys = []")
        if params["reverse"]:
            lines.append(f"for _x in {xs}[::-1]:")
        else:
            lines.append(f"for _x in {xs}:")

        if num_carry > 1:
            if len(aux_jaxpr.jaxpr.invars) == num_carry + 1:
                lines.append(f"    _carry_nd_y = aux_fn0(*_carry, _x)")
            elif len(aux_jaxpr.jaxpr.invars) == num_carry:
                lines.append(f"    _carry_nd_y = aux_fn0(*_carry)")
            else:
                raise ValueError(f"Scan primitive: {len(aux_jaxpr.jaxpr.invars)} invars and {num_carry} num_carry")
        else:
            if len(aux_jaxpr.jaxpr.invars) == num_carry + 1:
                lines.append(f"    _carry_nd_y = aux_fn0(_carry, _x)")
            elif len(aux_jaxpr.jaxpr.invars) == num_carry:
                lines.append(f"    _carry_nd_y = aux_fn0(_carry)")
            else:
                raise ValueError(f"Scan primitive: {len(aux_jaxpr.jaxpr.invars)} invars and {num_carry} num_carry")

        if num_carry > 1:
            lines.append(f"    _carry = _carry_nd_y[:{num_carry}]")
        else:
            lines.append(f"    _carry = _carry_nd_y[0]")
        if len(aux_jaxpr.jaxpr.invars) == num_carry + 1:
            lines.append(f"    _y = _carry_nd_y[-1]")
            lines.append("    _ys.append(_y)")

        if len(aux_jaxpr.jaxpr.invars) == num_carry + 1:
            _dtype = params["jaxpr"].jaxpr.outvars[-1].aval.dtype
            lines.append(f"_ys_stacked = np.empty((len({xs}),), dtype=np.{_dtype})")
            lines.append(f"for i in range({length}):")
            lines.append(f"    _ys_stacked[i] = _ys[i]")
            if params["num_carry"] > 1:
                lines.append(f"_carry + (_ys_stacked,)")
            else:
                lines.append(f"_carry, _ys_stacked")
        elif len(aux_jaxpr.jaxpr.invars) == num_carry:
            lines.append(f"_carry")

        return lines, [aux_jaxpr]


class BinaryOperator(Primitive):
    def __init__(self, symbol):
        super().__init__(n_args=2)
        self.symbol = symbol

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        names = self.varnames(*args)
        name_l, name_r = names[:2]

        lines = [f"{name_l} {self.symbol} {name_r}"]

        return lines, []


class Cond(Primitive):
    def __init__(self):
        super().__init__(n_args=(1, 100), valid_kwargs=["branches"])

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        assert len(kwargs["branches"]) == 2

        names = self.varnames(*args)
        cond_name = names[0]
        var_names = names[1:]
        args_ = ",".join(var_names)

        lines = []
        lines.append(f"aux_fn0({args_}) if {cond_name} else aux_fn1({args_})")

        branch_true = kwargs["branches"][1]
        branch_false = kwargs["branches"][0]

        return lines, [branch_true, branch_false]


class SelectN(Primitive):
    def __init__(self):
        super().__init__(n_args=(3, None))

    def _call(self, cond: jax.core.Var | jax.core.Literal, *args, **kwargs) -> tuple[
        list[str], list[jax.core.ClosedJaxpr]]:
        cond_name = self.varname(cond)
        args_name = self.varnames(*args)
        n_args = len(args)

        if len(cond.aval.shape) == 0:
            # scalars
            cond_str = f"int({cond_name})"
            args_str = ", ".join(args_name)
            lines = [f"[{args_str}][{cond_str}]"]
        else:
            if "int" not in str(cond.aval.dtype):
                cond_name = f"np.asarray({cond_name}, dtype=np.int32)"

            cond_str = ",".join([f"{cond_name} == {i}" for i in range(n_args)])
            args_str = ", ".join([f"np.asarray({a})" for a in args_name])

            lines = [f"np.select([{cond_str}], [{args_str}])"]

        return lines, []


class ReduceOperator(Primitive):
    def __init__(self, symbol: str):
        super().__init__(
            n_args=1,
            valid_kwargs=("axes",)
        )
        self.symbol = symbol

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        varname = self.varname(args[0])
        axes = kwargs["axes"]

        lines = [f"{varname}_aux = {varname}"]
        for ax in sorted(axes)[::-1]:
            lines.append(f"{varname}_aux = np.{self.symbol}({varname}_aux, axis={ax})")
        lines[-1] = lines[-1].split(" = ")[-1]

        return lines, []


class BroadcastInDim(Primitive):
    def __init__(self):
        super().__init__(
            n_args=1,
            valid_kwargs=("shape", "broadcast_dimensions", "sharding")
        )

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        varname = self.varname(args[0])
        shape = kwargs["shape"]
        broadcast_dimensions = kwargs["broadcast_dimensions"]
        assert kwargs["sharding"] is None

        n_dim_in = len(args[0].aval.shape)
        n_dim_out = len(shape)

        if n_dim_in < n_dim_out:
            new_shape = tuple(args[0].aval.shape) + tuple([1] * (n_dim_out - n_dim_in))
            new_shape_str = ",".join([str(v) for v in new_shape])
            if len(args[0].aval.shape) == 0:
                varname = f"np.array({varname})"
            varname = f"np.reshape({varname}, newshape=({new_shape_str}))"
        lines = [f"np.broadcast_to({varname}, shape={shape})"]
        return lines, []


class Reshape(Primitive):
    def __init__(self):
        super().__init__(
            n_args=1,
            valid_kwargs=("new_sizes", "dimensions", "sharding")
        )

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        assert kwargs["dimensions"] is None
        assert kwargs["sharding"] is None

        varname = self.varname(args[0])
        if len(args[0].aval.shape) == 0:
            varname = f"np.array({varname})"
        axes_str = "(" + ",".join([str(v) for v in kwargs["new_sizes"]]) + ")"

        list = [f"np.reshape({varname}, newshape={axes_str})"]
        return list, []


class Transpose(Primitive):
    def __init__(self):
        super().__init__(
            n_args=1,
            valid_kwargs=("permutation",)
        )

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        varname = self.varname(args[0])
        axes_str = "(" + ",".join([str(v) for v in kwargs["permutation"]]) + ")"

        lines = []
        lines.append(f"np.transpose({varname}, axes={axes_str})")

        return lines, []


class PJit(Primitive):
    def __init__(self):
        super().__init__(
            n_args=(1, None),
            valid_kwargs=(
                'jaxpr',
                'in_shardings', 'out_shardings',
                'in_layouts', 'out_layouts',
                'resource_env',
                'donated_invars',
                'name',
                'keep_unused',
                'inline',
                'compiler_options_kvs')
        )

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        assert all([isinstance(v, jax._src.sharding_impls.UnspecifiedValue) for v in kwargs["in_shardings"]])
        assert all([isinstance(v, jax._src.sharding_impls.UnspecifiedValue) for v in kwargs["out_shardings"]])
        assert all([v is None for v in kwargs["in_layouts"]])
        assert all([v is None for v in kwargs["out_layouts"]])
        assert kwargs["resource_env"] is None
        assert all([not v for v in kwargs["donated_invars"]])
        assert not kwargs["keep_unused"]

        varnames = self.varnames(*args)

        varnames_str = ", ".join(varnames)
        lines = [f"aux_fn0({varnames_str})"]

        return lines, [kwargs["jaxpr"]]


_numpy_primitive_mapping = {
    jax.lax.add_p: AddPrimitive(),
    jax.lax.sub_p: SubPrimitive(),
    jax.lax.mul_p: MulPrimitive(),
    jax.lax.div_p: DivPrimitive(),
    jax.lax.integer_pow_p: IntergerPow(),
    jax.lax.pow_p: Pow(),
    jax._src.ad_util.add_any_p: AddPrimitive(),

    jax.lax.exp_p: NumpyUnaryFn('exp'),
    jax.lax.log_p: NumpyUnaryFn('log'),
    jax.lax.sin_p: NumpyUnaryFn('sin'),
    jax.lax.cos_p: NumpyUnaryFn('cos'),
    jax.lax.neg_p: NumpyUnaryFn('negative'),
    jax.lax.sqrt_p: NumpyUnaryFn('sqrt'),

    jax.lax.gt_p: BinaryOperator(">"),
    jax.lax.lt_p: BinaryOperator("<"),
    jax.lax.ge_p: BinaryOperator(">="),
    jax.lax.le_p: BinaryOperator("<="),
    jax.lax.eq_p: BinaryOperator("=="),

    jax.lax.reduce_sum_p: ReduceOperator("sum"),
    jax.lax.reduce_prod_p: ReduceOperator("prod"),
    jax.lax.reduce_min_p: ReduceOperator("min"),
    jax.lax.reduce_max_p: ReduceOperator("max"),
    jax.lax.reduce_and_p: ReduceOperator("and"),
    jax.lax.reduce_or_p: ReduceOperator("or"),
    jax.lax.reduce_xor_p: ReduceOperator("xor"),

    jax.lax.convert_element_type_p: ConvertElementType(),
    jax.lax.broadcast_in_dim_p: BroadcastInDim(),
    jax.lax.reshape_p: Reshape(),
    jax.lax.transpose_p: Transpose(),

    jax.lax.iota_p: Iota(),

    jax.lax.scan_p: Scan(),
    jax.lax.cond_p: Cond(),
    jax._src.pjit.pjit_p: PJit(),
    jax.lax.select_n_p: SelectN()
}


def get(primitive) -> Primitive:
    return _numpy_primitive_mapping[primitive]
