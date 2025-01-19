import jax

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

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        raise NotImplemented

    def __call__(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        self._validate(*args, **kwargs)
        return self._call(*args, **kwargs)

    def varnames(self, *args: jax.core.Var | jax.core.Literal) -> tuple[str]:
        return tuple(self.varname(arg) for arg in args)

    def varname(self, arg: jax.core.Var | jax.core.Literal) -> str:
        def _jaxpr_var_name(v):
            name = str(v)
            if isinstance(v, jax.core.Var):
                name = name.split(":")[0]
                name = name.replace("(", "_")
                name = name.replace(")", "_")
                name = name.replace("=", "_")
            return name

        # name: str = (str(arg) + "_") if isinstance(arg, jax.core.Var) else str(arg)
        name: str = _jaxpr_var_name(arg)
        return name


class AddPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=(1, None))

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [" + ".join(self.varnames(*args))], []


class MulPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=(1, None))

    def _call(self, *args, **kwargs) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        return [" * ".join(self.varnames(*args))], []


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
        super().__init__(valid_kwargs=['reverse', 'length', 'jaxpr', 'num_consts', 'num_carry', 'linear', 'unroll',
                                       '_split_transpose'],
                         n_args=2)

    def _call(self, *args, **params) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
        assert not params['reverse']  # TODO
        assert params['num_consts'] == 0  # TODO
        assert params['num_carry'] == 1
        assert params['linear'] == (False, False)
        assert params['unroll'] == 1

        init, xs = self.varnames(*args)[:2]

        aux_jaxpr = params['jaxpr']

        lines = []
        lines.append(f"_carry = {init}")
        lines.append("_ys = []")
        lines.append(f"for _x in {xs}:")
        lines.append(f"    _carry, _y = aux_fn0(_carry, _x)")
        lines.append("    _ys.append(_y)")
        lines.append(f"_ys_stacked = np.empty((len({xs}),))")
        lines.append(f"for i in range(len({xs})):")
        lines.append(f"    _ys_stacked[i] = _ys[i]")
        lines.append(f"_carry, _ys_stacked")

        return lines, [aux_jaxpr]


_numpy_primitive_mapping = {
    jax.lax.add_p: AddPrimitive(),
    jax.lax.mul_p: MulPrimitive(),
    jax.lax.exp_p: NumpyUnaryFn('exp'),
    jax.lax.log_p: NumpyUnaryFn('log'),
    jax.lax.sin_p: NumpyUnaryFn('sin'),
    jax.lax.cos_p: NumpyUnaryFn('cos'),

    jax.lax.convert_element_type_p: ConvertElementType(),
    jax.lax.iota_p: Iota(),
    jax.lax.scan_p: Scan()
}


def get(primitive) -> Primitive:
    return _numpy_primitive_mapping[primitive]
