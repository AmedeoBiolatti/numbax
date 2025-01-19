import jax

import numpy as np

_NP = 'np'


def numpize(
        jaxpr: jax.core.ClosedJaxpr
) -> str:
    out = _process_closed_jaxpr(jaxpr, name='__fn')

    out = "\n".join(out)
    d = {_NP: np}
    exec(out, d)
    return d['__fn']


def _process_closed_jaxpr(jaxpr: jax.core.check_jaxpr, name='_fn') -> list[str]:
    jaxpr: jax.core.Jaxpr = jaxpr.jaxpr
    return _process_jaxpr(jaxpr, name=name)


def _process_jaxpr(jaxpr: jax.core.Jaxpr, name='_fn') -> list[str]:
    out = []

    # invars
    str_invars = ', '.join([str(v) for v in jaxpr.invars])
    out.append(f"def {name}({str_invars}):")

    # eqns
    for eqn in jaxpr.eqns:
        lines = _process_eqn(eqn)
        out.extend(lines)

    # outvars
    str_outvars = ', '.join([str(v) for v in jaxpr.outvars])
    out.append(f"return {str_outvars}")

    out = out[:1] + [f"    {o}" for o in out[1:]]
    return out


def _process_eqn(eqn: jax.core.JaxprEqn) -> list[str]:
    invars_str = [str(v) for v in eqn.invars]
    outvars_str = [str(v) for v in eqn.outvars]

    primitive = None
    if eqn.primitive in _numpy_primitive_mapping:
        primitive = _numpy_primitive_mapping[eqn.primitive]

    if primitive is None:
        raise ValueError(f"Doesn't know how to handle primitive '{eqn.primitive}'")

    out = primitive(*invars_str, **eqn.params)

    lhs = ', '.join(outvars_str)
    out[-1] = f"{lhs} = {out[-1]}"

    return out


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
        assert all([k in self.valid_kwargs for k in kwargs.keys()])

    def _call(self, *args, **kwargs) -> list[str]:
        raise NotImplemented

    def __call__(self, *args, **kwargs) -> list[str]:
        self._validate(*args, **kwargs)
        return self._call(*args, **kwargs)


class AddPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=(1, None))

    def _call(self, *args, **kwargs) -> list[str]:
        return [" + ".join(args)]


class MulPrimitive(Primitive):
    def __init__(self):
        super().__init__(n_args=(1, None))

    def _call(self, *args, **kwargs) -> list[str]:
        return [" * ".join(args)]


class NumpyUnaryFn(Primitive):
    def __init__(self, fn_name: str):
        super().__init__(n_args=1)
        self.fn_name = fn_name

    def _call(self, *args, **kwargs) -> list[str]:
        return [f"{_NP}.{self.fn_name}({args[0]})"]


class ConvertElementType(Primitive):
    def __init__(self):
        super().__init__(valid_kwargs=['new_dtype', 'weak_type'], n_args=1)

    def _call(self, *args, **kwargs) -> list[str]:
        # TODO: what to do with 'weak_type'?
        return [f"{_NP}.array({args[0]}).astype('{kwargs['new_dtype']}')"]


class Iota(Primitive):
    def __init__(self):
        super().__init__(valid_kwargs=['dtype', 'shape', 'dimension'], n_args=0)

    def _call(self, *args, **kwargs) -> list[str]:
        assert kwargs['dimension'] == 0
        assert len(kwargs['shape']) == 1

        return [f"{_NP}.arange({kwargs['shape'][0]}, dtype='{kwargs['dtype']}')"]


class Scan(Primitive):
    def __init__(self):
        super().__init__(valid_kwargs=['reverse', 'length', 'jaxpr', 'num_consts', 'num_carry', 'linear', 'unroll'],
                         n_args=2)

    def _call(self, *args, **params) -> list[str]:
        assert not params['reverse']  # TODO
        assert params['num_consts'] == 0  # TODO
        assert params['num_carry'] == 1
        assert params['linear'] == (False, False)
        assert params['unroll'] == 1

        init, xs = args

        jaxpr_lines = _process_closed_jaxpr(params['jaxpr'], name='_scan_step')

        for_lines = []
        for_lines.append(f"_carry = {init}")
        for_lines.append("_ys = []")
        for_lines.append(f"for _x in {xs}:")
        for_lines.append(f"    _carry, _y = _scan_step(_carry, _x)")
        for_lines.append("    _ys.append(_y)")
        for_lines.append(f"_carry, {_NP}.stack(_ys)")

        lines = jaxpr_lines + for_lines
        return lines


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
