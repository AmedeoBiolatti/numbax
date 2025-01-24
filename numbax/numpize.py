import jax
import numpy as np
import numba

from numbax import primitives, utils


def numpize_jaxpr(jaxpr: jax.core.ClosedJaxpr, as_str: bool = False):
    lines = _process_closed_jaxpr(jaxpr, name='__fn')
    d = {'np': np, 'numba': numba}
    if as_str:
        return "\n".join(lines)
    for line in lines:
        exec(line, d)
    return d['__fn']


def _process_closed_jaxpr(jaxpr: jax.core.ClosedJaxpr, name, with_signature: bool = False) -> list[str]:
    return _process_jaxpr(jaxpr.jaxpr, name, with_signature=with_signature)


def _process_jaxpr(jaxpr: jax.core.Jaxpr, name: str, with_signature: bool = False) -> list[str]:
    jaxpr_lines: list[str] = []
    other_jaxprs: list[jax.core.ClosedJaxpr] = []

    # invars
    str_invars = ', '.join(utils.jaxpr_vars_name(jaxpr.invars, with_typing=True))
    if with_signature:
        jaxpr_lines.append(_build_cfunc_decorator(jaxpr.invars, jaxpr.outvars))
    jaxpr_lines.append(f"def {name}({str_invars}):")

    # eqns
    for eqn in jaxpr.eqns:
        eqn_lines, eqn_other_jaxprs = _process_eqn(eqn, name, j=len(other_jaxprs))
        jaxpr_lines.extend(eqn_lines)
        other_jaxprs.extend(eqn_other_jaxprs)

    # outvars
    str_outvars = ', '.join(utils.jaxpr_vars_name(jaxpr.outvars))
    jaxpr_lines.append(f"return {str_outvars}")

    i = 2 if with_signature else 1
    jaxpr_lines = jaxpr_lines[:i] + [f"    {o}" for o in jaxpr_lines[i:]]
    jaxpr_str = '\n'.join(jaxpr_lines)

    out = [jaxpr_str]
    for i, other_jaxpr in enumerate(other_jaxprs):
        out = _process_closed_jaxpr(other_jaxpr, name=f"{name}__aux{i}", with_signature=True) + out

    return out


def _process_eqn(eqn: jax.core.JaxprEqn, name: str, j: int = 0) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
    outvars_str = utils.jaxpr_vars_name(eqn.outvars, with_typing=True)

    primitive = primitives.get(eqn.primitive)

    eqn_lines, eqn_aux_jaxprs = primitive(*eqn.invars, **eqn.params)
    for i in range(len(eqn_aux_jaxprs)):
        eqn_lines = [line.replace(f"aux_fn{i}", f"{name}__aux{i + j}") for line in eqn_lines]

    lhs = ", ".join(outvars_str)
    eqn_lines[-1] = f"{lhs} = {eqn_lines[-1]}"

    return eqn_lines, eqn_aux_jaxprs


def _dtype_mapping(dtype):
    dtype = str(dtype)
    _mapping = {
        "int16": "numba.types.int16",
        "int32": "numba.types.int32",
        "int64": "numba.types.int64",
        "float16": "numba.types.float16",
        "float32": "numba.types.float32",
        "float64": "numba.types.float64",
        "bool": "numba.types.bool"
    }
    return _mapping[dtype]


def _build_cfunc_decorator(invars, outvars) -> str:
    in_descr = []
    for a in invars:
        if len(a.aval.shape) >= 1:
            in_descr.append(f"numba.types.Array({_dtype_mapping(a.aval.dtype)}, {len(a.aval.shape)}, 'C')")
        else:
            in_descr.append(_dtype_mapping(a.aval.dtype))
    if len(in_descr) == 1:
        in_descr = in_descr[0]
    else:
        in_descr = ', '.join(in_descr)

    out_descr = []
    for a in outvars:
        if len(a.aval.shape) >= 1:
            out_descr.append(f"numba.types.Array({_dtype_mapping(a.aval.dtype)}, {len(a.aval.shape)}, 'C')")
        else:
            out_descr.append(_dtype_mapping(a.aval.dtype))
    if len(out_descr) == 1:
        out_descr = out_descr[0]
    else:
        out_descr = f"numba.types.Tuple(({', '.join(out_descr)}))"

    return f"@numba.cfunc({out_descr}({in_descr}))"
