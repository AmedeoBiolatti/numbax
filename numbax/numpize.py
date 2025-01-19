import jax
import numpy as np
import numba

from numbax import primitives


def numpize_jaxpr(jaxpr: jax.core.ClosedJaxpr):
    lines = _process_closed_jaxpr(jaxpr, name='__fn')
    d = {'np': np, 'numba': numba}
    for line in lines:
        exec(line, d)
    return d['__fn']


def _process_closed_jaxpr(jaxpr: jax.core.ClosedJaxpr, name, with_signature: bool = False) -> list[str]:
    return _process_jaxpr(jaxpr.jaxpr, name, with_signature=with_signature)


def _jaxpr_var_name(v):
    name = str(v)
    if isinstance(v, jax.core.Var):
        name = name.split(":")[0]
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = name.replace("=", "_")
    return name


def _process_jaxpr(jaxpr: jax.core.Jaxpr, name: str, with_signature: bool = False) -> list[str]:
    jaxpr_lines: list[str] = []
    other_jaxprs: list[jax.core.ClosedJaxpr] = []

    # invars
    # str_invars = ', '.join([(str(v) + "_" if isinstance(v, jax.core.Var) else str(v)) for v in jaxpr.invars])
    str_invars = ', '.join([_jaxpr_var_name(v) for v in jaxpr.invars])
    if with_signature:
        jaxpr_lines.append(_build_cfunc_decorator(jaxpr.invars, jaxpr.outvars))
    jaxpr_lines.append(f"def {name}({str_invars}):")

    # eqns
    for eqn in jaxpr.eqns:
        eqn_lines, eqn_other_jaxprs = _process_eqn(eqn, name)
        jaxpr_lines.extend(eqn_lines)
        other_jaxprs.extend(eqn_other_jaxprs)

    # outvars
    # str_outvars = ', '.join([(str(v) + "_" if isinstance(v, jax.core.Var) else str(v)) for v in jaxpr.outvars])
    str_outvars = ', '.join([_jaxpr_var_name(v) for v in jaxpr.outvars])
    jaxpr_lines.append(f"return {str_outvars}")

    i = 2 if with_signature else 1
    jaxpr_lines = jaxpr_lines[:i] + [f"    {o}" for o in jaxpr_lines[i:]]
    jaxpr_str = '\n'.join(jaxpr_lines)

    out = [jaxpr_str]
    for i, other_jaxpr in enumerate(other_jaxprs):
        out = _process_closed_jaxpr(other_jaxpr, name=f"{name}__aux{i}", with_signature=True) + out

    return out


def _process_eqn(eqn: jax.core.JaxprEqn, name: str) -> tuple[list[str], list[jax.core.ClosedJaxpr]]:
    # invars_str = [(str(v) + "_" if isinstance(v, jax.core.Var) else str(v)) for v in eqn.invars]
    # outvars_str = [(str(v) + "_" if isinstance(v, jax.core.Var) else str(v)) for v in eqn.outvars]
    invars_str = [_jaxpr_var_name(v) for v in eqn.invars]
    outvars_str = [_jaxpr_var_name(v) for v in eqn.outvars]

    primitive = primitives.get(eqn.primitive)

    eqn_lines, eqn_aux_jaxprs = primitive(*eqn.invars, **eqn.params)
    eqn_lines = [line.replace('aux_fn0', f"{name}__aux0") for line in eqn_lines]

    lhs = ', '.join(outvars_str)
    eqn_lines[-1] = f"{lhs} = {eqn_lines[-1]}"

    return eqn_lines, eqn_aux_jaxprs


def _build_cfunc_decorator(invars, outvars) -> str:
    in_descr = []
    for a in invars:
        if len(a.aval.shape) >= 1:
            in_descr.append(f"numba.types.Array('{a.aval.dtype}', {len(a.aval.shape)}, 'C')")
        else:
            in_descr.append(f"numba.types.{a.aval.dtype}")
    if len(in_descr) == 1:
        in_descr = in_descr[0]
    else:
        in_descr = ', '.join(in_descr)

    out_descr = []
    for a in outvars:
        if len(a.aval.shape) >= 1:
            out_descr.append(f"numba.types.Array('{a.aval.dtype}', {len(a.aval.shape)}, 'C')")
        else:
            out_descr.append(f"numba.types.{a.aval.dtype}")
    if len(out_descr) == 1:
        out_descr = out_descr[0]
    else:
        out_descr = f"numba.types.Tuple(({', '.join(out_descr)}))"

    return f"@numba.cfunc({out_descr}({in_descr}))"
