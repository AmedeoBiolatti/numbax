import jax


def jaxpr_var_name(v) -> str:
    name = str(v)
    if isinstance(v, jax.core.Var):
        name = name.split(":")[0]
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = name.replace("=", "_")
    if isinstance(v, jax.core.Literal):
        dtype = v.aval.dtype
        val = v.val
        name = f"np.{dtype}({val})"
    return name


def jaxpr_vars_name(v) -> list[str]:
    return [jaxpr_var_name(vi) for vi in v]
