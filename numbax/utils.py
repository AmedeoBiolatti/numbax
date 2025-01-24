import jax


def jaxpr_var_name(v, with_typing=False) -> str:
    name = str(v)
    if isinstance(v, jax.core.Var):
        name = name.split(":")[0]
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = name.replace("=", "_")

        type_name = str(v.aval)
    if isinstance(v, jax.core.Literal):
        dtype = v.aval.dtype
        val = v.val
        name = f"np.{dtype}({val})"

    if with_typing:
        typing_str = f" : '{type_name}'"
        name = name + typing_str

    return name


def jaxpr_vars_name(v, with_typing=False) -> list[str]:
    return [jaxpr_var_name(vi, with_typing=with_typing and len(v) == 1) for vi in v]
