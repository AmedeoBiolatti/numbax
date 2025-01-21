import jax

from numbax.primitives import _numpy_primitive_mapping

with open("primitives_progress.md", "w") as file:
    for name in sorted(jax.lax.__dir__()):
        if name.endswith('_p'):
            symbol = "x" if getattr(jax.lax, name) in _numpy_primitive_mapping else " "
            file.write(f"- [{symbol}] {name}\n")
