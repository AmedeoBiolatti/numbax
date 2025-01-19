# numbax

An experimental numba compiler for JAX

### Quickstart

*coming soon*: The decorator is not working yet but will be available soon.

```python
import jax
import numbax


@numbax.jit
def fn(x):
    return jax.numpy.sin(x) + 1


fn(1)
```

### Mini-benchmark

The mini-benchmark in `examples/minimal_example.py` returns the following times (in seconds)

|                       | Compile + first run | 100 successive runs | 
|-----------------------|---------------------|---------------------|
| jax                   | 0.162113            | 2.724355            |
| jax-jit               | 0.059200            | 0.021689            |
| jax *(fori_loop)*     | 0.060702            | 4.823792            |
| jax-jit *(fori_loop)* | 0.040832            | 0.006865            |
| numpy                 | 0.000317            | 0.019987            |
| numba                 | 9.049266            | 0.005530            |
| numba-fast            | 11.035710           | 0.004697            |

# WARNING: Very Early Development 🚧

This project is in its very early stages of development. Features may be incomplete, APIs unstable, and breaking changes
likely. Use at your own risk!
