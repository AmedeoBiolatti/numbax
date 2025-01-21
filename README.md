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

|                      | Compile + first run | 100 successive runs |
|----------------------|---------------------|---------------------|
| jax                  | 0.088991            | 1.660780            |
| jax-jit              | 0.056053            | 0.049823            |
| numpy                | 0.000353            | 0.025763            |
| numba                | 16.327550           | 0.011057            |
| numba-fast           | 16.965312           | 0.010609            |
| jax:fori_loop        | 0.030407            | 2.830550            |
| jax-jit:fori_loop    | 0.023105            | 0.014057            |
| numpy:fori_loop      | 0.000431            | 0.031376            |
| numba:fori_loop      | 0.132297            | 0.005540            |
| numba-fast:fori_loop | 0.131045            | 0.005505            |

# WARNING: Very Early Development 🚧

This project is in its very early stages of development. Features may be incomplete, APIs unstable, and breaking changes
likely. Use at your own risk!
