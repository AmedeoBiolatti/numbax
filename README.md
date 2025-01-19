# numbax

An experimental numba compiler for JAX

### Quickstart

*coming soon*: the decorator is currently not working yet, but it will be available soon

```python
import jax
import numbax


@numbax.jit
def fn(x):
    return jax.numpy.sin(x) + 1


fn(1)
```

### Mini-benchmark

The mini-benchmark in `examples/minimal_example.py` returns the folling times

| Test Case | Compile + first run (seconds) | 100 successive runs (seconds) | 
|-----------|-------------------------------|------------------------|
| JAX           | 0.210734                      | 2.960469               |
| JAX-JIT       | 0.074698                      | 0.023648               |
| NumPy         | 0.000551                      | 0.026877               |
| Numba         | 9.911505                      | 0.005360               |
| Numba-Fast    | 10.324011                     | 0.004864               |


# WARNING: Very Early Development 🚧

This project is in its very early stages of development. Features may be incomplete, APIs unstable, and breaking changes
likely. Use at your own risk!
