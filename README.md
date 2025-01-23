# numbax

An experimental numba compiler for JAX

### Quickstart

```python
import jax
import numbax


@numbax.jit
def fn(x):
    return jax.numpy.sin(x) + 1


fn(1)
```

### Mini-benchmark

The mini-benchmark in `examples/minimal_example.py` returns the following times (in seconds on CPU)

|                     | Compile + first run | 100 runs              |
|---------------------|---------------------|-----------------------|
| jax-jit             | 0.0393              | 0.1052 ($\pm$ 0.0029) |
| numbax:numpize      | 0.0005              | 0.0379 ($\pm$ 0.0009) |
| numbax:jit          | 0.2435              | 0.0116 ($\pm$ 0.0002) |
| numbax:jit-fastmath | 0.1798              | 0.0116 ($\pm$ 0.0002) |

# WARNING: Very Early Development 🚧

This project is in its very early stages of development. Features may be incomplete, APIs unstable, and breaking changes
likely. Use at your own risk!
