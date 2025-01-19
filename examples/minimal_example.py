import jax
import numba
import time
import numpy as np

from numbax import numpize


def fn(x):
    def step(carry, x):
        y = carry + x
        return carry, y

    stacked, x = jax.lax.scan(step, x, xs=jax.numpy.arange(100))

    for _ in range(100):
        x = jax.numpy.cos(x)
        x = jax.numpy.sin(x)

    x = x * 2 + 1
    return x


jaxpr = jax.make_jaxpr(fn)(1)

fn_numpized = numpize.numpize_jaxpr(jaxpr)

fn_numba = numba.njit(fn_numpized, parallel=False, fastmath=False)
fn_numba_fast = numba.njit(fn_numpized, parallel=False, fastmath=True)

for (f_, f_name) in [
    (fn, 'jax'),
    (jax.jit(fn), 'jax-jit'),
    (fn_numpized, 'numpy'),
    (fn_numba, 'numba'),
    (fn_numba_fast, 'numba-fast')
]:
    t0 = time.perf_counter()
    f_(np.int32(1))
    t1 = time.perf_counter()
    for _ in range(100):
        f_(np.int32(1))
    t2 = time.perf_counter()

    print("%10s\t%.6f\t%.6f" % (f_name, t2 - t1, t1 - t0))
    time.sleep(0.1)

fn(np.int32(1))
fn_numpized(np.int32(1))
fn_numba(np.int32(1))
