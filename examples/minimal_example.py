import jax
import numba
import time
import numpy as np

from numbax import numpize

N = 200


def fn(x):
    def step(carry, x):
        y = carry + x
        return carry, y

    stacked, x = jax.lax.scan(step, x, xs=jax.numpy.arange(N))

    for _ in range(N):
        x = jax.numpy.cos(x)
        x = jax.numpy.sin(x)

    x = x * 2 + 1
    return x


def fn_loop(x):
    def step(carry, x):
        y = carry + x
        return carry, y

    stacked, x = jax.lax.scan(step, x, xs=jax.numpy.arange(N))

    def step(_, x):
        x = jax.numpy.cos(x)
        x = jax.numpy.sin(x)
        return x

    x = jax.lax.fori_loop(0, N, step, x)

    x = x * 2 + 1
    return x


jaxpr = jax.make_jaxpr(fn)(1.)
fn_numpized = numpize.numpize_jaxpr(jaxpr)
fn_numba = numba.njit(fn_numpized, parallel=False, fastmath=False)
fn_numba_fast = numba.njit(fn_numpized, parallel=False, fastmath=True)

jaxpr = jax.make_jaxpr(fn_loop)(1.)
fn_loop_numpized = numpize.numpize_jaxpr(jaxpr)
fn_loop_numba = numba.njit(fn_loop_numpized, parallel=False, fastmath=False)
fn_loop_numba_fast = numba.njit(fn_loop_numpized, parallel=False, fastmath=True)

print("| | Compile + first run | 100 successive runs |")
print("|--|--|--|")
for (f_, f_name) in [
    (fn, 'jax'),
    (jax.jit(fn), 'jax-jit'),
    (fn_numpized, 'numpy'),
    (fn_numba, 'numba'),
    (fn_numba_fast, 'numba-fast'),
    (fn_loop, 'jax:fori_loop'),
    (jax.jit(fn_loop), 'jax-jit:fori_loop'),
    (fn_loop_numpized, 'numpy:fori_loop'),
    (fn_loop_numba, 'numba:fori_loop'),
    (fn_loop_numba_fast, 'numba-fast:fori_loop'),
]:
    try:
        t0 = time.perf_counter()
        f_(np.float32(1.))
        t1 = time.perf_counter()
        for _ in range(100):
            out = f_(np.float32(1.))
            if isinstance(out, jax.Array):
                out.block_until_ready()
        t2 = time.perf_counter()

        print("| %20s  |  %.6f  |  %.6f |" % (f_name, t1 - t0, t2 - t1))
        time.sleep(0.1)
    except Exception as e:
        print(e)
