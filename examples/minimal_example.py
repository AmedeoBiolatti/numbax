import gc

import jax
import numba
import time
import numpy as np

import numbax
from numbax import numpize

N = 1000


def fn(x):
    # def step(carry, x):
    #     y = carry + x
    #     return carry, y
    #
    # stacked, x = jax.lax.scan(step, x, xs=jax.numpy.arange(N))

    for _ in range(N):
        x = jax.numpy.cos(x)
        x = jax.numpy.sin(x)

    x = x * 2 + 1
    return x


def fn_loop(x):
    # def step(carry, x):
    #     y = carry + x
    #     return carry, y
    #
    # stacked, x = jax.lax.scan(step, x, xs=jax.numpy.arange(N))

    def step(i, x):
        x = jax.lax.cond(i >= 10, lambda: jax.numpy.sin(x), lambda: jax.numpy.cos(x))
        x = jax.numpy.cos(x)
        x = jax.numpy.sin(x)
        return x

    x = jax.lax.fori_loop(0, N, step, x)

    x = x * 2 + 1
    return x


jaxpr = jax.make_jaxpr(fn)(1.)
fn_numpized = numpize.numpize_jaxpr(jaxpr)

jaxpr = jax.make_jaxpr(fn_loop)(1.)
fn_loop_numpized = numpize.numpize_jaxpr(jaxpr)

print("| | Compile + first run | 100 successive runs |")
print("|--|--|--|")
for (f_, f_name) in [
    # (fn, 'jax'),
    # (jax.jit(fn), 'jax-jit'),
    # (fn_numpized, 'numpy'),
    # (numbax.jit(fn), 'numba'),
    # (numbax.jit(fn, fast=True), 'numba-fast'),
    # (fn_loop, 'jax:fori_loop'),
    (jax.jit(fn_loop), 'jax-jit:fori_loop'),
    (fn_loop_numpized, 'numpy:fori_loop'),
    (numbax.jit(fn_loop), 'numba:fori_loop'),
    (numbax.jit(fn_loop, fast=True), 'numba-fast:fori_loop'),
]:
    # X = np.zeros((100,), dtype=np.float32)
    X = np.float32(0.5)
    if "jax" in f_name:
        X = jax.device_put(X)

    try:
        t0 = time.perf_counter()
        f_(X)
        t1 = time.perf_counter()

        times = []
        for _ in range(20):
            t2 = time.perf_counter()
            for _ in range(100):
                out = f_(X)
                if isinstance(out, jax.Array):
                    out.block_until_ready()
            t3 = time.perf_counter()
            times.append(t3 - t2)
            gc.collect()
            time.sleep(0.0001)

        print("| %20s  |  %.6f  |  %.6f (+-%.6f)  |" % (f_name, t1 - t0, 1000 * np.mean(times), 2000 * np.std(times)))
        time.sleep(0.1)
    except Exception as e:
        print(e)
