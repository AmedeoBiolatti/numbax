import gc

import jax
import time
import numpy as np

import numbax
from numbax import numpize

N = 100


def fn(x):
    def step(i, x):
        x = jax.numpy.cos(x + i)
        x = jax.numpy.sin(x + i)
        return x

    x = jax.lax.fori_loop(0, N, step, x)
    x = x * 2 + 1
    return x


jaxpr = jax.make_jaxpr(fn)(1.)
fn_loop_numpized = numpize.numpize_jaxpr(jaxpr)

print("| | Compile + first run | 100 runs |")
print("|--|--|--|")
for (f_, f_name) in [
    (jax.jit(fn), 'jax-jit'),
    (fn_loop_numpized, 'numbax:numpize'),
    (numbax.jit(fn), 'numbax:jit'),
    (numbax.jit(fn, fast=True), 'numbax:jit-fastmath'),
]:
    X = np.zeros((1024,), dtype=np.float32)
    if "jax" in f_name:
        X = jax.device_put(X)

    try:
        t0 = time.perf_counter()
        f_(X)
        t1 = time.perf_counter()

        times = []
        for _ in range(100):
            t2 = time.perf_counter()
            for _ in range(100):
                out = f_(X)
                if isinstance(out, jax.Array):
                    out.block_until_ready()
            t3 = time.perf_counter()
            times.append(t3 - t2)
        time.sleep(0.001)

        times = np.array(times)
        print("| %20s  |  %.4f  |  %.4f (+-%.4f)  |" % (f_name, t1 - t0, np.mean(times), np.std(times)))
        time.sleep(0.1)
    except Exception as e:
        print(e)
