import time
import numpy as np

import jax
import optax
import numbax

x = np.random.normal(size=(10_000, 10)).astype("float32")
y = np.reshape(x[:, 0] + 0.5 * x[:, 1], (-1, 1))
y = np.random.normal(y, 0.1).astype("float32")

params = (
    jax.numpy.zeros((1, 10)),
    jax.numpy.zeros((1, 1))
)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)


def loss_fn(params, x, y):
    w, b = params
    p = jax.numpy.sum(w * x, axis=-1, keepdims=True) + b
    mse = jax.numpy.mean((y - p) ** 2)
    return mse


vg = jax.value_and_grad(loss_fn)


def train_step(_, state):
    params, opt_state, x, y = state
    loss_value, updates = vg(params, x, y)
    updates, opt_state = optimizer.update(updates, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, x, y


def train_loop(params, opt_state, x, y):
    params, opt_state, _, _ = jax.lax.fori_loop(0, 10_000, train_step, (params, opt_state, x, y))
    return params, opt_state


train_jax = jax.jit(train_loop)
train_numbax = numbax.jit(train_loop)

t0 = time.perf_counter()
res = train_jax(params, opt_state, x, y)[0][0]
jax.tree.map(lambda t: t.block_until_ready(), res)
t1 = time.perf_counter()
res = train_jax(params, opt_state, x, y)
jax.tree.map(lambda t: t.block_until_ready(), res)
t2 = time.perf_counter()

_ = train_numbax(params, opt_state, x, y)
t3 = time.perf_counter()
res = train_numbax(params, opt_state, x, y)
t4 = time.perf_counter()

#
print("jax:jit    | %9.6f | %9.6f" % (t1 - t0, t2 - t1))
print("numbax:jit | %9.6f | %9.6f" % (t3 - t2, t4 - t3))
