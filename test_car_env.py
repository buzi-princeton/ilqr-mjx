from car_env import Car
from brax import envs
import jax
from jax import numpy as jp
import time
import os
from mujoco import viewer, mjx
import mujoco
import copy

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

env_name = 'car'
try:
    env = envs.get_environment(env_name)
except:
    envs.register_environment(env_name, Car)
    env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))

model = copy.copy(env.mj_model)
data = mujoco.MjData(model)
viewer = viewer.launch_passive(model, data)
mjx.get_data_into(data, model, state.pipeline_state)
viewer.sync()

while viewer.is_running():
    ctrl = 0.1 * jp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    mjx.get_data_into(data, model, state.pipeline_state)
    viewer.sync()
    time.sleep(0.01)
