from car_env import Car
from brax import envs
import jax
from jax import numpy as jp
import time
import os
from mujoco import viewer, mjx
import mujoco
import copy

jp.set_printoptions(precision=2, suppress=True)


def sstep(diffwrt, dobj, sys):
    """
    Diffwrt: [state, ctrl]. Zero-D array.
    dobj: mjxData
    S for separated step. 
    """

    dobj = dobj.tree_replace({
        'qpos': diffwrt[:sys.nq],
        'qvel': diffwrt[sys.nq:sys.nq + sys.nv],
        'ctrl': diffwrt[sys.nq + sys.nv:]
    })

    dobj = jmjxstep(sys, dobj)

    state = jp.squeeze(
        jp.concatenate(
            [jp.expand_dims(dobj.qpos, 1),
             jp.expand_dims(dobj.qvel, 1)],
            axis=0))
    return state


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
jmjxstep = jax.jit(mjx.step)
fjacstep = jax.jit(jax.jacfwd(sstep, 0))

state = jit_reset(jax.random.PRNGKey(0))

model = copy.copy(env.mj_model)
data = mujoco.MjData(model)
viewer = viewer.launch_passive(model, data)
mjx.get_data_into(data, model, state.pipeline_state)
viewer.sync()

state_dim = env.sys.nq + env.sys.nv
ctrl_dim = env.sys.nu

while viewer.is_running():
    cur_time = time.time()
    ctrl = 0.2 * jp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    x_i = jp.squeeze(
        jp.concatenate([
            state.pipeline_state.qpos.reshape(env.sys.nq),
            state.pipeline_state.qvel.reshape(env.sys.nv), ctrl
        ],
                       axis=0))
    assert len(x_i) == state_dim + ctrl_dim
    cur_jac = fjacstep(x_i, state.pipeline_state, env.sys)  # 17x19
    state_jac = cur_jac[:, :state_dim]  # 17x17
    ctrl_jac = cur_jac[:, state_dim:state_dim + ctrl_dim]  # 17x2
    # print(state_jac.shape, ctrl_jac.shape)
    # print(state_jac)

    mjx.get_data_into(data, model, state.pipeline_state)
    viewer.sync()
    # time.sleep(0.01)

    print(time.time() - cur_time)
    cur_time = time.time()
