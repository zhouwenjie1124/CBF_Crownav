import copy
from typing import Optional

from .base import MultiAgentEnv
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .linear_drone import LinearDrone
from .dubins_car import DubinsCar
from .crazyflie import CrazyFlie
from .robot_ped_env import RobotPedEnv


ENV = {
    'SingleIntegrator': SingleIntegrator,
    'DoubleIntegrator': DoubleIntegrator,
    'LinearDrone': LinearDrone,
    'DubinsCar': DubinsCar,
    'CrazyFlie': CrazyFlie,
    'RobotPedEnv': RobotPedEnv,
}


DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        num_ped: Optional[int] = None,
        ped_ignore_robot_frac: Optional[float] = None,
        env_params: Optional[dict] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = copy.deepcopy(ENV[env_id].PARAMS if env_params is None else env_params)
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    if num_ped is not None:
        params['n_ped'] = num_ped
    if ped_ignore_robot_frac is not None:
        params['ped_ignore_robot_frac'] = ped_ignore_robot_frac
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.1,
        params=params
    )
