from .base import MultiAgentController
from .dec_share_cbf import DecShareCBF
from .gcbf import GCBF
from .gcbf_plus import GCBFPlus
from .centralized_cbf import CentralizedCBF
from .ppo import PPO


def make_algo(algo: str, **kwargs) -> MultiAgentController:
    if algo == 'gcbf':
        return GCBF(**kwargs)
    elif algo == 'gcbf+':
        return GCBFPlus(**kwargs)
    elif algo == 'centralized_cbf':
        return CentralizedCBF(**kwargs)
    elif algo == 'dec_share_cbf':
        return DecShareCBF(**kwargs)
    elif algo == 'ppo':
        return PPO(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
