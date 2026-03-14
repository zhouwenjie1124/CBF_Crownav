from .base import MultiAgentController
from .dec_share_cbf import DecShareCBF
from .ppo import PPO


def make_algo(algo: str, **kwargs) -> MultiAgentController:
    if algo == "dec_share_cbf":
        return DecShareCBF(**kwargs)
    if algo == "ppo":
        return PPO(**kwargs)
    raise ValueError(f"Unknown algorithm: {algo}")
