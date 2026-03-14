import os
import numpy as np
import jax
import jax.random as jr
import functools as ft

from time import time
from tqdm import tqdm

from .data import Rollout
from .curriculum import CurriculumConfig, CurriculumManager, CurriculumState
from .utils import rollout, rollout_rnn
from ..env import MultiAgentEnv
from ..algo.base import MultiAgentController
from ..utils.utils import jax_vmap
from ..utils.wandb_compat import wandb


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: MultiAgentController,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True,
            start_step: int = 0
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='higcbf', dir=self.log_dir)

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.start_step = start_step
        self.update_steps = start_step
        self.key = jax.random.PRNGKey(seed)
        self.timesteps_per_update = self.n_env_train * self.env.max_episode_steps
        self.total_timesteps = 0
        self.curriculum_state_path = os.path.join(self.log_dir, "curriculum_state.yaml")
        self.curriculum: CurriculumManager | None = None
        self._init_curriculum()

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def _init_curriculum(self) -> None:
        cfg = self.params.get("curriculum_config", None)
        if cfg is None:
            return
        if isinstance(cfg, dict):
            cfg = CurriculumConfig.from_dict(cfg)
        if not isinstance(cfg, CurriculumConfig):
            raise ValueError("curriculum_config must be a CurriculumConfig or a dict.")
        if not cfg.enabled:
            return

        state: CurriculumState | None = None
        if self.start_step > 0 and os.path.exists(self.curriculum_state_path):
            state = CurriculumManager.load_state(self.curriculum_state_path)

        if state is None:
            start_stage = self.params.get("curriculum_start_stage", None)
            if start_stage is None:
                start_stage = cfg.start_stage
            state = CurriculumState(stage_idx=int(start_stage))

        self.curriculum = CurriculumManager(cfg, state)
        self.curriculum.apply_stage(self.env, self.env_test)
        if self.save_log:
            self.curriculum.save_state(self.curriculum_state_path)

    def _build_rollout_and_test_fns(self):
        if getattr(self.algo, "use_gru", False):
            # preprocess recurrent rollout function
            def rollout_fn_single(params, key):
                init_carry = self.algo.init_actor_carry()
                actor = lambda graph, k, carry, done_prev: self.algo.step_rnn(
                    graph, k, carry, done_prev, params=params
                )
                return rollout_rnn(self.env, actor, init_carry, key)

            def rollout_fn(params, keys):
                return jax.vmap(ft.partial(rollout_fn_single, params))(keys)

            rollout_fn = jax.jit(rollout_fn)

            # preprocess recurrent test function
            def test_fn_single(params, key):
                init_carry = self.algo.init_actor_carry()

                def actor(graph, k, carry, done_prev):
                    action, new_carry = self.algo.act_rnn(graph, carry, done_prev, params=params)
                    return action, jax.numpy.array(0.0), new_carry

                return rollout_rnn(self.env_test, actor, init_carry, key)

            def test_fn(params, keys):
                return jax.vmap(ft.partial(test_fn_single, params))(keys)

            test_fn = jax.jit(test_fn)
        else:
            # preprocess feed-forward rollout function
            def rollout_fn_single(params, key):
                return rollout(self.env, ft.partial(self.algo.step, params=params), key)

            def rollout_fn(params, keys):
                return jax.vmap(ft.partial(rollout_fn_single, params))(keys)

            rollout_fn = jax.jit(rollout_fn)

            # preprocess feed-forward test function
            def test_fn_single(params, key):
                return rollout(self.env_test, lambda graph, k: (self.algo.act(graph, params), None), key)

            def test_fn(params, keys):
                return jax.vmap(ft.partial(test_fn_single, params))(keys)

            test_fn = jax.jit(test_fn)

        finish_fn = jax_vmap(jax_vmap(self.env_test.finish_mask))
        return rollout_fn, test_fn, finish_fn

    def _sample_reward_components(self, key: jax.Array) -> dict:
        """
        Run one eval episode Python-side (not jitted) with get_eval_info=True
        to collect per-component reward averages for wandb logging.
        """
        try:
            env = self.env_test
            graph = env.reset(key)
            components_acc: dict[str, list] = {}
            for _ in range(env.max_episode_steps):
                action = self.algo.act(graph, self.algo.actor_params)
                next_graph, _reward, _cost, done, info = env.step(graph, action, get_eval_info=True)
                for k, v in info.items():
                    try:
                        components_acc.setdefault(k, []).append(float(np.asarray(v)))
                    except Exception:
                        pass
                graph = next_graph
                if bool(done):
                    break
            return {f"eval/r/{k}": float(np.mean(vs)) for k, vs in components_acc.items() if vs}
        except Exception:
            return {}

    def train(self):
        # record start time
        start_time = time()
        rollout_fn, test_fn, finish_fn = self._build_rollout_and_test_fns()

        # start training
        test_key = jr.PRNGKey(self.seed)
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar_total = max(self.steps - self.start_step, 0)
        pbar = tqdm(total=pbar_total, ncols=80)
        for step in range(self.start_step, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                t_eval0 = time()
                test_rollouts: Rollout = test_fn(self.algo.actor_params, test_keys)
                reward_reduce_axes = tuple(range(1, test_rollouts.rewards.ndim))
                total_return = test_rollouts.rewards.sum(axis=reward_reduce_axes)
                assert total_return.shape == (self.n_env_test,)
                reward_min, reward_max = total_return.min(), total_return.max()
                reward_mean = np.mean(total_return)
                reward_mean_per_step = test_rollouts.rewards.mean(axis=1).mean(axis=-1).mean()
                finish_env = np.asarray(finish_fn(test_rollouts.graph).max(axis=1), dtype=bool)
                finish = finish_env.mean()
                cost = test_rollouts.costs.sum(axis=-1).mean()
                unsafe_env = np.asarray(test_rollouts.costs.max(axis=-1) >= 1e-6, dtype=bool)
                unsafe_frac = np.mean(unsafe_env)
                success_rate = np.mean(np.logical_and(np.logical_not(unsafe_env), finish_env))
                eval_info = {
                    "eval/reward": reward_mean,
                    "eval/reward_per_step": reward_mean_per_step,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/finish": finish,
                    "eval/success": success_rate,
                    "time/eval_s": time() - t_eval0,
                    "step": step,
                    "Charts/total_timesteps": self.total_timesteps,
                }

                curriculum_msg = ""
                if self.curriculum is not None:
                    update = self.curriculum.update_on_eval(success_rate=float(success_rate), unsafe_frac=float(unsafe_frac))
                    eval_info["curriculum/stage_idx"] = int(update["stage_idx"])
                    eval_info["curriculum/transition"] = update["transition"]
                    eval_info["curriculum/eval_success_rate"] = float(success_rate)
                    eval_info["curriculum/eval_unsafe_frac"] = float(unsafe_frac)
                    eval_info.update(self.curriculum.stage_log_values())

                    if update["stage_changed"]:
                        self.curriculum.apply_stage(self.env, self.env_test, int(update["stage_idx"]))
                        rollout_fn, test_fn, finish_fn = self._build_rollout_and_test_fns()
                        curriculum_msg = (
                            f", curriculum: {update['transition']} "
                            f"{int(update['prev_stage_idx'])}->{int(update['stage_idx'])}"
                        )
                    if self.save_log:
                        self.curriculum.save_state(self.curriculum_state_path)

                reward_component_info = self._sample_reward_components(test_key)
                eval_info.update(reward_component_info)

                wandb.log(eval_info, step=self.update_steps)
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'reward/step: {reward_mean_per_step:7.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}, finish: {finish:6.2f}, '
                                f'success: {success_rate:6.2f}, total timesteps: {self.total_timesteps}'
                                f'{curriculum_msg}')
                tqdm.write(eval_verbose)
                if self.save_log and step % self.save_interval == 0:
                    self.algo.save(os.path.join(self.model_dir), step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            t_rollout0 = time()
            rollouts: Rollout = rollout_fn(self.algo.actor_params, key_x0)
            t_rollout = time() - t_rollout0

            # update the algorithm
            t_update0 = time()
            update_info = self.algo.update(rollouts, step)
            t_update = time() - t_update0
            self.total_timesteps += self.timesteps_per_update
            update_info["time/rollout_s"] = t_rollout
            update_info["time/update_s"] = t_update
            update_info["time/iter_s"] = t_rollout + t_update
            update_info["train/total_timesteps"] = self.total_timesteps
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
