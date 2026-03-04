from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    n_obs: int
    ped_count_min: int
    ped_count_max: int
    ped_ignore_robot_frac: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionConfig:
    success_threshold: float
    unsafe_threshold_max: float
    consecutive_evals: int
    cooldown_evals: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemotionConfig:
    enabled: bool
    success_threshold: float
    consecutive_evals: int
    cooldown_evals: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumConfig:
    enabled: bool
    metric: str
    promotion: PromotionConfig
    demotion: DemotionConfig
    stages: list[StageConfig]
    start_stage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumConfig":
        return _parse_curriculum_dict(data)


@dataclass
class CurriculumState:
    stage_idx: int
    up_streak: int = 0
    down_streak: int = 0
    cooldown_remaining: int = 0
    last_transition: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumState":
        return cls(
            stage_idx=int(data.get("stage_idx", 0)),
            up_streak=int(data.get("up_streak", 0)),
            down_streak=int(data.get("down_streak", 0)),
            cooldown_remaining=int(data.get("cooldown_remaining", 0)),
            last_transition=str(data.get("last_transition", "none")),
        )


def _require_dict(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Curriculum field '{field_name}' must be a mapping.")
    return data


def _require_int(data: dict[str, Any], key: str) -> int:
    if key not in data:
        raise ValueError(f"Curriculum missing required key '{key}'.")
    try:
        return int(data[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Curriculum key '{key}' must be an integer.") from exc


def _require_float(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"Curriculum missing required key '{key}'.")
    try:
        return float(data[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Curriculum key '{key}' must be a float.") from exc


def _parse_curriculum_dict(data: dict[str, Any]) -> CurriculumConfig:
    root = _require_dict(data, "root")
    enabled = bool(root.get("enabled", True))
    metric = str(root.get("metric", "success_rate"))
    if metric != "success_rate":
        raise ValueError(f"Unsupported curriculum metric '{metric}'. Expected 'success_rate'.")

    promotion_raw = _require_dict(root.get("promotion", {}), "promotion")
    promotion = PromotionConfig(
        success_threshold=_require_float(promotion_raw, "success_threshold"),
        unsafe_threshold_max=_require_float(promotion_raw, "unsafe_threshold_max"),
        consecutive_evals=_require_int(promotion_raw, "consecutive_evals"),
        cooldown_evals=_require_int(promotion_raw, "cooldown_evals"),
    )

    demotion_raw = _require_dict(root.get("demotion", {}), "demotion")
    demotion = DemotionConfig(
        enabled=bool(demotion_raw.get("enabled", False)),
        success_threshold=_require_float(demotion_raw, "success_threshold"),
        consecutive_evals=_require_int(demotion_raw, "consecutive_evals"),
        cooldown_evals=_require_int(demotion_raw, "cooldown_evals"),
    )

    stages_raw = root.get("stages")
    if not isinstance(stages_raw, list) or len(stages_raw) == 0:
        raise ValueError("Curriculum key 'stages' must be a non-empty list.")

    stages: list[StageConfig] = []
    for idx, stage_raw in enumerate(stages_raw):
        stage = _require_dict(stage_raw, f"stages[{idx}]")
        ped_ignore_robot_frac = stage.get("ped_ignore_robot_frac", None)
        if ped_ignore_robot_frac is not None:
            ped_ignore_robot_frac = float(ped_ignore_robot_frac)
        this_stage = StageConfig(
            n_obs=_require_int(stage, "n_obs"),
            ped_count_min=_require_int(stage, "ped_count_min"),
            ped_count_max=_require_int(stage, "ped_count_max"),
            ped_ignore_robot_frac=ped_ignore_robot_frac,
        )
        if this_stage.ped_count_min > this_stage.ped_count_max:
            raise ValueError(
                f"Invalid stage[{idx}]: ped_count_min ({this_stage.ped_count_min}) "
                f"> ped_count_max ({this_stage.ped_count_max})."
            )
        stages.append(this_stage)

    start_stage = int(root.get("start_stage", 0))
    start_stage = max(0, min(start_stage, len(stages) - 1))

    return CurriculumConfig(
        enabled=enabled,
        metric=metric,
        promotion=promotion,
        demotion=demotion,
        stages=stages,
        start_stage=start_stage,
    )


def load_curriculum_yaml(path: str | Path) -> CurriculumConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Curriculum config at '{path}' is empty.")
    if not isinstance(data, dict):
        raise ValueError(f"Curriculum config at '{path}' must be a mapping.")
    return _parse_curriculum_dict(data)


class CurriculumManager:
    def __init__(self, config: CurriculumConfig, state: CurriculumState | None = None):
        if len(config.stages) == 0:
            raise ValueError("Curriculum config must include at least one stage.")
        self.config = config
        self.state = state if state is not None else CurriculumState(stage_idx=config.start_stage)
        self.state.stage_idx = self._clamp_stage_idx(self.state.stage_idx)

    def _clamp_stage_idx(self, idx: int) -> int:
        return max(0, min(int(idx), len(self.config.stages) - 1))

    @property
    def max_stage_idx(self) -> int:
        return len(self.config.stages) - 1

    @property
    def current_stage(self) -> StageConfig:
        return self.config.stages[self.state.stage_idx]

    def stage_log_values(self) -> dict[str, float | int]:
        stage = self.current_stage
        return {
            "curriculum/n_obs": int(stage.n_obs),
            "curriculum/ped_count_min": int(stage.ped_count_min),
            "curriculum/ped_count_max": int(stage.ped_count_max),
        }

    def apply_stage(self, env: Any, env_test: Any, stage_idx: int | None = None) -> StageConfig:
        if stage_idx is None:
            stage_idx = self.state.stage_idx
        stage_idx = self._clamp_stage_idx(stage_idx)
        self.state.stage_idx = stage_idx
        stage = self.current_stage
        for this_env in (env, env_test):
            params = this_env.params
            if "n_obs" not in params:
                raise KeyError("Environment params missing 'n_obs' required by curriculum.")
            if "ped_count_min" not in params:
                raise KeyError("Environment params missing 'ped_count_min' required by curriculum.")
            if "ped_count_max" not in params:
                raise KeyError("Environment params missing 'ped_count_max' required by curriculum.")
            params["n_obs"] = int(stage.n_obs)
            params["ped_count_min"] = int(stage.ped_count_min)
            params["ped_count_max"] = int(stage.ped_count_max)
            if stage.ped_ignore_robot_frac is not None:
                if "ped_ignore_robot_frac" not in params:
                    raise KeyError("Environment params missing 'ped_ignore_robot_frac' required by curriculum.")
                params["ped_ignore_robot_frac"] = float(stage.ped_ignore_robot_frac)
        return stage

    def update_on_eval(self, success_rate: float, unsafe_frac: float) -> dict[str, Any]:
        transition = "none"
        stage_changed = False
        prev_stage = self.state.stage_idx

        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            self.state.up_streak = 0
            self.state.down_streak = 0
            self.state.last_transition = transition
            return {
                "stage_changed": stage_changed,
                "transition": transition,
                "prev_stage_idx": prev_stage,
                "stage_idx": self.state.stage_idx,
            }

        promote_hit = (
            float(success_rate) >= float(self.config.promotion.success_threshold)
            and float(unsafe_frac) <= float(self.config.promotion.unsafe_threshold_max)
        )
        demote_hit = (
            bool(self.config.demotion.enabled)
            and float(success_rate) < float(self.config.demotion.success_threshold)
        )

        self.state.up_streak = self.state.up_streak + 1 if promote_hit else 0
        self.state.down_streak = self.state.down_streak + 1 if demote_hit else 0

        if self.state.up_streak >= self.config.promotion.consecutive_evals:
            if self.state.stage_idx < self.max_stage_idx:
                self.state.stage_idx += 1
                stage_changed = True
                transition = "up"
                self.state.cooldown_remaining = max(0, int(self.config.promotion.cooldown_evals))
            self.state.up_streak = 0
            self.state.down_streak = 0
        elif self.state.down_streak >= self.config.demotion.consecutive_evals:
            if self.config.demotion.enabled and self.state.stage_idx > 0:
                self.state.stage_idx -= 1
                stage_changed = True
                transition = "down"
                self.state.cooldown_remaining = max(0, int(self.config.demotion.cooldown_evals))
            self.state.up_streak = 0
            self.state.down_streak = 0

        self.state.stage_idx = self._clamp_stage_idx(self.state.stage_idx)
        self.state.last_transition = transition
        return {
            "stage_changed": stage_changed,
            "transition": transition,
            "prev_stage_idx": prev_stage,
            "stage_idx": self.state.stage_idx,
        }

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": self.state.to_dict()}
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    @staticmethod
    def load_state(path: str | Path) -> CurriculumState:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict) or "state" not in payload:
            raise ValueError(f"Invalid curriculum state file: {path}")
        state_data = payload["state"]
        if not isinstance(state_data, dict):
            raise ValueError(f"Invalid curriculum state payload in: {path}")
        return CurriculumState.from_dict(state_data)
