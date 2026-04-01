"""
SB3-compatible training callbacks.

CurriculumCallback
------------------
Reads ``curriculum_stage_idx`` and ``curriculum_stage`` from the step info
dicts and writes them to the SB3 logger (TensorBoard / W&B via sb3 logger).
"""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Logs the active curriculum stage at every environment step.

    Writes two scalars:
      ``curriculum/stage``       — integer index (0 = easy, 1 = medium, 2 = hard)
      ``curriculum/stage_name``  — string label (not all loggers support strings;
                                   written only when the value changes)

    Usage
    -----
        from quantflow.training.callbacks import CurriculumCallback
        cb = CurriculumCallback(verbose=0)
        model.learn(total_timesteps=50_000, callback=cb)
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._last_stage: int = -1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        if not infos or "curriculum_stage_idx" not in infos[0]:
            return True

        stage     = int(infos[0]["curriculum_stage_idx"])
        stage_str = str(infos[0].get("curriculum_stage", "unknown"))

        self.logger.record("curriculum/stage", stage)

        if stage != self._last_stage:
            if self.verbose >= 1:
                print(
                    f"[CurriculumCallback] step {self.num_timesteps}: "
                    f"stage → {stage_str} ({stage})"
                )
            self._last_stage = stage

        return True
