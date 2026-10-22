from habitat_baselines.common.baseline_registry import baseline_registry

from ..common.base_il_trainer import BaseVLNCETrainer


@baseline_registry.register_trainer(name="navjev")
class NavJevEvaluator(BaseVLNCETrainer):

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()
