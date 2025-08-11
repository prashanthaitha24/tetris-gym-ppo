from stable_baselines3.common.callbacks import BaseCallback
class EpisodeMetrics(BaseCallback):
    def __init__(self): super().__init__(); self._lines=0; self._steps=0
    def _on_step(self) -> bool:
        info = self.locals.get("infos",[{}])[0]
        self._lines += int(info.get("lines_delta",0)); self._steps += 1
        if self.locals.get("dones",[False])[0]:
            self.logger.record("custom/episode_lines", self._lines)
            self.logger.record("custom/episode_len", self._steps)
            self._lines=0; self._steps=0
        return True
