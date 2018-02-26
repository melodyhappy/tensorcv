from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorcv.logger import logger


class SecondOrPerfactStepTimer(SecondOrStepTimer):
    def should_trigger_for_step(self, step):
        if self._last_triggered_step is None:
            return True

        if self._last_triggered_step == step:
            return False

        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True

        if self._every_steps is not None:
            if step % self._every_steps == 0:
                return True

        return False


class CheckpointPerfactSaverHook(CheckpointSaverHook):
    def __init__(self,
                 checkpoint_dir,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 scaffold=None,
                 listeners=None):
        super(CheckpointPerfactSaverHook, self).__init__(
            checkpoint_dir, save_secs, save_steps, saver,
            checkpoint_basename, scaffold, listeners)
        self._timer = SecondOrPerfactStepTimer(every_secs=save_secs,
                                               every_steps=save_steps)

