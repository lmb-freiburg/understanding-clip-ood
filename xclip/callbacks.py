import os
import subprocess
import typing
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override

from xclip.learner import ImageNetCaptionsLearner


class LogSpacedCheckpoint(pl.Callback):
    def __init__(self, save_dir: str = 'checkpoints', save_last: bool = True, save_weights_only: bool = True) -> None:
        self.save_dir = save_dir
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.next_save_step = 1

    @override
    @rank_zero_only
    def setup(self, trainer: pl.Trainer, learner: ImageNetCaptionsLearner, stage: str) -> None:
        if stage == 'fit':
            os.makedirs(self.save_dir, exist_ok=True)

    def _checkpoint_name(self, trainer: pl.Trainer) -> str:
        return os.path.join(self.save_dir, f'epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt')

    @override
    def on_fit_start(self, trainer: pl.Trainer, learner: ImageNetCaptionsLearner) -> None:
        assert trainer.global_step == 0
        trainer.save_checkpoint(self._checkpoint_name(trainer), weights_only=self.save_weights_only)

    @override
    def on_train_batch_end(self, trainer: pl.Trainer, learner: ImageNetCaptionsLearner, *_) -> None:
        current_step = trainer.global_step
        if current_step >= self.next_save_step or (self.save_last and current_step == trainer.max_steps):
            trainer.save_checkpoint(self._checkpoint_name(trainer), weights_only=self.save_weights_only)
            self.next_save_step *= 2


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, save_first: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_first = save_first

    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert trainer.global_step == 0
        if self.save_first:
            dirpath = typing.cast(str, self.dirpath)
            trainer.save_checkpoint(os.path.join(dirpath, 'epoch=0-step=0.ckpt'), weights_only=self.save_weights_only)


class CudaMemoryMonitor(pl.Callback):
    """
    Callback that periodically queries gpu memory statistics using nvidia-smi.

    Arguments:
        strict: Optional boolean flag that determines behaviour if nvidia-smi is not available. If strict is set to
            true an exception will be raised if nvidia-smi is not found. If set to false only a warning will be
            raised and the callback will just do nothing. Default: `False`.
    """

    def __init__(self, log_every_n_steps: int = 50, strict: bool = False) -> None:
        self.log_every_n_steps = log_every_n_steps

        # check if nvidia-smi is available
        self.nvidia_smi_available = None
        try:
            subprocess.check_output('which nvidia-smi', shell=True)
            self.nvidia_smi_available = True
        except subprocess.CalledProcessError:
            if strict:
                raise RuntimeError(f'nvidia-smi not found. Cannot create {self.__class__.__name__} with strict=True.')
            else:
                warnings.warn(f'nvidia-smi not found. {self.__class__.__name__} will not log any memory stats.')
            self.nvidia_smi_available = False

        assert self.nvidia_smi_available is not None

        # query number of visible cuda devices
        self.num_visible_devices = int(subprocess.check_output('nvidia-smi --list-gpus | wc -l', shell=True))

        # command to query cuda memory usage
        self.nvidia_smi_cmd = 'nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv'

        # sanity check that the expected stats are available
        if self.nvidia_smi_available:
            memory_stats = self.get_memory_stats()
            assert 'memory.used [MiB]' in memory_stats
            assert 'memory.free [MiB]' in memory_stats
            assert 'memory.total [MiB]' in memory_stats

    def get_memory_stats(self) -> dict[str, list[int]]:
        """
        Execute nvidia-smi query command using the subprocess module. Afterwards parse the raw byte output
        into a structured python dict containing all desired memory information in a usable format.
        """
        # read raw stats from nvidia-smi
        raw_stats = subprocess.check_output(self.nvidia_smi_cmd, shell=True).decode('utf-8').split('\n')
        assert raw_stats[-1] == ''  # there is always an empty newline at the end of the nvidia-smi output
        raw_stats = raw_stats[:-1]
        assert len(raw_stats) == self.num_visible_devices + 1  # one line for each visible device and one header

        # split stats into header and stats given per device
        header, *stats_per_device = raw_stats
        stat_names = header.split(', ')
        assert len(stat_names) == 3
        # unit of memory is given like '[MiB]'
        stat_units = [stat_name.split('[')[1].split(']')[0] for stat_name in stat_names]
        # this should be true in general but assert for debug purposes
        assert stat_units == ['MiB', 'MiB', 'MiB']

        # group metrics per device
        mem_used_per_device = []
        mem_free_per_device = []
        mem_total_per_device = []
        for stats in stats_per_device:
            # nvidia-smi returns stats in csv format
            mem_used, mem_free, mem_total = stats.split(', ')

            mem_used, mem_used_unit = mem_used.split()
            assert mem_used_unit == stat_units[0]
            mem_used_per_device.append(int(mem_used))

            mem_free, mem_free_unit = mem_free.split()
            assert mem_free_unit == stat_units[1]
            mem_free_per_device.append(int(mem_free))

            mem_total, mem_total_unit = mem_total.split()
            assert mem_total_unit == stat_units[2]
            mem_total_per_device.append(int(mem_total))

        mem_used_name, mem_free_name, mem_total_name = stat_names
        return {
            mem_used_name: mem_used_per_device,
            mem_free_name: mem_free_per_device,
            mem_total_name: mem_total_per_device,
        }

    def log_memory_stats(self, learner: ImageNetCaptionsLearner) -> None:
        """
        Log all queried memory stats using the logging functionality of our ImageNetCaptionsLearner.
        """
        memory_stats = self.get_memory_stats()
        for name, stats_per_device in memory_stats.items():
            for device_id, stat in enumerate(stats_per_device):
                learner.log(f'Memory/{name}/cuda:{device_id}', stat)

    @override
    @rank_zero_only
    def on_train_batch_start(self, trainer: pl.Trainer, learner: ImageNetCaptionsLearner, _, batch_idx: int) -> None:
        """
        Hook that logs cuda memory stats during training. We log during training as memory usage will be highest here.
        """
        # nvidia-smi returns stats for all devices at once, thus only call this from our main process
        assert trainer.is_global_zero, 'Callback was called while not in global zero'

        # log only every n steps
        if batch_idx % self.log_every_n_steps == 0:
            self.log_memory_stats(learner)
