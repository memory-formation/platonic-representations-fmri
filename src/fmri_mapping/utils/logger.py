"""
This code is sourced and adapted from the repository:
    https://github.com/rjha18/vec2vec

Please cite the original work as:

Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025).
*Harnessing the universal geometry of embeddings*.
arXiv preprint arXiv:2505.12540. https://arxiv.org/abs/2505.12540
"""

import logging
from typing import Dict, Iterable, Union, Optional, Literal

import torch
import wandb
from dmf.env import getenv

ModeType = Literal["disabled", "online", "offline"]


class WandbLogger:
    """Wandb logger that tracks running averages of metrics."""

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        log_frequency: int = 250,
        mode: ModeType = "online",
        **kws,
    ):
        self.vals = TensorRunningAverages()
        self.enabled = mode in ["online", "offline"]
        self.log_frequency = log_frequency
        self.log_step = 0

        if self.enabled:
            entity = entity or getenv("WANDB_ENTITY")
            project = project or getenv("WANDB_PROJECT")
            self.wandb_run = wandb.init(
                name=name,
                mode=mode,
                config=config,
                entity=entity,
                project=project,
                **kws,
            )
        else:
            self.wandb_run = wandb.init(mode="disabled")

    def logkv(
        self, k: str, v: Union[int, float, torch.Tensor]
    ) -> Union[int, float, torch.Tensor]:
        """Log a single key-value pair to W&B."""
        val = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
        self.vals.update(k, val)
        return v

    def logkvs(self, kvs: Dict[str, Union[int, float, torch.Tensor]]) -> None:
        """Log multiple key-value pairs to W&B."""
        for k, v in kvs.items():
            val = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
            self.vals.update(k, val)

    def dumpkvs(self, force: bool = False) -> None:
        """Log all averaged metrics to W&B every `log_frequency` steps."""
        self.log_step += 1
        if self.log_step % self.log_frequency == 0 or force:
            metrics = self.vals.get_and_clear_all()
            if self.enabled:
                self.wandb_run.log(metrics)
            else:
                logging.info(f"Metrics: {metrics}")

    def finish(self) -> None:
        """Finish the W&B run."""
        self.dumpkvs(force=True)
        if self.enabled:
            self.wandb_run.finish()


class TensorRunningAverages:
    """Tracks running averages of scalar tensors by key."""

    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        """Initialize empty storage for sums and counts."""
        self._store_sum = {}
        self._store_total = {}

    def __iter__(self) -> Iterable[str]:
        """Iterate over all metric keys."""
        return iter(self._store_sum.keys())

    def update(self, key: str, val: Union[int, float, torch.Tensor]) -> None:
        """Add a new value for the given metric key."""
        if key not in self._store_sum:
            self.clear(key)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)  # tensor -> num
        val = val.cpu()
        self._store_sum[key] += val
        self._store_total[key] += 1

    def get(self, key: str) -> float:
        """Return the current average value for a metric key."""
        total = max(self._store_total.get(key), torch.tensor(1.0))
        return (self._store_sum[key] / float(total.item())) or 0.0

    def clear(self, key: str) -> None:
        """Reset sum and count for a specific metric key."""
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float32)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)

    def clear_all(self) -> None:
        """Reset all stored metrics."""
        for key in self._store_sum:
            self.clear(key)

    def get_and_clear_all(self):
        """Return averaged metrics that have at least one update, then reset them."""
        metrics = {}
        for key in list(self._store_sum.keys()):
            count = self._store_total[key].item()
            if count > 0:
                metrics[key] = (self._store_sum[key] / count).item()
            self.clear(key)
        return metrics
