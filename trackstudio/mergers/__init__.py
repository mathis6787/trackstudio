"""Cross-camera merger interfaces and registry."""

from __future__ import annotations

import logging

from .base import VisionMerger

logger = logging.getLogger(__name__)


class MergerRegistry:
    """Lazy registry for vision mergers."""

    def __init__(self) -> None:
        self._mergers: dict[str, type[VisionMerger] | None] = {"bev_cluster": None}

    def register(self, name: str, merger_class: type[VisionMerger]) -> None:
        if not issubclass(merger_class, VisionMerger):
            raise ValueError(f"{merger_class} must inherit from VisionMerger")
        self._mergers[name] = merger_class
        logger.info(f"Registered merger: {name}")

    def get(self, name: str) -> type[VisionMerger]:
        if name not in self._mergers:
            raise ValueError(f"Unknown merger: {name}. Available: {list(self._mergers.keys())}")

        merger_class = self._mergers[name]
        if merger_class is None and name == "bev_cluster":
            from .bev_cluster import BEVClusterMerger  # noqa: PLC0415

            merger_class = BEVClusterMerger
            self._mergers[name] = merger_class

        if merger_class is None:
            raise ValueError(f"Unknown merger: {name}")
        return merger_class

    def create(self, name: str, **kwargs) -> VisionMerger:
        return self.get(name)(**kwargs)

    def list_available(self) -> list[str]:
        return list(self._mergers.keys())


merger_registry = MergerRegistry()

__all__ = ["VisionMerger", "BEVClusterMerger", "merger_registry"]


def __getattr__(name: str):
    if name == "BEVClusterMerger":
        from .bev_cluster import BEVClusterMerger  # noqa: PLC0415

        return BEVClusterMerger
    raise AttributeError(name)
