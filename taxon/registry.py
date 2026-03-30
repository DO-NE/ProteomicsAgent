"""Dynamic registry for taxon inference plugins."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import requests
from rich.console import Console

from .base_plugin import TaxonPlugin, TaxonResult


class TaxonRegistry:
    """Discover, list, and execute taxon plugins from algorithms package."""

    def __init__(self) -> None:
        """Initialize registry and discover available plugins."""

        self.console = Console()
        self.plugins: dict[str, TaxonPlugin] = {}
        self._discover()

    def _discover(self) -> None:
        """Discover plugin subclasses in taxon/algorithms/*.py."""

        algorithms_dir = Path(__file__).resolve().parent / "algorithms"
        for file in algorithms_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
            module_name = f"taxon.algorithms.{file.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                self.console.print(f"[yellow]Warning: failed to import {module_name}: {exc}[/yellow]")
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls is TaxonPlugin or not issubclass(cls, TaxonPlugin):
                    continue
                plugin: TaxonPlugin = cls()
                self.plugins[plugin.name] = plugin

    def list_plugins(self) -> list[dict]:
        """Return plugin metadata list."""

        return [
            {
                "name": plugin.name,
                "description": plugin.description,
                "requires_internet": plugin.requires_internet,
            }
            for plugin in self.plugins.values()
        ]

    def run(self, plugin_name: str, peptides: list[str], config: dict) -> list[TaxonResult]:
        """Run selected plugin with optional internet connectivity precheck."""

        if plugin_name not in self.plugins:
            raise KeyError(f"Unknown taxon plugin: {plugin_name}")

        plugin = self.plugins[plugin_name]
        if not plugin.validate_config(config):
            raise ValueError(f"Invalid config for plugin '{plugin_name}'.")

        if plugin.requires_internet:
            try:
                requests.get("https://api.unipept.ugent.be", timeout=3)
            except Exception as exc:  # noqa: BLE001
                raise ConnectionError(
                    "Internet connectivity check failed for UniPept API. "
                    "Please verify network access before running this plugin."
                ) from exc

        return plugin.run(peptides, config)
