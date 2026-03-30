"""Taxon inference plugin interfaces and result datamodel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TaxonResult:
    """Taxonomic assignment record produced by a plugin."""

    taxon_id: str
    taxon_name: str
    rank: str
    abundance: float
    confidence: float
    peptide_count: int
    peptides: list[str] = field(default_factory=list)


class TaxonPlugin(ABC):
    """Abstract interface for taxon inference plugins."""

    name: str
    description: str
    requires_internet: bool

    @abstractmethod
    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        """Run taxon inference and return TaxonResult list."""

    @abstractmethod
    def validate_config(self, config: dict) -> bool:
        """Validate plugin-specific configuration."""
