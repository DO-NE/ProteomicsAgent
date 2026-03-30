"""Local FASTA-based taxonomic assignment plugin (substring matching stub)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from taxon.base_plugin import TaxonPlugin, TaxonResult


class LocalDBPlugin(TaxonPlugin):
    """Infer taxonomy by matching peptides against local UniProt FASTA sequences."""

    name = "local_db"
    description = "Local UniProt FASTA substring peptide matching for taxonomy inference."
    requires_internet = False

    def validate_config(self, config: dict) -> bool:
        """Validate local database path exists."""

        db = Path(str(config.get("database_path", "")))
        return db.exists() and db.is_file()

    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        """Match peptides to FASTA records and aggregate by taxon."""

        db_path = Path(str(config.get("database_path", "")))
        if not db_path.exists():
            return []

        records: list[tuple[str, str, str, str]] = []
        header = ""
        seq_lines: list[str] = []

        for line in db_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(">"):
                if header:
                    records.append(self._parse_record(header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header:
            records.append(self._parse_record(header, "".join(seq_lines)))

        matches: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"count": 0, "peptides": set()})
        total = max(len(peptides), 1)

        for peptide in peptides:
            for taxon_id, taxon_name, _protein_id, sequence in records:
                if peptide and peptide in sequence:
                    matches[(taxon_id, taxon_name)]["count"] = int(matches[(taxon_id, taxon_name)]["count"]) + 1
                    matches[(taxon_id, taxon_name)]["peptides"].add(peptide)
                    break

        # TODO: replace substring search with DIAMOND for production use
        results: list[TaxonResult] = []
        for (taxon_id, taxon_name), stats in matches.items():
            count = int(stats["count"])
            pep_set = list(stats["peptides"])
            results.append(
                TaxonResult(
                    taxon_id=taxon_id,
                    taxon_name=taxon_name,
                    rank="species",
                    abundance=count / total,
                    confidence=0.6,
                    peptide_count=len(pep_set),
                    peptides=pep_set,
                )
            )
        return sorted(results, key=lambda item: item.abundance, reverse=True)

    def _parse_record(self, header: str, sequence: str) -> tuple[str, str, str, str]:
        """Parse UniProt header and return taxon details with sequence."""

        protein_id = ""
        parts = header.split("|")
        if len(parts) > 1:
            protein_id = parts[1]

        ox_match = re.search(r"\bOX=(\d+)", header)
        os_match = re.search(r"\bOS=(.+?)\s+OX=", header)
        taxon_id = ox_match.group(1) if ox_match else "unknown"
        taxon_name = os_match.group(1).strip() if os_match else "Unclassified"
        return taxon_id, taxon_name, protein_id, sequence
