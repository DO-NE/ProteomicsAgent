"""UniPept API taxonomic LCA plugin."""

from __future__ import annotations

from collections import defaultdict

import requests

from taxon.base_plugin import TaxonPlugin, TaxonResult


class UnipeptAPIPlugin(TaxonPlugin):
    """Infer taxonomy via UniPept pept2lca API."""

    name = "unipept_api"
    description = "Query UniPept pept2lca endpoint for taxonomic LCA assignments."
    requires_internet = True

    def validate_config(self, config: dict) -> bool:
        """Validate UniPept plugin config."""

        return True

    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        """Run batched UniPept requests and aggregate abundance/confidence."""

        endpoint = "https://api.unipept.ugent.be/api/v2/pept2lca.json"
        batch_size = 100
        grouped: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
            lambda: {"count": 0, "peptides": []}
        )

        total_peptides = len(peptides)
        if total_peptides == 0:
            return []

        for start in range(0, total_peptides, batch_size):
            batch = peptides[start : start + batch_size]
            payload = {"input": batch, "extra": True, "names": True}
            response_json: list[dict] | None = None

            for _attempt in range(2):
                try:
                    response = requests.post(endpoint, json=payload, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    response_json = data if isinstance(data, list) else data.get("peptides", [])
                    break
                except Exception:
                    response_json = None

            if not response_json:
                continue

            for item in response_json:
                sequence = str(item.get("sequence", ""))
                taxon_id = str(item.get("taxon_id", "unknown"))
                taxon_name = str(item.get("taxon_name", "Unclassified"))
                taxon_rank = str(item.get("taxon_rank", "other"))
                key = (taxon_id, taxon_name, taxon_rank)
                grouped[key]["count"] = int(grouped[key]["count"]) + 1
                if sequence:
                    grouped[key]["peptides"].append(sequence)

        if not grouped:
            return []

        rank_conf = {"species": 1.0, "genus": 0.7, "family": 0.5}
        results: list[TaxonResult] = []
        for (taxon_id, taxon_name, rank), stats in grouped.items():
            count = int(stats["count"])
            results.append(
                TaxonResult(
                    taxon_id=taxon_id,
                    taxon_name=taxon_name,
                    rank=rank,
                    abundance=count / total_peptides,
                    confidence=rank_conf.get(rank, 0.3),
                    peptide_count=count,
                    peptides=list(stats["peptides"]),
                )
            )

        return sorted(results, key=lambda x: x.abundance, reverse=True)
