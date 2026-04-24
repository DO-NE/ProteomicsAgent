"""Quick POC: count unique taxa (species) in a FASTA file."""

import re
import sys
from collections import Counter
from pathlib import Path


def extract_species(header: str) -> str | None:
    # OS= field (UniProt style)
    m = re.search(r"\bOS=(.+?)(?:\s[A-Z]{2}=|$)", header)
    if m:
        return m.group(1).strip()
    # Quoted species= field (GeneMark / ATCC style)
    m = re.search(r'\bspecies="([^"]+)"', header)
    if m:
        return m.group(1).strip()
    # Bracket-delimited fallback
    m = re.search(r"\[([^\]]+)\]", header)
    if m:
        return m.group(1).strip()
    return None


def main(fasta_path: str) -> None:
    path = Path(fasta_path)
    if not path.exists():
        print(f"File not found: {fasta_path}")
        sys.exit(1)

    species_counter: Counter[str] = Counter()
    total_proteins = 0
    no_species = 0
    decoy_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            total_proteins += 1

            # Skip decoys
            first_token = header.split(None, 1)[0] if header else ""
            if first_token.upper().startswith("DECOY"):
                decoy_count += 1
                continue

            species = extract_species(header)
            if species:
                species_counter[species] += 1
            else:
                no_species += 1

    print(f"Total protein entries:  {total_proteins}")
    print(f"  Decoy entries:       {decoy_count}")
    print(f"  No species parsed:   {no_species}")
    print(f"  With species:        {sum(species_counter.values())}")
    print(f"\nUnique taxa (species): {len(species_counter)}")
    print(f"\n{'Rank':<6} {'Count':<8} Species")
    print("-" * 60)
    for i, (species, count) in enumerate(species_counter.most_common(), 1):
        print(f"{i:<6} {count:<8} {species}")


if __name__ == "__main__":
    fasta = sys.argv[1] if len(sys.argv) > 1 else "/home/scjlau/own_copy/PASAKP_decoy.fasta"
    main(fasta)
