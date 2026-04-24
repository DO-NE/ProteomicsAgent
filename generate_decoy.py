"""Generate a target-decoy FASTA by appending reversed sequences."""

import sys
from pathlib import Path


def generate_decoy(target_fasta: str, output_fasta: str | None = None) -> None:
    """Read a target FASTA and write target + reversed decoy entries.

    Each decoy header is prefixed with ``DECOY_`` and the sequence is reversed.
    """
    target = Path(target_fasta)
    if not target.exists():
        print(f"File not found: {target_fasta}")
        sys.exit(1)

    if output_fasta is None:
        output_fasta = str(target.with_stem(target.stem + "_decoy"))

    headers: list[str] = []
    sequences: list[str] = []
    buf: list[str] = []

    with target.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    sequences.append("".join(buf))
                    buf = []
                headers.append(line.rstrip())
            else:
                stripped = line.strip()
                if stripped:
                    buf.append(stripped)
        if buf:
            sequences.append("".join(buf))

    with open(output_fasta, "w", encoding="utf-8") as out:
        # Write target entries
        for hdr, seq in zip(headers, sequences):
            out.write(hdr + "\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i : i + 60] + "\n")

        # Write reversed decoy entries
        for hdr, seq in zip(headers, sequences):
            accession = hdr[1:].split()[0]
            out.write(f">DECOY_{accession} Reversed decoy\n")
            rev = seq[::-1]
            for i in range(0, len(rev), 60):
                out.write(rev[i : i + 60] + "\n")

    print(f"Target proteins:  {len(headers)}")
    print(f"Decoy proteins:   {len(headers)}")
    print(f"Total:            {len(headers) * 2}")
    print(f"Output:           {output_fasta}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target.fasta> [output.fasta]")
        sys.exit(1)
    target = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    generate_decoy(target, output)
