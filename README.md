# Metaprot Agent

## 1) Overview
Metaprot Agent is a terminal-first, conversational metaproteomics analysis system. It combines a local LLM (served by `llama-cpp-python` in OpenAI-compatible mode) with an orchestrated proteomics pipeline: format conversion, peptide identification, validation, quantitation, protein assignment, taxon inference, and figure/report export.

## 2) Hardware requirements
- NVIDIA GPU with **8GB+ VRAM**.
- Tested on **NVIDIA TITAN X Pascal (12GB VRAM)**.
- Ubuntu 24.04 Linux server over SSH.

## 3) Installation
### a) Create Conda environment
```bash
conda env create -f environment.yaml
conda activate ProteomicsAgent
```

### b) Install llama-cpp-python with CUDA support
```bash
pip install llama-cpp-python[server] \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

### c) Download a GGUF model
```bash
pip install huggingface_hub

hf download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
  Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf --local-dir ~/models
```

## 4) Configuration (.env setup)
```bash
cp .env.example .env
```
Then edit `.env` and set your paths (`MODEL_PATH`, `TPP_BIN_PATH`, `MSFRAGGER_PATH`, etc.).

## 5) Starting the LLM server
```bash
python main.py start-server
```
This command launches:
```bash
python -m llama_cpp.server --model <MODEL_PATH> --n_gpu_layers -1 --port 8000
```

## 6) Bioinformatics tool installation
### MSFragger
- Project: https://msfragger.nesvilab.org/
- Download JAR and make a wrapper direcotry: mkdir -p {path_to_wrapper_dir}
- Make script: cat > {path_to_wrapper_dir}/msfragger <<'EOF'
#!/usr/bin/env bash
java -jar {path_to_downloaded_JAR} "$@"
EOF
- set `MSFRAGGER_PATH` in `.env`.

### Comet
- Project:
- Download exe and set `COMET_PATH` in `.env`.
- Activate the exe: chmod +x {path_to_exe}
- Generate default parameter file: {path_to_exe} -p

### Trans-Proteomic Pipeline (TPP)
- Project: https://tools.proteomecenter.org/wiki/index.php?title=Software:TPP
- Install binaries (`PeptideProphet`, `ASAPRatio`, `ProteinProphet`) and set `TPP_BIN_PATH`.

### Percolator
- Project: https://github.com/percolator/percolator
- Install executable and set `PERCOLATOR_PATH`.

## 7) Usage examples
### Start a run
```bash
python main.py run --input data/sample.mzML --db db/community.fasta --autonomy balanced
```

### Resume latest run
```bash
python main.py resume
```

### List available taxon algorithms
```bash
python main.py list-algorithms
```

### Sample terminal session
```text
$ python main.py run --input LFQ_Orbitrap_DDA_Condition_A.mzML --db community.fasta
Found existing run from 2026-03-20T14:12:04+00:00 (completed: ['format_conversion']). Resume? (y/n) n
> Start peptide identification with MSFragger
(assistant explains why)
<ACTION>
tool: run_pipeline_stage
stage: peptide_id
params: {"tool": "msfragger"}
</ACTION>
```

## 8) Adding a custom taxon inference algorithm
Implement `TaxonPlugin` from `taxon/base_plugin.py`, then place your `.py` file in `taxon/algorithms/`.
No registration edits are required because the registry auto-discovers plugins.

### Example custom plugin
```python
from __future__ import annotations

import random
from taxon.base_plugin import TaxonPlugin, TaxonResult

class RandomTaxonPlugin(TaxonPlugin):
    name = "random_demo"
    description = "Random demo taxon assignment plugin"
    requires_internet = False

    def validate_config(self, config: dict) -> bool:
        return True

    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        if not peptides:
            return []
        taxa = [
            ("123", "Bacterium alpha", "species"),
            ("456", "Bacterium beta", "genus"),
            ("789", "Bacterium gamma", "family"),
        ]
        counts = {k: 0 for k in range(len(taxa))}
        for _pep in peptides:
            counts[random.randint(0, len(taxa) - 1)] += 1
        total = len(peptides)
        out = []
        for idx, count in counts.items():
            tid, name, rank = taxa[idx]
            out.append(
                TaxonResult(
                    taxon_id=tid,
                    taxon_name=name,
                    rank=rank,
                    abundance=count / total,
                    confidence=0.4,
                    peptide_count=count,
                    peptides=[],
                )
            )
        return sorted(out, key=lambda r: r.abundance, reverse=True)
```

Drop your `.py` file in `taxon/algorithms/` — no other changes needed.

## 9) Dataset
Benchmark dataset: **Kleiner et al. metaproteomics benchmark**, PRIDE accession **PXD019910**. Download spectra and metadata via PRIDE FTP:
- https://www.ebi.ac.uk/pride/archive/projects/PXD019910

## 10) Troubleshooting
- **LLM server not starting**: verify `MODEL_PATH` and `llama-cpp-python[server]` installation.
- **CUDA out of memory**: reduce GPU layers, e.g. start server with smaller `--n_gpu_layers`.
- **nvcc not found**: install CUDA toolkit in conda env:
  ```bash
  conda install -c nvidia cuda-toolkit
  ```
- **MSFragger Java heap error**: increase JVM memory, e.g. run with `java -Xmx32g -jar ...`.
