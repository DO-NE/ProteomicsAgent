# ProteomicsAgent

A terminal-first, conversational metaproteomics analysis system. ProteomicsAgent pairs a large language model (local or cloud-hosted) with an orchestrated bioinformatics pipeline — format conversion, peptide identification, PSM validation, quantitation, protein assignment, and taxon inference — driven by natural-language instructions from the researcher.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Bioinformatics tool setup](#5-bioinformatics-tool-setup)
6. [Starting the LLM server](#6-starting-the-llm-server)
7. [Usage](#7-usage)
8. [Pipeline stages](#8-pipeline-stages)
9. [Taxon inference plugins](#9-taxon-inference-plugins)
10. [Adding a custom taxon inference algorithm](#10-adding-a-custom-taxon-inference-algorithm)
11. [Running tests](#11-running-tests)
12. [Benchmark dataset](#12-benchmark-dataset)
13. [Output structure](#13-output-structure)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Architecture overview

```
ProteomicsAgent/
├── main.py                   # CLI entry point (click-based)
├── run_direct.py             # Lightweight direct pipeline (no LLM, no TPP)
├── config.py                 # Settings dataclass + tool-path checks
├── agent/
│   ├── orchestrator.py       # Conversational loop, action dispatch
│   ├── llm_client.py         # llama-cpp / Anthropic API client
│   ├── prompts.py            # System prompt template
│   └── state_manager.py      # Run-state serialisation (run_state.json)
├── pipeline/
│   ├── format_conversion.py  # RAW → mzML (msconvert)
│   ├── peptide_id.py         # PSM search (MSFragger or Comet)
│   ├── validation.py         # FDR control (PeptideProphet or Percolator)
│   ├── quantitation.py       # Spectral counting / ASAPRatio
│   └── protein_assignment.py # Protein grouping (ProteinProphet)
├── taxon/
│   ├── base_plugin.py        # TaxonPlugin ABC + TaxonResult dataclass
│   ├── registry.py           # Auto-discovery of plugins in algorithms/
│   └── algorithms/
│       ├── local_db.py       # FASTA substring matching
│       ├── unipept_api.py    # UniPept pept2lca REST API
│       ├── abundance_em.py   # Probabilistic EM abundance estimator (plugin)
│       └── abundance_em_core/
│           ├── model.py      # Multinomial mixture EM algorithm
│           ├── mapping_matrix.py  # FASTA → peptide-taxon matrix
│           ├── identifiability.py # Rank/collinearity diagnostics
│           ├── synthetic.py  # Synthetic community generator
│           └── tests/
│               └── test_model.py
└── visualization/
    ├── figures.py            # Bar chart, pie chart, heatmap, score dist
    └── report.py             # TSV + plain-text summary export
```

The system has three operating modes:

| Mode | Entry point | LLM | Use case |
|---|---|---|---|
| **Interactive agent** | `python main.py run` | required | Conversational, LLM proposes and explains each action |
| **No-LLM sequential** | `python main.py run --no-llm` | not required | Runs all pipeline stages automatically; prompts between stages |
| **Direct pipeline** | `python run_direct.py` | not required | Minimal: Comet + taxon inference only, no TPP |

---

## 2. Requirements

### Python

Python 3.11 via Conda is strongly recommended. The environment file pins all dependencies.

### LLM backend (choose one)

| Option | What you need |
|---|---|
| **Local llama-cpp** (default) | CUDA-capable GPU with ≥ 8 GB VRAM; any GGUF-format model |
| **No-LLM mode** | Neither — pipeline runs without any LLM |

### Bioinformatics tools (choose one from each category)

| Category | Tool | Required for |
|---|---|---|
| Format conversion | `msconvert` (ProteoWizard) | Vendor RAW → mzML |
| Peptide ID | MSFragger or Comet | Spectral library search |
| Validation | PeptideProphet (TPP) or Percolator | PSM FDR control |
| Quantitation | Spectral counting (built-in) or ASAPRatio (TPP) | Abundance estimates |
| Protein grouping | ProteinProphet (TPP) | Protein-level inference |

If you already have mzML files (e.g. downloaded from PRIDE), msconvert is not needed.

---

## 3. Installation

### a) Clone the repository

```bash
git clone https://github.com/DO-NE/ProteomicsAgent.git
cd ProteomicsAgent
```

### b) Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate ProteomicsAgent
```

This installs Python 3.11, NumPy, SciPy, pandas, matplotlib, seaborn, requests, rich, click, and other dependencies listed in `environment.yaml`.

> **New dependencies added for the `abundance_em` plugin:** `scipy >= 1.11` and
> (optionally) `pyteomics >= 4.7`. Both are declared in `environment.yaml`. If
> you skip `pyteomics`, a self-contained trypsin fallback is used instead.

### c) Install the LLM server (skip if using Claude API or no-LLM mode)

```bash
# Install llama-cpp-python with CUDA support
pip install "llama-cpp-python[server]" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

Replace `cu122` with your CUDA version (e.g. `cu118`, `cu124`). For CPU-only inference, just run `pip install "llama-cpp-python[server]"`.

### d) Download a GGUF model (local LLM only)

Any instruction-tuned GGUF model works. A good default for a mid-range GPU:

```bash
# Requires: pip install huggingface_hub (already in environment.yaml)
huggingface-cli download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
  Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --local-dir ~/models
```

Set `MODEL_PATH` in `.env` to the downloaded `.gguf` path (see below).

---

## 4. Configuration

Copy the example environment file and fill in your paths:

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
# ── LLM backend ──────────────────────────────────────────────────────────────
# "llama"  → local llama-cpp-python server (default)
# "claude" → Anthropic API (no GPU needed)
LLM_BACKEND=llama

# URL for the llama-cpp OpenAI-compatible server
LLAMA_SERVER_URL=http://localhost:8000/v1

# Anthropic API key — required only when LLM_BACKEND=claude
ANTHROPIC_API_KEY=

# ── Bioinformatics tool paths ─────────────────────────────────────────────────
MSFRAGGER_PATH=/path/to/msfragger        # executable wrapper or JAR path
COMET_PATH=/path/to/comet                # Comet executable
TPP_BIN_PATH=/path/to/tpp/bin            # directory containing TPP binaries
PERCOLATOR_PATH=/path/to/percolator      # Percolator executable

# ── Runtime settings ──────────────────────────────────────────────────────────
OUTPUT_DIR=./output                      # base directory for all run artifacts
DEFAULT_AUTONOMY_MODE=balanced           # full | balanced | supervised
DATABASE_PATH=/path/to/database.fasta    # default protein FASTA database

# ── Local LLM ─────────────────────────────────────────────────────────────────
MODEL_PATH=/path/to/model.gguf           # GGUF model for llama_cpp.server

# ── No-LLM bypass ────────────────────────────────────────────────────────────
NO_LLM_MODE=false                        # set true to skip LLM entirely
```

Verify your tool paths with:

```bash
python main.py check-tools
```

This prints a table showing which tools are found and which are missing.

---

## 5. Bioinformatics tool setup

### MSFragger

1. Download the JAR from [msfragger.nesvilab.org](https://msfragger.nesvilab.org/).
2. Create a wrapper script so `MSFRAGGER_PATH` points to an executable:

```bash
mkdir -p ~/tools/msfragger
cat > ~/tools/msfragger/msfragger <<'EOF'
#!/usr/bin/env bash
exec java -Xmx32g -jar /absolute/path/to/MSFragger-4.x.jar "$@"
EOF
chmod +x ~/tools/msfragger/msfragger
```

3. Set `MSFRAGGER_PATH=~/tools/msfragger/msfragger` in `.env`.

### Comet

1. Download the appropriate binary from [uwpr.github.io/Comet](https://uwpr.github.io/Comet/).
2. Make it executable:

```bash
chmod +x /path/to/comet.linux.exe
```

3. Set `COMET_PATH=/path/to/comet.linux.exe` in `.env`.
4. Optionally generate a default parameter file for manual tuning:

```bash
/path/to/comet.linux.exe -p
```

### Trans-Proteomic Pipeline (TPP)

Provides `PeptideProphet`, `ProteinProphet`, and `ASAPRatio`.

1. Follow the installation instructions at [tools.proteomecenter.org/wiki](https://tools.proteomecenter.org/wiki/index.php?title=Software:TPP).
2. Set `TPP_BIN_PATH` to the directory containing the TPP binaries (e.g. `/usr/local/tpp/bin`).

TPP 7.x renamed several binaries. The agent looks for both legacy and current names automatically.

### Percolator

1. Download from [github.com/percolator/percolator](https://github.com/percolator/percolator) or install via `conda install -c bioconda percolator`.
2. Set `PERCOLATOR_PATH=/path/to/percolator` in `.env`.

### msconvert (ProteoWizard)

Required only for converting vendor RAW files to mzML. Skip this step if your data is already in mzML format.

1. Download from [proteowizard.sourceforge.io](https://proteowizard.sourceforge.io/).
2. Add `msconvert` to your `PATH` or specify its full path.

---

## 6. Starting the LLM server

### Local llama-cpp server

```bash
python main.py start-server
```

This launches `llama_cpp.server` on `http://127.0.0.1:8000/v1` using the model specified in `MODEL_PATH`. Adjust GPU layer count via the `--n_gpu_layers` parameter if needed.

### Anthropic Claude API

Set `LLM_BACKEND=claude` and `ANTHROPIC_API_KEY=<your key>` in `.env`. No server needs to be started.

### No-LLM mode

Set `NO_LLM_MODE=true` in `.env`, or pass `--no-llm` on the command line. The orchestrator runs all pipeline stages sequentially without any LLM, prompting you between stages.

---

## 7. Usage

### Interactive agent run (with LLM)

```bash
python main.py run \
  --input data/sample.mzML \
  --db    data/community.fasta
```

The agent starts a conversation loop. Type natural-language requests and the LLM proposes, explains, and executes pipeline actions:

```
Metaproteomics Agent
Run ID: a3f91b…
Autonomy: balanced
Input files: data/sample.mzML
LLM backend: llama
Available taxon algorithms: local_db, unipept_api, abundance_em

> Run peptide identification with MSFragger

I'll run MSFragger against community.fasta. This produces a pepXML file that
feeds into PeptideProphet for validation.
<ACTION>
tool: run_pipeline_stage
stage: peptide_id
params: {"tool": "msfragger"}
</ACTION>

╭─ Action Result ─────────────────────────────────────────────────────────╮
│ { "status": "ok", "output": "output/a3f91b/mzml/sample.pepXML",        │
│   "stage": "peptide_id" }                                               │
╰─────────────────────────────────────────────────────────────────────────╯
```

Autonomy modes control how much human approval is required:

| Mode | Behaviour |
|---|---|
| `full` | All proposed actions execute immediately |
| `balanced` | Pipeline stage actions require approval; informational tools do not |
| `supervised` | Every action requires explicit approval; params can be edited inline |

Override the default from `.env` per-run:

```bash
python main.py run --input sample.mzML --db community.fasta --autonomy supervised
```

### Non-interactive run (no-LLM)

Runs all pipeline stages in sequence, prompting before each stage transition:

```bash
python main.py run \
  --input data/sample.mzML \
  --db    data/community.fasta \
  --no-llm
```

Or set `NO_LLM_MODE=true` once in `.env` and omit the flag.

### Non-interactive, fully automated run

Runs the full pipeline without any prompts:

```bash
python main.py run-pipeline \
  --input data/sample.mzML \
  --db    data/community.fasta
```

### Resume the most recent run

Run state is checkpointed to `output/<run_id>/run_state.json` after every stage. Resume from the last successful stage:

```bash
python main.py resume
```

The `run` command also detects a previous run automatically and offers to resume.

### List available taxon algorithms

```bash
python main.py list-algorithms
```

```
         Available Taxon Algorithms
┌──────────────┬──────────────────────────────────────┬──────────┐
│ Name         │ Description                           │ Internet │
├──────────────┼──────────────────────────────────────┼──────────┤
│ abundance_em │ Estimates relative taxon abundance …  │ no       │
│ local_db     │ Local UniProt FASTA substring …       │ no       │
│ unipept_api  │ Query UniPept pept2lca endpoint …     │ yes      │
└──────────────┴──────────────────────────────────────┴──────────┘
```

### Check configured tool paths

```bash
python main.py check-tools
```

---

### Direct pipeline (no LLM, no TPP)

`run_direct.py` is a minimal, dependency-light script that runs
**Comet → xcorr filter → taxon inference → TSV**. Useful for quick exploratory
runs when TPP is not installed.

```bash
python run_direct.py \
  --input  data/sample.mzML \
  --db     data/community.fasta \
  --comet  /path/to/comet \
  --algorithm local_db          # or: unipept_api
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `./output_direct` | Output directory |
| `--algorithm` | `local_db` | Taxon inference plugin (`local_db` or `unipept_api`) |
| `--xcorr` | `2.0` | Comet xcorr score cutoff (lower if too few peptides pass) |

Results are printed to stdout (top 10 taxa) and written to `<output-dir>/<run_id>/taxon/results.tsv`.

---

## 8. Pipeline stages

Each stage is implemented as a `PipelineStage` subclass in `pipeline/`. Stages are executed in order; each stage's output path is automatically forwarded as the next stage's input.

| # | Stage | Tool options | Input | Output |
|---|---|---|---|---|
| 1 | `format_conversion` | msconvert (auto-detected) or copy | `.raw` / `.mzML` | `.mzML` in `mzml/` |
| 2 | `peptide_id` | MSFragger (default) or Comet | `.mzML` | `.pepXML` |
| 3 | `validation` | PeptideProphet (default) or Percolator | `.pepXML` | annotated `.pepXML` or PSM `.txt` |
| 4 | `quantitation` | Spectral counting (default) or ASAPRatio | validated `.pepXML` | `spectral_counts.tsv` |
| 5 | `protein_assignment` | ProteinProphet | validated `.pepXML` | `proteins.prot.xml` |

The default search engine is MSFragger; switch to Comet by passing `{"tool": "comet"}` as action params in the interactive session, or set the `tool` key in `--no-llm` mode params.

---

## 9. Taxon inference plugins

Taxon plugins are auto-discovered from `taxon/algorithms/*.py` at runtime — no registration edits are ever needed.

### `local_db` — FASTA substring matching

Performs substring search of each identified peptide against all protein sequences in the FASTA database. Assigns each peptide to the first organism whose sequence contains it.

```python
config = {"database_path": "/path/to/community.fasta"}
```

Suitable for quick exploration; limited accuracy when the same peptide occurs in multiple organisms.

### `unipept_api` — UniPept pept2lca

Queries the [UniPept](https://unipept.ugent.be/) REST API (`/api/v2/pept2lca`) to compute the lowest-common-ancestor (LCA) taxon for each peptide. Requires an internet connection.

```python
config = {}   # no local paths needed
```

### `abundance_em` — Probabilistic multinomial-mixture EM

A quantitative model that estimates **relative taxon abundances** as a continuous vector on the probability simplex, rather than just assigning each peptide to one taxon.

#### How it works

Given:
- **A** — binary peptide-to-taxon mapping matrix built by in-silico tryptic digestion of the FASTA database
- **y** — observed spectral-count vector (one count per distinct peptide)

The model assumes each taxon *t* emits its member peptides with equal probability: `M[p,t] = A[p,t] / n_t`. The marginal probability of observing peptide *p* is the mixture `φ_p = Σ_t π_t · M[p,t]`, and `y ~ Multinomial(N, φ(π))`.

Inference is MAP Expectation-Maximization with a Dirichlet(α) prior on **π** (α < 1 induces sparsity; default α = 0.5). The algorithm:

1. **E-step** — compute responsibility `r[p,t] = (π_t · M[p,t]) / φ_p`
2. **M-step** — update `π_t ∝ Σ_p y_p · r[p,t] + (α − 1)`
3. Iterate until `‖π_new − π‖₁ < tol` or `max_iter` is reached

Approximate standard errors are computed from the observed Fisher information matrix and converted to per-taxon confidence scores.

An identifiability diagnostic (`identifiability_report`) checks for rank deficiency, near-collinear taxa (cosine similarity > 0.95), and taxa with no unique peptides — all logged as warnings before the fit.

```python
config = {
    "fasta_path": "/path/to/community.fasta",   # required
    "spectral_counts": {"PEPTIDESEQ": 42, ...}, # optional; defaults to count=1
    "alpha": 0.5,           # Dirichlet prior (< 1 = sparse)
    "max_iter": 500,
    "tol": 1e-6,
    "n_restarts": 1,        # increase for better global optima
    "min_abundance": 1e-4,  # taxa below this threshold are zeroed
    "enzyme": "trypsin",
    "missed_cleavages": 2,
    "seed": 42,
}
```

The core algorithm (`taxon/algorithms/abundance_em_core/`) is fully independent of the ProteomicsAgent runtime and can be imported and used standalone:

```python
from taxon.algorithms.abundance_em_core.model import AbundanceEM
from taxon.algorithms.abundance_em_core.synthetic import generate_synthetic_community

data = generate_synthetic_community(n_taxa=5, shared_fraction=0.15, total_psms=10000)
model = AbundanceEM(alpha=0.5, max_iter=500)
model.fit(data["A"], data["y"])
print(model.pi_)          # estimated abundance vector
print(model.standard_errors_)
```

---

## 10. Adding a custom taxon inference algorithm

1. Create a new Python file anywhere in `taxon/algorithms/`.
2. Implement the `TaxonPlugin` abstract base class from `taxon/base_plugin.py`.
3. Drop the file in `taxon/algorithms/` — that is all. The registry discovers it automatically on the next run.

```python
# taxon/algorithms/my_plugin.py
from __future__ import annotations
import random
from taxon.base_plugin import TaxonPlugin, TaxonResult


class MyTaxonPlugin(TaxonPlugin):
    name = "my_plugin"
    description = "Example custom taxon inference plugin."
    requires_internet = False

    def validate_config(self, config: dict) -> bool:
        # Return False if required config keys are missing.
        return True

    def run(self, peptides: list[str], config: dict) -> list[TaxonResult]:
        """
        Parameters
        ----------
        peptides : list of str
            Identified peptide sequences from the PSM results.
        config : dict
            Arbitrary key-value configuration (paths, thresholds, etc.).

        Returns
        -------
        list of TaxonResult, sorted by abundance descending.
        """
        taxa = [
            ("1423", "Bacteroides thetaiotaomicron", "species"),
            ("562",  "Escherichia coli",             "species"),
            ("1351", "Enterococcus faecalis",         "species"),
        ]
        counts = {i: 0 for i in range(len(taxa))}
        for _ in peptides:
            counts[random.randint(0, len(taxa) - 1)] += 1
        total = max(len(peptides), 1)

        results = []
        for idx, count in counts.items():
            tid, name, rank = taxa[idx]
            results.append(
                TaxonResult(
                    taxon_id=tid,
                    taxon_name=name,
                    rank=rank,
                    abundance=count / total,
                    confidence=0.5,
                    peptide_count=count,
                    peptides=[],
                )
            )
        return sorted(results, key=lambda r: r.abundance, reverse=True)
```

### `TaxonResult` fields

| Field | Type | Description |
|---|---|---|
| `taxon_id` | `str` | NCBI TaxID or other identifier |
| `taxon_name` | `str` | Organism name |
| `rank` | `str` | Taxonomic rank (`"species"`, `"genus"`, `"family"`, …) |
| `abundance` | `float` | Relative abundance in `[0, 1]` (proportion, not percent) |
| `confidence` | `float` | Confidence score in `[0, 1]` |
| `peptide_count` | `int` | Number of peptides assigned to this taxon |
| `peptides` | `list[str]` | Peptide sequences assigned to this taxon |

### Verify discovery

```bash
python main.py list-algorithms
# my_plugin should now appear in the table
```

---

## 11. Running tests

The `abundance_em_core` module ships with a unit-test suite covering 9 test cases (perfect recovery, sparse community detection, EM monotonicity, standard errors, identifiability diagnostics, Dirichlet prior effect, edge cases, and reproducibility).

```bash
python -m pytest taxon/algorithms/abundance_em_core/tests/ -v
```

Expected output:

```
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_perfect_recovery_no_sharing
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_recovery_with_sharing
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_sparse_community
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_em_monotonicity
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_standard_errors
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_identifiability_report
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_dirichlet_prior_effect
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_single_taxon
PASSED taxon/algorithms/abundance_em_core/tests/test_model.py::test_reproducibility_with_seed
9 passed in 0.1s
```

---

## 12. Benchmark dataset

The reference dataset used for development and validation is the **Kleiner et al. metaproteomics benchmark**:

- **PRIDE accession:** [PXD019910](https://www.ebi.ac.uk/pride/archive/projects/PXD019910)
- **Acquisition:** Label-free quantification (LFQ), Orbitrap DDA, tryptic digest
- **Description:** A defined synthetic microbial community with known composition, designed specifically to benchmark metaproteomics workflows

Download spectra and metadata via the PRIDE FTP or `PRIDE Downloader`:

```bash
# Using wget (replace file names with actual PRIDE FTP paths)
wget -r -np ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD019910/
```

The FASTA protein database bundled with the dataset is suitable for both MSFragger/Comet searches and the `abundance_em` plugin.

---

## 13. Output structure

Each run creates a self-contained directory under `OUTPUT_DIR`:

```
output/
└── <run_id>/
    ├── run_state.json         # Checkpoint: completed stages, output paths
    ├── mzml/
    │   └── sample.mzML
    ├── params/
    │   └── fragger.params     # Auto-generated search parameter file
    ├── validation/
    │   └── sample.pepXML      # PeptideProphet-annotated identifications
    ├── quant/
    │   └── spectral_counts.tsv
    ├── protein/
    │   └── proteins.prot.xml
    ├── taxon/
    │   └── results.tsv        # taxon_id, taxon_name, rank, abundance_pct, confidence, peptide_count
    ├── figures/
    │   ├── taxon_bar_chart.png/.pdf
    │   ├── taxon_pie_chart.png/.pdf
    │   ├── peptide_heatmap.png/.pdf
    │   └── score_distribution.png/.pdf
    └── report/
        └── summary.txt
```

The TSV output format:

```
taxon_id    taxon_name                        rank     abundance_pct  confidence  peptide_count
1423        Bacteroides thetaiotaomicron      species  32.1400        0.9200      412
562         Escherichia coli                  species  18.7200        0.8800      278
```

---

## 14. Troubleshooting

### LLM server not starting
- Verify `MODEL_PATH` is set and the file exists.
- Confirm `llama-cpp-python[server]` is installed: `python -c "from llama_cpp.server import app"`.
- If on a CPU-only machine, omit the `--extra-index-url` from the install command.

### CUDA out of memory
- Reduce the number of GPU layers. Edit `start-server` in `main.py` or set `--n_gpu_layers 20` (or any value below the model's total layer count).
- Switch to a smaller or more aggressively quantised model (Q2_K instead of Q4_K_M).

### CUDA toolkit not found
```bash
conda install -c nvidia cuda-toolkit
```

### MSFragger Java heap error
Increase the JVM heap in your wrapper script:
```bash
exec java -Xmx32g -jar /path/to/MSFragger.jar "$@"
```

### Too few peptides passing the xcorr filter (run_direct.py)
Lower the threshold:
```bash
python run_direct.py --input sample.mzML --db db.fasta --comet /path/to/comet --xcorr 1.5
```

### `abundance_em` warnings about identifiability
Warnings like *"N taxon(a) have no unique peptides"* or *"near-collinear taxa"* mean that those taxa's abundances cannot be independently resolved from the data — they share all their peptides with other taxa. This is a property of the database, not a bug. The model will still produce estimates, but their uncertainty (standard errors / confidence) will be high. Consider collapsing those taxa to genus level.

### Stage fails mid-run
Run state is checkpointed after every stage. Fix the underlying issue (wrong path, missing tool), then resume:
```bash
python main.py resume
```
The completed stages are skipped automatically.
