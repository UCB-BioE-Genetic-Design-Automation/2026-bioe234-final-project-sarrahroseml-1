# CRE-seq / lentiMPRA Analysis Pipeline
**BioE 134 Final Project**

End-to-end lentiMPRA analysis toolkit with a Streamlit frontend, Claude API QC agent, and a FastMCP server. Built around the Agarwal et al. 2025 lentiMPRA dataset (ENCODE ENCSR463IRX, HepG2 pilot).

---

## Architecture

```
creseq_mcp/
├── processing/
│   ├── association.py   # mappy (minimap2) alignment + STARCODE barcode clustering
│   ├── pipeline.py      # simpler barcode-match pipeline (episomal / CRE-seq mode)
│   └── counting.py      # DNA and RNA barcode counting from FASTQs
├── qc/
│   ├── library.py       # 8-check library QC suite (barcode complexity, collision,
│   │                    #   uniformity, synthesis errors, oligo recovery, GC bias,
│   │                    #   plasmid depth, variant family coverage)
│   ├── activity.py      # normalization, log2(RNA/DNA), activity calling (z-test + BH FDR),
│   │                    #   variant delta scores
│   └── motifs.py        # JASPAR PWM fetching + oligo motif annotation
├── stats/
│   └── library.py       # motif enrichment summary, ENCODE/JASPAR/PubMed search,
│                        #   RAG context preparation, literature interpretation
├── data/
│   └── papers/
│       └── agarwal2025_lentimpra.txt   # full text of Agarwal et al. 2025 (Nature)
├── schema.py            # Pydantic input models for all tools
└── server.py            # FastMCP server — exposes all tools + paper resource
                         #   MCP resource URI: paper://agarwal2025-lentimpra

frontend/
├── app.py               # Streamlit UI (Upload, Chat, QC & Plots, Results pages)
└── agent.py             # ClaudeQCAgent — Anthropic API tool-use loop dispatching
                         #   to creseq_mcp functions
```

---

## Pipeline Steps

### 1. Association (barcode → oligo mapping)
- **Input:** R1 FASTQ (oligo reads) + barcode index FASTQ (15bp i5) + design FASTA
- **Tool:** `run_association()` in `association.py`
- STARCODE sphere-clustering at edit-distance 1 (error correction)
- mappy/minimap2 alignment of R1 reads to design FASTA (`preset="sr"`)
- Filter: ≥3 reads/barcode AND ≥50% mapping to the same oligo
- **Output:** `mapping_table.tsv`, `plasmid_counts.tsv`, `design_manifest.tsv` → `~/Desktop/creseq_outputs/`
- The UI has a **Skip association** toggle to reuse existing output files

### 2. Counting
- **DNA counting:** `process_dna_counting()` — counts barcode occurrences in plasmid DNA FASTQ, joins to mapping table → `plasmid_counts.tsv`
- **RNA counting:** `process_rna_counting()` — counts barcodes across N replicate FASTQs → `rna_counts.tsv` with columns `rna_count_rep1`, `rna_count_rep2`, ...
- DNA and RNA counting run in parallel via `ThreadPoolExecutor`

### 3. Activity Analysis
- **Tool:** `activity_report()` in `activity.py`
- RPM-style size-factor normalization per sample
- Per-barcode log2(RNA/DNA), averaged across replicates
- Collapse to per-oligo median (min 2 barcodes)
- Activity calling: z-test vs. negative control distribution → BH FDR at 5%
- Falls back to log2_ratio > 1 threshold when no negative controls are present
- **Output:** `activity_results.tsv`

### 4. Variant Delta Scores (optional)
- **Tool:** `compute_variant_delta_scores()` in `activity.py`
- Requires `variant_family` + `is_reference` columns in design manifest
- Computes Δlog2 = mutant − reference for each variant family
- z-test + BH FDR across all deltas
- **Output:** `variant_delta_scores.tsv`
- **Note:** Not applicable to the ENCODE HepG2 pilot (no variant families in that library)

### 5. Motif Analysis (optional, via Chat agent)
- **Tool:** `annotate_top_motifs()` in `motifs.py`
- Fetches PWMs from JASPAR online for a default HepG2-relevant TF list
- Scans oligo sequences, adds `top_motif` column to `activity_results.tsv`
- Then `motif_enrichment_summary()` computes active-vs-inactive enrichment ratios
- Requires internet access and an existing `activity_results.tsv`

---

## Data

The project uses the **Agarwal et al. 2025** ENCODE HepG2 pilot lentiMPRA dataset:
- **ENCODE accession:** ENCSR463IRX
- **Paper:** Agarwal, Inoue et al., *Nature* 639, 411–422 (2025). DOI: 10.1038/s41586-024-08430-9
- Full paper text stored as MCP resource `paper://agarwal2025-lentimpra`

Key files used:
| Role | ENCODE file | Notes |
|---|---|---|
| Association R1 (oligo reads) | ENCFF414OGO | ~200bp reads aligned to design FASTA |
| Association barcode index | ENCFF610IIQ | 15bp i5 barcode reads |
| DNA counting | ENCFF062NLF | 15bp barcode reads from plasmid pool |
| RNA rep1 | ENCFF388KGK | 15bp barcode reads (R1, not R2) |
| RNA rep2 | ENCFF184MJL | 15bp barcode reads |
| Design FASTA | reference.fa | gzip-compressed without .gz extension — handled by magic-byte detection |

Association results (saved to `~/Desktop/creseq_outputs/`):
- 25.5M/25.6M reads aligned (99.5%)
- 599K barcodes → 9,364 oligos after filtering

---

## AI Integration

### Claude QC Agent (`frontend/agent.py`)
- Uses Anthropic API (`claude-sonnet-4-6`) with tool-use loop
- Dispatches natural-language requests to `creseq_mcp` functions
- Accessible in the **Chat** tab; requires `ANTHROPIC_API_KEY` env variable
- Falls back to a stub agent when no API key is set

### MCP Server (`creseq_mcp/server.py`)
- FastMCP server exposing all QC + analysis tools
- Run with: `python -m creseq_mcp.server` or `mcp run creseq_mcp/server.py`
- Paper resource: `paper://agarwal2025-lentimpra` returns full Agarwal et al. 2025 text

---

## Output Files (`~/Desktop/creseq_outputs/`)

| File | Description |
|---|---|
| `mapping_table.tsv` | barcode, oligo_id, n_reads, cigar, md |
| `plasmid_counts.tsv` | barcode, oligo_id, dna_count |
| `design_manifest.tsv` | oligo_id, sequence, designed_category |
| `rna_counts.tsv` | barcode, oligo_id, rna_count_rep1, rna_count_rep2, ... |
| `activity_results.tsv` | per-oligo log2_ratio, pval, fdr, active |
| `variant_delta_scores.tsv` | per-variant delta_log2, pval, fdr, significant |

---

## Known Weaknesses / Technical Debt

### 1. Placeholder CIGAR/MD tags
The association step sets `cigar = "{barcode_len}M"` and `md = "{barcode_len}"` for every barcode — these are not real per-read alignments. As a result, the **Synthesis Error Profile** QC check always reports ~100% perfect synthesis, which is not meaningful. Real per-read CIGAR would require a BAM-level aligner (BWA/STAR) post-association, which is out of scope.

### 2. UPLOAD_DIR mismatch between server and frontend
- `frontend/app.py` writes to `~/Desktop/creseq_outputs/`
- `creseq_mcp/server.py` defaults to `~/.creseq/uploads/`

If the MCP server is run standalone (not through the Streamlit UI), it looks in a different directory and won't find the pipeline outputs. The `UPLOAD_DIR` in `server.py` line 52 should be updated to match.

### 3. Results page shows demo data
The **Results** tab (`📋 Results`) renders `st.session_state.data`, which is initialized from `get_demo_data()` in `mock_data.py`. It does not read from `activity_results.tsv`. A real implementation would load the activity results file into session state after the pipeline completes.

### 4. `oligo_length_qc` badge is always grey
`oligo_length_qc` appears in the UI's `tool_labels` dict (QC & Plots → Library QC tab) but was removed from the `manifest_tools` list in `library_summary_report`. It will always show the ⚪ "no data" badge. Either remove it from `tool_labels` or add it back to `manifest_tools` with the same silent-skip pattern used for `variant_family_coverage`.

### 5. Motif analysis requires internet
`annotate_top_motifs()` fetches PWMs from the JASPAR REST API at runtime. If JASPAR is unreachable, motif annotation silently returns empty results. There is no local PWM fallback.

### 6. Synthesis error profile uses mapping_table CIGAR, not per-read
Relatedly: even if real CIGARs were available, the current data model aggregates reads per barcode before writing `mapping_table.tsv`. Per-oligo synthesis error rates would require storing per-read alignment details, not just per-barcode summaries.

---

## Running the App

```bash
# Install dependencies
pip install -e ".[dev]"

# Set API key (optional — enables real Claude agent)
export ANTHROPIC_API_KEY=sk-ant-...

# Run Streamlit frontend
cd frontend
streamlit run app.py

# Run MCP server (separately, for Claude Code / MCP clients)
python -m creseq_mcp.server
```

---

## Dependencies

Key packages: `mcp>=1.0.0`, `pandas`, `numpy`, `scipy`, `pydantic`, `mappy`, `streamlit`, `plotly`, `anthropic`, `pyjaspar`, `biopython`

Optional (not pip-installable): `starcode` (conda: `conda install -c bioconda starcode`)
