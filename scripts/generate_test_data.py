"""
Generate synthetic CRE-seq test data and write all 4 QC-ready TSVs
to ~/.creseq/uploads/.

Design (CRE-seq spec — see creseq_mcp/schema.py):
- 600 oligos: 350 test_elements, 100 positive_controls, 150 negative_controls
- OLIGO_LEN = 170 bp (mid-range of 84–200 bp CRE-seq spec)
- BARCODE_LEN = 10 bp (mid-range of 8–12 bp CRE-seq spec)
- 50 variant families: reference carries a planted motif; 4 alts per family
  disrupt it with a 1 bp point mutation inside the motif.
- 20 barcodes per oligo  →  barcode_complexity PASS
- ~3% true barcode collisions (duplicated rows w/ different oligo_ids)
- ~70% perfect CIGAR/MD  →  synthesis_error_profile PASS
- Activity is sequence-driven (motif content), NOT label-driven.  Negative
  controls have no planted motifs, so they cluster around 0 by construction
  rather than by hardcoded label.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

UPLOAD_DIR = Path.home() / ".creseq" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLIGO_LEN = 170
BARCODE_LEN = 10
N_OLIGOS = 600
N_BARCODES_PER_OLIGO = 20

# JASPAR-style motif consensus sequences and per-motif log2 effect sizes.
# Effects are additive when multiple motifs co-occur.
MOTIFS = {
    "GATA1": "AGATAAGG",
    "AP1":   "TGAGTCA",
    "SP1":   "GGGCGGG",
}
MOTIF_EFFECTS = {"GATA1": 2.5, "AP1": 1.8, "SP1": 1.2}

# ── helpers ──────────────────────────────────────────────────────────────────


def rand_seq(n: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=n))


def mutate(seq: str, n_mut: int = 1) -> str:
    bases = list(seq)
    positions = rng.choice(len(seq), size=n_mut, replace=False)
    for p in positions:
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = str(rng.choice(alts))
    return "".join(bases)


def plant_motif(seq: str, motif: str, position: int) -> str:
    """Insert *motif* into *seq* starting at *position*, overwriting bases."""
    end = position + len(motif)
    if end > len(seq):
        raise ValueError("Motif does not fit within sequence at given position")
    return seq[:position] + motif + seq[end:]


def disrupt_motif_at(seq: str, motif_start: int, motif_len: int) -> str:
    """Flip a single base inside the motif window to break it."""
    flip_pos = motif_start + int(rng.integers(0, motif_len))
    bases = list(seq)
    alts = [b for b in "ACGT" if b != bases[flip_pos]]
    bases[flip_pos] = str(rng.choice(alts))
    return "".join(bases)


def find_motifs(seq: str) -> list[str]:
    """Return motif names whose consensus appears anywhere in *seq*."""
    return [name for name, consensus in MOTIFS.items() if consensus in seq]


def make_cigar_md(obs: str, ref: str) -> tuple[str, str]:
    """
    Build a SAM-spec-compliant CIGAR + MD pair for an observed-vs-reference
    alignment.  Soft-clips (when obs is shorter than ref) are emitted at the 3'
    end of the CIGAR; the MD tag covers only the aligned region.
    """
    aligned_len = min(len(obs), len(ref))
    clip = abs(len(obs) - len(ref))

    md_parts: list[str] = []
    run = 0
    last_was_match = True
    for o, r in zip(obs[:aligned_len], ref[:aligned_len]):
        if o == r:
            run += 1
            last_was_match = True
        else:
            # SAM spec: consecutive mismatches require a "0" separator
            if last_was_match:
                md_parts.append(str(run))
            else:
                md_parts.append("0")
            md_parts.append(r)
            run = 0
            last_was_match = False
    md_parts.append(str(run))

    md = "".join(md_parts)
    cigar = f"{aligned_len}M" + (f"{clip}S" if clip else "")
    return cigar, md


def negative_binomial_counts(mean: float, n: float = 5.0, size: int = 1) -> np.ndarray:
    """
    Sample non-negative integer counts with a target *mean*, parameterised so
    the dispersion (n) is constant across calls.  Falls back to Poisson when
    the requested mean is small enough that NB is overkill.
    """
    if mean <= 0:
        return np.zeros(size, dtype=int)
    p = n / (n + mean)
    return rng.negative_binomial(n, p, size=size)


# ── design manifest ──────────────────────────────────────────────────────────

print("Building design manifest…")
oligos: list[dict] = []
oligo_motifs: dict[str, list[tuple[str, int]]] = {}  # oligo_id -> [(motif_name, start_pos), ...]

# 50 variant families: 1 reference (motif-planted) + 4 alts (motif-disrupted)
motif_names = list(MOTIFS.keys())
for fam_i in range(50):
    fam_id = f"FAM{fam_i:03d}"

    # Reference: random sequence with one planted motif at a random position
    base_seq = rand_seq(OLIGO_LEN)
    motif_name = motif_names[fam_i % len(motif_names)]
    motif_seq = MOTIFS[motif_name]
    motif_start = int(rng.integers(20, OLIGO_LEN - len(motif_seq) - 20))
    ref_seq = plant_motif(base_seq, motif_seq, motif_start)

    ref_id = f"{fam_id}_ref"
    oligos.append({
        "oligo_id": ref_id,
        "sequence": ref_seq,
        "designed_category": "test_element",
        "variant_family": fam_id,
        "is_reference": True,
    })
    oligo_motifs[ref_id] = [(motif_name, motif_start)]

    # 4 alts: disrupt the planted motif with a single base flip inside the motif
    for mut_i in range(4):
        alt_seq = disrupt_motif_at(ref_seq, motif_start, len(motif_seq))
        alt_id = f"{fam_id}_mut{mut_i}"
        oligos.append({
            "oligo_id": alt_id,
            "sequence": alt_seq,
            "designed_category": "test_element",
            "variant_family": fam_id,
            "is_reference": False,
        })
        # Disrupted motif is no longer detected by find_motifs; record empty.
        oligo_motifs[alt_id] = []

# Remaining test elements (no family).  Each gets 0–2 random motifs planted
# so test elements span a realistic activity distribution.
for i in range(350 - 250):
    seq = rand_seq(OLIGO_LEN)
    n_planted = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
    planted: list[tuple[str, int]] = []
    used_positions: list[int] = []
    for _ in range(n_planted):
        m = motif_names[int(rng.integers(0, len(motif_names)))]
        ms = MOTIFS[m]
        for _attempt in range(10):
            pos = int(rng.integers(20, OLIGO_LEN - len(ms) - 20))
            if all(abs(pos - p) > len(ms) + 2 for p in used_positions):
                break
        seq = plant_motif(seq, ms, pos)
        planted.append((m, pos))
        used_positions.append(pos)
    oid = f"TEST{i:04d}"
    oligos.append({
        "oligo_id": oid,
        "sequence": seq,
        "designed_category": "test_element",
        "variant_family": None,
        "is_reference": False,
    })
    oligo_motifs[oid] = planted

# Positive controls: 2 strong motifs each → high activity by construction
for i in range(100):
    seq = rand_seq(OLIGO_LEN)
    chosen = list(rng.choice(motif_names, size=2, replace=False))
    planted = []
    used_positions = []
    for m in chosen:
        ms = MOTIFS[m]
        for _attempt in range(10):
            pos = int(rng.integers(20, OLIGO_LEN - len(ms) - 20))
            if all(abs(pos - p) > len(ms) + 2 for p in used_positions):
                break
        seq = plant_motif(seq, ms, pos)
        planted.append((m, pos))
        used_positions.append(pos)
    oid = f"POSCTRL{i:03d}"
    oligos.append({
        "oligo_id": oid,
        "sequence": seq,
        "designed_category": "positive_control",
        "variant_family": None,
        "is_reference": False,
    })
    oligo_motifs[oid] = planted

# Negative controls: random sequence, no planted motifs.  (Uniform random ACGT
# — no GC-content tweak; the *point* is the absence of regulatory motifs, not
# any particular GC composition.)
for i in range(150):
    seq = rand_seq(OLIGO_LEN)
    oid = f"NEGCTRL{i:03d}"
    oligos.append({
        "oligo_id": oid,
        "sequence": seq,
        "designed_category": "negative_control",
        "variant_family": None,
        "is_reference": False,
    })
    # Detect any *accidental* motif occurrences from the random sequence so the
    # activity model stays consistent.
    oligo_motifs[oid] = [(m, seq.find(MOTIFS[m])) for m in motif_names if MOTIFS[m] in seq]

# Serialize is_reference as lowercase strings so round-tripping through CSV is
# unambiguous (the previous Python-bool default produced the string "True",
# which equality-compares to anything).
manifest_df = pd.DataFrame(oligos)
manifest_df["is_reference"] = manifest_df["is_reference"].map(
    {True: "true", False: "false"}
)
manifest_df.to_csv(UPLOAD_DIR / "design_manifest.tsv", sep="\t", index=False)
print(f"  {len(manifest_df)} oligos written")

# ── barcode pool ─────────────────────────────────────────────────────────────

print("Generating barcodes…")
target_pool = N_OLIGOS * N_BARCODES_PER_OLIGO + 500
all_barcodes: set[str] = set()
while len(all_barcodes) < target_pool:
    all_barcodes.add(rand_seq(BARCODE_LEN))
all_barcodes = list(all_barcodes)

oligo_ids = manifest_df["oligo_id"].tolist()
oligo_to_barcodes: dict[str, list[str]] = {}
bc_pool = iter(all_barcodes)

for oid in oligo_ids:
    bcs = [next(bc_pool) for _ in range(N_BARCODES_PER_OLIGO)]
    oligo_to_barcodes[oid] = bcs

# ── mapping table ────────────────────────────────────────────────────────────

print("Building mapping table…")
seq_lookup = dict(zip(manifest_df["oligo_id"], manifest_df["sequence"]))
mapping_rows: list[dict] = []

for oid in oligo_ids:
    ref_seq = seq_lookup[oid]
    for bc in oligo_to_barcodes[oid]:
        n_reads = max(1, int(rng.poisson(50)))

        r = rng.random()
        if r < 0.70:
            obs = ref_seq
        elif r < 0.90:
            obs = mutate(ref_seq, n_mut=int(rng.integers(1, 4)))
        else:
            clip = int(rng.integers(5, 20))
            obs = ref_seq[:-clip]  # truncated → soft clip

        cigar, md = make_cigar_md(obs, ref_seq)
        mapping_rows.append({
            "barcode": bc,
            "oligo_id": oid,
            "cigar": cigar,
            "md": md,
            "n_reads": n_reads,
        })

# Inject ~3% true barcode collisions: duplicate the row with a different
# oligo_id so mapping_df contains two records that share the same barcode but
# point to different oligos.  This is what the upstream barcode→oligo
# assignment step would actually emit when a barcode is ambiguous.
n_collisions = int(len(mapping_rows) * 0.03)
collision_indices = rng.choice(len(mapping_rows), size=n_collisions, replace=False)
for idx in collision_indices:
    src = mapping_rows[idx]
    other_oid = oligo_ids[int(rng.integers(0, len(oligo_ids)))]
    while other_oid == src["oligo_id"]:
        other_oid = oligo_ids[int(rng.integers(0, len(oligo_ids)))]
    other_ref = seq_lookup[other_oid]

    # Fresh observation against the *other* oligo so the duplicate row is
    # internally consistent (CIGAR/MD vs. the oligo_id it claims to map to).
    rr = rng.random()
    if rr < 0.70:
        obs2 = other_ref
    elif rr < 0.90:
        obs2 = mutate(other_ref, n_mut=int(rng.integers(1, 4)))
    else:
        clip = int(rng.integers(5, 20))
        obs2 = other_ref[:-clip]
    cigar2, md2 = make_cigar_md(obs2, other_ref)

    mapping_rows.append({
        "barcode": src["barcode"],
        "oligo_id": other_oid,
        "cigar": cigar2,
        "md": md2,
        "n_reads": max(1, int(rng.poisson(50))),
    })

mapping_df = pd.DataFrame(mapping_rows)
mapping_df.to_csv(UPLOAD_DIR / "mapping_table.tsv", sep="\t", index=False)
print(f"  {len(mapping_df)} barcode rows written ({n_collisions} collisions injected)")

# ── plasmid counts ───────────────────────────────────────────────────────────

print("Building plasmid counts…")
plasmid_df = mapping_df[["barcode", "oligo_id"]].copy()
# DNA counts are NB-distributed around a high target mean (~1000) to match
# typical CRE-seq plasmid prep depth.
plasmid_df["dna_count"] = negative_binomial_counts(mean=1000.0, size=len(plasmid_df))
# Zero out 5% of barcodes to simulate low-depth dropout.
zero_mask = rng.random(len(plasmid_df)) < 0.05
plasmid_df.loc[zero_mask, "dna_count"] = 0
plasmid_df.to_csv(UPLOAD_DIR / "plasmid_counts.tsv", sep="\t", index=False)
print(f"  {len(plasmid_df)} rows written")

# ── RNA counts ───────────────────────────────────────────────────────────────

print("Building RNA counts…")


def oligo_activity(oid: str) -> float:
    """log2 activity driven by motif content + small Gaussian noise."""
    motifs_here = [m for (m, _pos) in oligo_motifs.get(oid, [])]
    motif_bonus = sum(MOTIF_EFFECTS[m] for m in motifs_here)
    base = float(rng.normal(0.0, 0.3))
    return base + motif_bonus


# Per-oligo activity (shared across barcodes within the same oligo, plus a
# small per-barcode jitter so replicate noise is realistic).
per_oligo_activity = {oid: oligo_activity(oid) for oid in oligo_ids}

rna_rows = plasmid_df[["barcode", "oligo_id"]].copy()
for rep in ["rep1", "rep2"]:
    rna_counts = []
    for _, row in plasmid_df.iterrows():
        dna = int(row["dna_count"])
        bc_activity = per_oligo_activity[row["oligo_id"]] + float(rng.normal(0, 0.4))
        # Expected RNA scales with DNA (more plasmid → more transcript) and
        # 2**activity (definition of log2(RNA/DNA)).
        expected = max(0.0, (dna + 0.5) * (2.0 ** bc_activity))
        rna_counts.append(int(negative_binomial_counts(mean=expected, size=1)[0]))
    rna_rows[f"rna_count_{rep}"] = rna_counts

rna_rows.to_csv(UPLOAD_DIR / "rna_counts.tsv", sep="\t", index=False)
print(f"  {len(rna_rows)} rows, 2 replicates written")

# ── run activity analysis ────────────────────────────────────────────────────

print("Running activity analysis…")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from creseq_mcp.activity.normalize import activity_report

_, summary = activity_report(
    UPLOAD_DIR / "plasmid_counts.tsv",
    UPLOAD_DIR / "rna_counts.tsv",
    UPLOAD_DIR / "design_manifest.tsv",
    upload_dir=UPLOAD_DIR,
)
print(f"  Active: {summary['n_active']} / {summary['n_oligos_after_filter']} ({summary['activity_rate']:.1%})")

print(f"\nAll files written to {UPLOAD_DIR}")
print("  mapping_table.tsv")
print("  plasmid_counts.tsv")
print("  design_manifest.tsv")
print("  rna_counts.tsv")
print("  activity_results.tsv")
print("\nGo to Chat and ask the agent to run QC.")
