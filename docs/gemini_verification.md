# Gemini MCP verification — step-by-step

This doc walks through verifying that the `creseq-mcp` server is reachable
from Google's Gemini CLI as an MCP client. Three demo prompts cover the
core post-QC analysis pipeline: (1) running upstream activity calling,
(2) splitting the classified output into GC-matched FASTAs via
`extract_sequences` (your authored tool with the new `gc_match` flag),
and (3) chaining `tool_motif_enrichment` off those FASTAs to recover
enriched TFs. After completing the steps, four screenshots go in
`docs/screenshots/`.

---

## 1. Install Gemini CLI

```bash
# Option A — via npm (recommended on macOS)
npm install -g @google/generative-ai-cli

# Option B — via Homebrew if a tap is available
brew install google/tap/gemini-cli
```

Verify the binary is on PATH:

```bash
gemini --version
```

If you do not yet have an API key, get a free one from
<https://aistudio.google.com/app/apikey> and export it:

```bash
export GEMINI_API_KEY="paste-key-here"
```

(Persist this in your shell rc file if you want it to stick.)

---

## 2. Register `creseq-mcp` with Gemini

Open (or create) `~/.gemini/settings.json` and add:

```json
{
  "mcpServers": {
    "creseq": {
      "command": "/Users/sarrahrose/Downloads/x/BioE-134-Final-Proj/.venv/bin/python",
      "args": ["-m", "creseq_mcp.server"],
      "cwd": "/Users/sarrahrose/Downloads/x/BioE-134-Final-Proj"
    }
  }
}
```

Two notes:

- The absolute python path points at the project's venv, which already has
  `mcp`, `pandas`, `pyjaspar`, `biopython`, `statsmodels`, etc. installed.
  Without this, Gemini will spawn the system python and fail to import.
- The `cwd` is what makes default paths like `~/.creseq/uploads/` resolve
  correctly inside the tools.

If `~/.gemini/settings.json` already exists, merge the `mcpServers` block
into the existing JSON instead of overwriting.

---

## 3. Confirm the server is registered

```bash
gemini  # opens the interactive client
```

At the Gemini prompt, type:

```
/mcp list
```

You should see a `creseq` entry and a tool count of **26**. If the count
is 0 or the entry says "error", check:

- `~/.gemini/settings.json` is valid JSON (run `python -m json.tool < ~/.gemini/settings.json`).
- The python path in `command` is the venv path, not `/usr/bin/python3`.
- Running `.venv/bin/python -m creseq_mcp.server` from a separate terminal
  starts cleanly and prints no errors before being killed with Ctrl+C.

**📸 Screenshot 1 — `01_mcp_list.png`**: capture the `/mcp list` output
showing `creseq` registered with 26 tools.

---

## 4. Demo prompt 1 — pipeline setup with default paths

This first prompt runs the upstream pipeline so the later prompts have
something classified to work with. `tool_activity_report` (a teammate's
orchestrator that wraps the normalize + classify steps) takes no arguments
when files are in the default upload directory — it is the cheapest
end-to-end check that tool dispatch is working at all.

Paste this into Gemini:

> Run `tool_activity_report` on the default upload directory and tell me
> the summary — how many oligos passed filtering, how many were called
> active, and what method was used?

**Expected behavior:** Gemini issues a no-argument tool call to
`tool_activity_report`, the server returns a summary dict like
`{n_oligos_after_filter: 600, n_active: 83, method: "threshold_log2gt1", ...}`,
and Gemini phrases it back in natural language. The
`activity_results.tsv` file written to `~/.creseq/uploads/` is what later
prompts will consume.

**📸 Screenshot 2 — `02_activity_report.png`**: capture the tool-call
block plus Gemini's reply naming `n_oligos_after_filter`, `n_active`,
and `method`.

---

## 5. Demo prompt 2 — your code, with the new GC-matching feature

This prompt exercises **`extract_sequences`** — your bridge function
between activity calling and motif enrichment — using the `gc_match=True`
parameter you just added. It demonstrates: (a) a tool you authored,
(b) Gemini correctly passing a non-default boolean parameter, and (c) the
new return field `gc_matched` showing up in the response.

> Use `extract_sequences` to split the classified results at
> `~/.creseq/uploads/activity_results.tsv` into active and background
> FASTA files, using the design manifest at
> `~/.creseq/uploads/design_manifest.tsv` for the sequence source. Set
> `gc_match=True` so the background GC distribution matches the active
> set. Save the active FASTA to `/tmp/active_gemini.fa` and the background
> to `/tmp/background_gemini.fa`. Tell me how many records ended up in
> each file and confirm `gc_matched` is true in the response.

**Expected behavior:** Gemini issues an `extract_sequences` tool call
with `gc_match=true` in the arguments. Server returns
`{active_fasta: "/tmp/active_gemini.fa", background_fasta: "/tmp/background_gemini.fa",
n_active: <N>, n_background: <M>, gc_matched: true}`.

**📸 Screenshot 3 — `03_extract_sequences_gc_match.png`**: capture the
tool-use block (showing `gc_match: true` in the arguments) and the reply
naming the two record counts and `gc_matched: true`.

---

## 6. Demo prompt 3 — chained multi-tool, your code

This chains `tool_motif_enrichment` off the FASTAs that step 2 produced,
exercising another of your tools and showing the agent can pass output
paths from one tool call into the next. Use `score_threshold=0.85` to
keep the run fast (a stricter threshold means fewer hits to compute).

> Now make me a volcano plot of the classified table at
> Now run `tool_motif_enrichment` on `/tmp/active_gemini.fa` and
> `/tmp/background_gemini.fa` with `score_threshold=0.85`. Write the
> output table to `/tmp/motif_enrichment_demo.tsv` and tell me the top
> three significantly enriched TFs from the summary string.

**Expected behavior:** Gemini issues a `tool_motif_enrichment` call with
the two FASTA paths and `score_threshold=0.85`. First-time JASPAR loads
take ~30 seconds; subsequent calls are fast. The server returns
`{enrichment_table: "/tmp/motif_enrichment_demo.tsv", summary: "N motifs enriched at FDR < 0.05. Top 10: ..."}`.

After the response comes back, verify the TSV exists:

```bash
ls -la /tmp/motif_enrichment_demo.tsv
head -5 /tmp/motif_enrichment_demo.tsv
```

**📸 Screenshot 4 — `04_motif_enrichment.png`**: capture the tool-use
block (showing both FASTA paths from step 3 + `score_threshold: 0.85`)
and Gemini's reply naming the top 3 TFs from the summary.

---

## 7. Where the screenshots go

```
docs/
  screenshots/
    01_mcp_list.png                    # /mcp list showing creseq + 26 tools
    02_activity_report.png             # tool_activity_report call + summary reply
    03_extract_sequences_gc_match.png  # extract_sequences with gc_match=true
    04_motif_enrichment.png            # tool_motif_enrichment chained off step 3
```

These four images are the rubric-required evidence that the MCP wrappers
"work with the Gemini MCP code." Screenshots 3 and 4 specifically exercise
your authored tools (`extract_sequences` with the new `gc_match`
parameter, and `tool_motif_enrichment`); reference all four in
`README_sarrah.md` under a "Verified with Gemini CLI" section.

---

## Fallback: Claude Desktop

If Gemini CLI installation is blocked (network, permissions, API quota),
Claude Desktop uses the identical MCP protocol and is acceptable
evidence. Add the same `mcpServers` block to
`~/Library/Application Support/Claude/claude_desktop_config.json`,
restart Claude Desktop, and run the same four prompts. Screenshot the
tool-use blocks the same way and label the README section "Verified
with Claude Desktop (MCP reference client)".

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/mcp list` shows 0 tools | Wrong python path in settings.json — must be `.venv/bin/python` absolute path |
| Tool call returns "module not found" | The venv is missing dependencies; run `uv sync` or `pip install -e .` from the project root |
| `tool_activity_report` errors on missing file | Generate the test data first: `.venv/bin/python scripts/generate_test_data.py` |
| `tool_motif_enrichment` is slow | First call downloads JASPAR motifs (~30 s); subsequent calls are fast. Use `score_threshold=0.85` or higher to keep runtimes down. |
| Plot file not created | Confirm matplotlib's Agg backend is active by checking the tool's stderr — should be silent |
| Gemini can't find a tool you named | The tool surface is what's listed in `mcp_manifest.json` at repo root — check spelling. Tools authored upstream may be named `tool_*`; tools with `name=` in the decorator may have a shorter name. |
