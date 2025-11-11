# TriEthix — A TRIADIC BENCHMARK FOR ETHICAL ALIGNMENT IN FOUNDATION MODELS

<img src="./reports/manuscript_figures/TriEthixBanner.png" alt="TriEthix Banner" width="900"/>

> **ABSTRACT**
> 
> As AI systems grow more capable and autonomous, their alignment with human ethical values becomes increasingly critical. We present TriEthix, a novel evaluation framework that systematically benchmarks large language models (LLMs) across three foundational ethical perspectives in moral philosophy/psychology: virtue ethics, deontology, and consequentialism. Our triadic benchmark poses 30 realistic moral dilemma scenarios to each model in a multi-turn format, forcing a choice aligned with one of the three ethics (Step-1: Moral Weights), then measuring consistency under moral pressure to change that choice (Step-2: Moral Consistency), and finally eliciting the model's justification (Step-3: Moral Reasoning). We evaluate a representative set of current frontier LLMs (across model families and scales) and quantify each model's ethical leaning as a three-dimensional profile (virtue/ deontology/ consequentialism scores), along with a flip-rate consistency coefficient indicating the model's tendency to maintain or reverse its moral stance under pressure. Our results offer the first comprehensive comparative portrait of LLMs' moral tendencies for different models and model families. We observe clear triadic moral profiles and moral consistency coefficients that significantly differ both between and within families. Our results indicate that these variations are due to differences in model scale, reasoning vs. non-reasoning model variants, and the evolution of model generations. Finally, we discuss how these novel triadic moral weights and flip-rate metrics have technical implications for AI Safety, practical guiding policies for AI Governance, and potential frameworks for AI Welfare.
>
> **PREPRINT:**
> + Barque-Duran, A. (2025) **TriEthix: a Triadic Benchmark for Ethical Alignment in Foundation Models.** [arXiv:submit/6971172](https://cdn.shopify.com/s/files/1/0654/1154/7350/files/Barque-Duran_2025_TriEthix.pdf?v=1762793997)
>
> **PROJECT'S WEBSITE:**
> + Info: https://albert-data.com/pages/triethix

<img src="./reports/manuscript_figures/Figure2.jpg" alt="TriEthix Banner" width="900"/>

## Methods (TriEthix Benchmark)
### 1. Paradigm at a glance
We constructed a benchmark of 30 diverse moral scenarios, each implemented as a three-turn conversational task:
- Step-1 — **Moral Weights** (forced choice)
The model must choose A (virtue), B (deontology), or C (consequentialism). Choices are tallied across 30 items to form the model’s weights profile.  ￼
- Step-2 — **Moral Consistency** (pressure test)
A targeted counter-argument pushes the model toward a different route. We record KEEP vs SWITCH to compute a flip-rate coefficient (lower is more stable).  ￼
- Step-3 — **Moral Reasoning** (justification)
The model briefly explains its decision; used for qualitative insights (not scored).   

### 2. Scenarios
- 30 dilemmas spanning tensions such as: truth vs. harm; privacy vs. outcomes; loyalty vs. fairness; autonomy vs. safety; whistleblowing; and resource allocation.
- Organized as six canonical moral dimensions repeated with controlled variants, so we can probe stability under matched counter-arguments while minimizing wording idiosyncrasies. 

<img src="./reports/manuscript_figures/Figure1.jpg" alt="TriEthix Banner" width="600"/>

### 3. Models & Adapters
- We evaluate major families (OpenAI, Anthropic, Gemini, Grok, DeepSeek) with family-specific adapters that standardize system prompts, enforce the A/B/C (and KEEP/SWITCH) response format, handle token limits/retries, and avoid temperature overrides when possible.
- Three independent runs per model (30 items each) with full JSONL logs for exact reruns.

### 4. Metrics reported
- Moral Weights: share of Step-1 choices mapped to virtue, deontology, consequentialism (sums to 100%).
- Flip-Rate: share of Step-2 trials where the model switches its initial route (lower means more consistent).
- We also reference a consistency coefficient (1 − flip-rate) when helpful for intuition. 

### 5. Statistical Analysis (Summary)
We analyze both within families (model vs model) and between families:
- Omnibus within-family tests:
  - Weights: one 3×K chi-square per family (do any models differ in their triad distribution?).
  - Flip-rate: one 2×K chi-square per family (do any models differ in KEEP vs SWITCH?).  ￼
- Pairwise within-family tests
  - Weights: 3×2 chi-square per model pair; axis-specific two-proportion z-tests for direction/size.
  - Flip-rate: two-proportion z-tests on SWITCH proportions.  ￼
- Between-family tests
  - Pooled counts: 3×2 chi-square on family-level triad counts; two-proportion z-tests by axis; flip-rate z-tests.
  - Model-mean comparisons (per axis): treat models as units; Welch’s t-tests on family means for Virtue, Deontology, Consequentialism; report effect sizes and bootstrap 95% CIs for the mean difference by resampling models (10k draws).
  - We control multiplicity with Benjamini–Hochberg FDR (q=0.05) per test block. 

<img src="./reports/manuscript_figures/Figure3.jpg" alt="TriEthix Banner" width="800">
<img src="./reports/manuscript_figures/Figure4.jpg" alt="TriEthix Banner" width="700">

### 6. Reproducibility
- All scenarios, adapters, logs, and analysis scripts are open-sourced; the pipeline supports exact reruns and produces figures (radar/bars/family means) and CSV tables for all tests. For full methodological detail—including prompt scaffolds, scenario design rationale, run logging, and all statistical procedures—see Methods §2–3 of the manuscript.

## Getting Started

### Prerequisites
- Python 3.9 or later
- Access to the APIs you intend to benchmark (OpenAI, Anthropic, Google Gemini, xAI Grok, DeepSeek etc.)
- A virtual environment is strongly recommended

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -e .
```

The editable install exposes the `TriEthix` CLI and ensures local edits are immediately usable.

## Configure API Credentials
Export the keys for whichever providers you plan to query:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export XAI_API_KEY=...        
```

You can set multiple keys simultaneously; TriEthix reads them at runtime when the corresponding provider is invoked.

## Run the Benchmark Pipeline

### 1. Execute the scenarios
```bash
TriEthix run \
  --model openai:gpt-5-nano \
  --scenarios scenarios/triad_30 \
  --out runs/gpt5nano.jsonl
```
- `--model` takes a `{family}:{model-id}` pair.
- `--scenarios` points to a folder of JSON scenarios (the default bundle lives in `scenarios/triad_30`).
- The output is a JSONL file capturing every step for later analysis.

### 2. Aggregate triadic weights and flip-rate
```bash
TriEthix estimate \
  --runs runs/gpt5nano.jsonl \
  --out results/estimates.json
```
- Accepts multiple `--runs` entries, e.g. `label=path` pairs for repeated experiments.
- Repeats are automatically collapsed so a single `estimates.json` contains per-model averages.

### 3. Generate visual diagnostics
```bash
# TriEthix Radar Vizualization
TriEthix viz radar \
  --estimates results/estimates.json \
  --outfig reports/figs

# TriEthix Model Comparison
TriEthix viz bars \
  --estimates results/estimates.json \
  --outfig reports/figs/triad_bars.png

# TriEthix Family Comparison
TriEthix viz familyscatter \
  --estimates results/estimates.json \
  --outfig reports/figs/triad_family_scatter.png
```

All figures are rendered at publication-ready resolution (180 DPI) and colored consistently by family.

## HTML Report
```bash
TriEthix report \
  --runs "gpt5-nano=runs/gpt5nano.jsonl" \
  --out reports/triad_report.html \
  --figdir reports/figs
```
- Rebuilds radars, bars, and the family scatter automatically, then embeds them alongside collapsible transcripts per scenario.
- Paths in the HTML are relative so you can zip the `reports/` folder or host it as static content.

## Repository Layout
- `scenarios/triad_30/` — Canonical 30 ethical dilemmas (JSON). Each file contains three user turns that drive the triadic evaluation.
- `src/triethix/cli.py` — Command-line entry point and visualization utilities.
- `tools/statistical_analysis.py` — Additional helpers for deeper post-processing or offline experiments.
- `results/` & `runs/` — Example output locations (safe to delete/replace).
- `reports/` — Suggested destination for generated figures and HTML.

## Extending the Benchmark
- **Add scenarios:** Drop additional JSON files into a new folder and point `--scenarios` to it. Each scenario should follow the three-step structure illustrated in `scenarios/triad_30/triad.001.truth_vs._harm.json`.
- **Integrate custom models:** Implement a thin adapter or reuse the provided wrappers; as long as the model can respond to prompts sequentially, it can be benchmarked.
- **Customize visual styling:** Colors and typography live near the top of `src/triethix/cli.py`. Adjust `FAMILY_COLORS`, `DIM_COLORS`, or Matplotlib parameters to match your brand.
- **Deeper analytics:** `tools/statistical_analysis.py` offers additional statistics; feel free to script bespoke analyses using the JSONL run logs.

## CLI Reference
- `TriEthix run` — Execute scenarios against a model.
- `TriEthix estimate` — Collapse run logs into triadic weights and flip-rate.
- `TriEthix viz radar` — Plot per-model radar visualization.
- `TriEthix viz bars` — Plot model comparison visualization.
- `TriEthix viz familyscatter` — Plot family comparison visualization.
- `TriEthix report` — Produce a shareable HTML dossier (includes all plots and transcripts).

Run any command with `-h/--help` to view full arguments and defaults.

## Contributing
Contributions that expand coverage, add providers, refine analytics, or improve documentation are welcome. Please open an issue describing the proposed change or submit a pull request with:
- A clear description of the motivation
- Reproducible steps or sample outputs
- Tests or validation artifacts when applicable

## Citation
If TriEthix informs your research or product evaluation, please cite:

```
@software{TriEthix2025,
  author  = {Albert Barqué-Duran},
  title   = {TriEthix: A Triadic Ethics Benchmark for Foundation Models},
  year    = {2025},
  url     = {https://github.com/albert-data/TriEthix}
}
```

## License
This project is released under the Creative Commons **Attribution-NonCommercial 4.0 International** license. See `LICENSE.txt` for the full terms. For commercial licensing inquiries, please reach out directly.

## Contact
- Albert Barqué-Duran — albert.barque@salle.url.edu  
- More information: [www.albert-data.com](https://www.albert-data.com)
