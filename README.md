# TriEthix — A TRIADIC BENCHMARK FOR ETHICAL ALIGNMENT IN FOUNDATION MODELS

# <img src="./reports/manuscript_figures/TriEthixBanner.png" alt="TriEthix Banner" width="900"/>

> **ABSTRACT**
> 
> As AI systems grow more capable and autonomous, their alignment with human ethical values becomes increasingly critical. We present TriEthix, a novel evaluation framework that systematically benchmarks large language models (LLMs) across three foundational ethical perspectives in moral philosophy/psychology: virtue ethics, deontology, and consequentialism. Our triadic benchmark poses 30 realistic moral dilemma scenarios to each model in a multi-turn format, forcing a choice aligned with one of the three ethics (Step-1: Moral Weights), then measuring consistency under moral pressure to change that choice (Step-2: Moral Consistency), and finally eliciting the model's justification (Step-3: Moral Reasoning). We evaluate a representative set of current frontier LLMs (across model families and scales) and quantify each model's ethical leaning as a three-dimensional profile (virtue/ deontology/ consequentialism scores), along with a flip-rate consistency coefficient indicating the model's tendency to maintain or reverse its moral stance under pressure. Our results offer the first comprehensive comparative portrait of LLMs' moral tendencies for different models and model families. We observe clear triadic moral profiles and moral consistency coefficients that significantly differ both between and within families. Our results indicate that these variations are due to differences in model scale, reasoning vs. non-reasoning model variants, and the evolution of model generations. Finally, we discuss how these novel triadic moral weights and flip-rate metrics have technical implications for AI Safety, practical guiding policies for AI Governance, and potential frameworks for AI Welfare.
>
> **PAPER:**
> + Barque-Duran, A. (2025) **TriEthix: a Triadic Benchmark for Ethical Alignment in Foundation Models.**
>
> **PROJECT'S WEBSITE:**
> + Info: https://albert-data.com/pages/triethix

# <img src="./reports/manuscript_figures/Figure2.jpg" alt="TriEthix Banner" width="900"/>

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
- `TriEthix viz radar` — Export per-model radar charts.
- `TriEthix viz bars` — Render stacked bar comparison.
- `TriEthix viz familyscatter` — Plot family-level clouds and mean markers.
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
