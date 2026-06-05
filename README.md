# NeuroDecode SSVEP

This project analyzes SSVEP EEG decoding as a neural data pipeline: measurement, preprocessing, representation, modeling, inference, and action.

I started from a private OpenBCI/OpenVEP pilot recording, then used the public Wang2016 / Tsinghua SSVEP benchmark for the reproducible model comparison. The private recordings helped define the pipeline and question, but they are not included in this public repository.

## Project Links

- GitHub repository: https://github.com/NeuroBitEngineer/neurodecode-ssvep
- GitHub Pages site: https://neurobitengineer.github.io/neurodecode-ssvep/
- Final report HTML: `notebooks/COGS138_Final_Project_Submission.html`
- Detailed public analysis HTML: `notebooks/COGS138_Public_SSVEP_Expansion_MOABB.html`

## Research Question

Can SSVEP EEG signals be transformed into frequency-based representations that reliably decode intended visual targets, and does performance improve when the model is matched to SSVEP physiology?

## Main Claim

Simple target-frequency power detects some structure, but SSVEP-specific models such as CCA and TRCA decode targets more accurately because they model frequency-locked neural responses in visual cortex.

## Repository Structure

```text
index.html
requirements.txt
notebooks/
	COGS138_Final_Project_Submission.ipynb
	COGS138_Final_Project_Submission.html
	COGS138_Public_SSVEP_Expansion_MOABB.ipynb
	COGS138_Public_SSVEP_Expansion_MOABB.html
presentation/
	2-minute-presentation.pdf
results/
	wang2016_expanded_model_comparison.csv
scripts/
	run_wang2016_expansion.py
data/
	README.md
```

## Data

The public analysis uses Wang2016 through MOABB. The script downloads/loads the data into a local cache and writes the compact model-comparison table in `results/`.

The private OpenBCI/OpenVEP recordings are not pushed here because they are personal EEG recordings. The public benchmark is used for the reproducible results.

## Reproduce The Public Analysis

Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_wang2016_expansion.py
```

Export the final report HTML:

```bash
cd notebooks
../.venv/bin/python -m nbconvert --to html COGS138_Final_Project_Submission.ipynb
```

Open the report locally:

```bash
open COGS138_Final_Project_Submission.html
```
