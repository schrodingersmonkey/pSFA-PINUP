# pSFA: Inferring Non-Stationarity in Complex Systems

This repository contains the analysis and figure-generation code for the pSFA/PINUP project.

The overarching aim of the project is to infer latent sources of **non-stationarity directly from multivariate observations of complex dynamical systems**. Rather than assuming that the parameters governing a system remain fixed over time, we ask whether slowly varying changes in the system can be recovered from the evolving statistical structure of the observed data.

We first study this problem in controlled **time-varying vector autoregressive (VAR) systems**, where the latent source of non-stationarity is known and recovery can be quantified directly. These simulations are used to characterise when different representations of the observed system — including node- and edge-level statistics — contain sufficient information to recover the underlying time-varying parameter.

We then test the approach on neural recordings from the **Allen Brain Observatory Neuropixels Visual Coding dataset**. Here, the goal is to infer spontaneous arousal-linked non-stationarity from multivariate visual-cortical LFP activity alone, and subsequently compare the inferred latent trajectory with pupil diameter as an independent physiological proxy for arousal.

This repository contains the code used to generate the main VAR and Neuropixels figures, supplementary analyses, and associated result tables.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks

All three notebooks live at the repo root, next to `src/`, and can be opened and run directly
with Jupyter started from this directory:

- **`Figure_2_VAR.ipynb`** — main-text VAR figure (panels A, C/D, E).
- **`VAR_supplement.ipynb`** — supplementary VAR figures S1–S5.
- **`Figure_3_Neuropixels.ipynb`** — NeuroPixels application figure. Requires the raw
  NeuroPixels dataset (~39GB, not bundled). Point at your local copy before running:

  ```bash
  export NEUROPIXELS_DATA_ROOT=/path/to/your/neuropixels/functional_connectivity
  ```

Each notebook's own figures and tables are written under `outputs/`:

```
outputs/
├── figures/{var,neuropixels}/
└── results/{var,neuropixels}/
```

## Layout

```
src/            shared pipeline code (simulation, feature construction, SFA fitting, sweeps)
outputs/        figures and result tables written by the notebooks
```
