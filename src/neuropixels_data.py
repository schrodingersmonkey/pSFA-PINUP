"""Loading raw Neuropixels LFP + pupil data from data/new_data/data/functional_connectivity."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_tsv(path: Path) -> pd.DataFrame:
    """Read a functional_connectivity TSV, restricted to the time column and the
    last (area-average) column when more are present. These per-probe LFP files
    carry one column per recording channel plus the average -- only the average
    is ever used downstream, and the files run 100s of MB each, so reading every
    channel with pandas' comment-aware python engine (needed to skip the '#'
    metadata header) is a 10x-plus unnecessary slowdown. Peeking the header first
    to size `usecols` lets us use the fast C engine instead.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                ncols = len(line.rstrip("\n").split("\t"))
                break
    if ncols > 2:
        return pd.read_csv(path, sep="\t", comment="#", usecols=[0, ncols - 1])
    return pd.read_csv(path, sep="\t", comment="#")


def discover_vis_files(meta: dict, data_dir: Path) -> dict:
    """{area_key: filename} for every VIS-area LFP file present on disk for this subject.
    area_key is the bare structure name (e.g. "VISp"), suffixed "_2", "_3", ...
    when more than one probe recorded the same structure.
    """
    probe_structures = []
    for probe_id, structs in meta["probestructures"].items():
        vis = [s for s in structs if "VIS" in s]
        if vis:
            probe_structures.append((probe_id, vis[0]))
    counts = Counter(s for _, s in probe_structures)
    seen, lfp_files = {}, {}
    for probe_id, structure in probe_structures:
        fname = f"probe_{probe_id}_{structure}.tsv"
        if not (data_dir / fname).exists():
            continue
        if counts[structure] > 1:
            idx = seen.get(structure, 0) + 1
            seen[structure] = idx
            key = f"{structure}_{idx}"
        else:
            key = structure
        lfp_files[key] = fname
    return lfp_files


def load_subject_lfp(data_dir: Path, downsample_factor: int = 10,
                      area_filter: list | None = None, strict: bool = False):
    """Area-averaged LFP + pupil area for one subject, downsampled and time-aligned.

    area_filter=None: use every VIS-area file discovered for this subject (variable
        area count per subject).
    area_filter=<list>, strict=True: require every listed area present, else return
        None (use when feature width must match across subjects, e.g. LOSO pooling).
    area_filter=<list>, strict=False: use whichever areas in the list are present,
        minimum 2 areas, else return None (use when subjects may contribute partial
        area-pairs, e.g. the spatial map).

    Returns (X_ds, d_pupil, t_ds, areas) where X_ds is (T, len(areas)), areas is in
    area_filter order when area_filter is given, else discovery order.
    """
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    lfp_files = discover_vis_files(meta, data_dir)

    if area_filter is None:
        areas = list(lfp_files.keys())
        file_by_area = lfp_files
    else:
        base_files = {}
        for key, fname in lfp_files.items():
            base = key.split("_")[0]
            if base not in base_files:
                base_files[base] = fname
        if strict:
            if not all(a in base_files for a in area_filter):
                return None
            areas = list(area_filter)
        else:
            areas = [a for a in area_filter if a in base_files]
            if len(areas) < 2:
                return None
        file_by_area = {a: base_files[a] for a in areas}

    if not areas:
        return None

    area_times, area_avg = {}, {}
    for area in areas:
        df = load_tsv(data_dir / file_by_area[area])
        area_times[area] = df.iloc[:, 0].values
        area_avg[area] = df.iloc[:, -1].values

    t_start = max(v[0] for v in area_times.values())
    t_end = min(v[-1] for v in area_times.values())
    t_ref = area_times[areas[0]]
    mask = (t_ref >= t_start) & (t_ref <= t_end)
    t_lfp = t_ref[mask]
    T_ref = int(mask.sum())
    N = len(areas)

    X_all = np.zeros((T_ref, N))
    for col, area in enumerate(areas):
        X_all[:, col] = np.interp(t_lfp, area_times[area], area_avg[area])

    T_cut = (T_ref // downsample_factor) * downsample_factor
    X_ds = X_all[:T_cut].reshape(T_cut // downsample_factor, downsample_factor, N).mean(1)
    t_ds = t_lfp[:T_cut].reshape(T_cut // downsample_factor, downsample_factor).mean(1)

    pupil_df = load_tsv(data_dir / "pupilarea.tsv")
    pupil_df.columns = ["time", "pupil_area"]
    pupil_df = pupil_df.dropna()
    d_pupil = np.interp(t_ds, pupil_df["time"].values, pupil_df["pupil_area"].values)

    return X_ds, d_pupil, t_ds, areas


def list_subjects(data_root: Path, required_files=("metadata.json", "pupilarea.tsv"), exclude=()):
    subjects = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
        and all((p / f).exists() for f in required_files)
    )
    return [s for s in subjects if s not in exclude]
