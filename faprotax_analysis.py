#!/usr/bin/env python3
"""
FAPROTAX Functional Analysis
============================

This script visualises and explores FAPROTAX functional predictions for
16S rRNA amplicon data.  It produces:

* **Heatmap** of the *n* most abundant functions across aerosol samples
  ordered by an environmental variable (pressure, wind speed, …).
* **Spearman correlations** between each function’s relative abundance
  and the chosen variable, highlighting statistically significant
  relationships.
* **PNG+SVG figures** and a **TSV** table of the full correlation
  results.

-----------------------------------------------------------------------
Quick‑start
-----------
1. **Prepare the functional table with FAPROTAX** (run once):

   ```bash
   # Export taxonomy so the first row reads exactly:  #OTUID\ttaxonomy
   biom add-metadata \
       -i exported-feature-table/feature-table.biom \
       -o feature-table-with-tax.biom \
       --observation-metadata-fp exported-taxonomy/taxonomy.tsv \
       --sc-separated taxonomy

   collapse_table.py \
       -i feature-table-with-tax.biom \
       -o functional_table.biom \
       -g FAPROTAX.txt \
       -r report.txt \
       -n columns_after_collapsing \
       -v \
       --collapse_by_metadata "taxonomy"

   biom convert \
       -i functional_table.biom \
       -o functional_table.tsv \
       --to-tsv
   ```

2. **Install dependencies** (one‑off):
   ```bash
   pip install pandas seaborn matplotlib scipy
   ```

3. **Edit the CONFIGURATION section** below to point to your files and
   select the metadata variable you wish to correlate.

4. **Run the script**:
   ```bash
   python faprotax_analysis.py
   ```

-----------------------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# CONFIGURATION – UPDATE THESE PATHS / FIELDS FOR YOUR PROJECT
# ---------------------------------------------------------------------------
BASE_DIR = Path("/path/to/project")  # ← EDIT
ANALYSIS_DIR = BASE_DIR / "analysis" / "FAPROTAX"
FUNC_TABLE_FILE = ANALYSIS_DIR / "functional_table.tsv"
METADATA_FILE = BASE_DIR / "data" / "metadata.tsv"

# Metadata
SAMPLE_TYPE_FIELD = "Sample_Type"
AEROSOL_LABEL = "Aerosol"
METADATA_VAR = "Lower_Quartile_Pressure [mbar]"  # ← variable to correlate

# Analysis parameters
TOP_N_FUNCTIONS = 25
ALPHA = 0.05  # p‑value threshold for significance

# Plot styling
FONT_FAMILY = "Helvetica"
FONT_SIZE = 14

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the FAPROTAX functional table and sample metadata."""
    func = pd.read_csv(FUNC_TABLE_FILE, sep="\t", skiprows=1, index_col=0)
    meta = pd.read_csv(METADATA_FILE, sep="\t", index_col=0)
    return func, meta


def filter_aerosol_samples(
    func: pd.DataFrame, meta: pd.DataFrame, variable: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return functional data and variable values for aerosol samples sorted by *variable*."""
    aero_meta = meta.loc[meta[SAMPLE_TYPE_FIELD] == AEROSOL_LABEL].copy()
    aero_meta.sort_values(variable, ascending=False, inplace=True)

    sample_ids = aero_meta.index.intersection(func.columns)
    if sample_ids.empty:
        raise ValueError(
            "No overlapping aerosol samples found between metadata and functional table."
        )

    func_aero = func.loc[:, sample_ids]
    return func_aero, aero_meta[variable]


def to_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """Convert counts to per‑sample relative abundance."""
    return df.div(df.sum(axis=0), axis=1)


def select_top_functions(rel: pd.DataFrame, n: int) -> List[str]:
    """Return *n* functions with the highest overall abundance."""
    return rel.sum(axis=1).nlargest(n).index.tolist()


def plot_heatmap(rel: pd.DataFrame, var_vals: pd.Series, outdir: Path) -> None:
    """Plot and save heatmap of *rel* with samples ordered by *var_vals*."""
    sns.set(font=FONT_FAMILY, font_scale=1.5)
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(rel, cmap="viridis", cbar_kws={"label": "Relative abundance"})

    # Rename misleading label if present
    ax.set_yticklabels(
        [
            "animal_associated" if t.get_text() == "human_associated" else t.get_text()
            for t in ax.get_yticklabels()
        ],
        rotation=0,
    )

    ax.set_xticklabels(var_vals.round(1), rotation=90)
    ax.set_xlabel(var_vals.name)
    ax.set_ylabel("Function")

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        plt.savefig(outdir / f"top_metabolisms.{ext}", bbox_inches="tight")
    plt.close()


def spearman_correlations(rel: pd.DataFrame, var: pd.Series) -> pd.DataFrame:
    """Compute Spearman rho and p‑value for each function vs. *var*."""
    records = []
    for func in rel.index:
        rho, pval = spearmanr(rel.loc[func], var)
        records.append({"Function": func, "Spearman_rho": rho, "p_value": pval})
    return (
        pd.DataFrame(records)
        .set_index("Function")
        .sort_values("Spearman_rho", ascending=False)
    )


def plot_significant_correlations(corr: pd.DataFrame, outdir: Path, alpha: float) -> None:
    """Bar plot of correlations with p < alpha."""
    sig = corr[corr["p_value"] < alpha]
    if sig.empty:
        print(f"No correlations significant at α = {alpha}.")
        return

    plt.figure(figsize=(8, sig.shape[0] * 0.4 + 2))
    sig["Spearman_rho"].plot(kind="barh")
    plt.xlabel("Spearman correlation (rho)")
    plt.title(f"Significant correlations with {METADATA_VAR}\n(α = {alpha})")
    plt.tight_layout()

    for ext in ("png", "svg"):
        plt.savefig(outdir / f"spearman.{ext}")
    plt.close()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    func_tbl, meta = load_data()
    func_aero, variable_values = filter_aerosol_samples(func_tbl, meta, METADATA_VAR)

    rel = to_relative_abundance(func_aero)
    top_funcs = select_top_functions(rel, TOP_N_FUNCTIONS)
    rel_top = rel.loc[top_funcs]

    plot_heatmap(rel_top, variable_values, ANALYSIS_DIR)

    corr_df = spearman_correlations(rel_top, variable_values)
    corr_df.to_csv(ANALYSIS_DIR / "spearman_correlations.tsv", sep="\t")

    plot_significant_correlations(corr_df, ANALYSIS_DIR, ALPHA)

    print("Analysis complete – results saved to", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
