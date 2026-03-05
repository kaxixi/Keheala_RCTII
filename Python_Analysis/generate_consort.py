
"""
Keheala Study 2 - CONSORT Flow Diagram Generator
================================================

Purpose:
    Generates the data points for the CONSORT flow diagram.
    Outputs a structured text log (output/consort_flow.log) visualizing the flow:
    Assessed -> Excluded -> Randomized (MITT) -> Allocation -> Analysis (MITT).

Logic:
    1. Loads cleaned (de-identified) study data which preserves all rows
       including duplicates, TO, NC, and missing-outcome cases.
    2. Uses pre-computed flags (duplicate, MITT) and treatmentoutcome values
       to derive exclusion counts.

Output:
    - Console printout.
    - output/consort_flow.log

"""

import pandas as pd
from pathlib import Path
import os

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DEIDENTIFIED_DIR = ROOT_DIR / "deidentified_data" / "level2"
INPUT_DATA = DEIDENTIFIED_DIR / "study2_cleaned.csv"
OUTPUT_DIR = ROOT_DIR / "Python_Analysis/output"

def generate_consort():
    log_buffer = []

    def log(msg=""):
        print(msg)
        log_buffer.append(msg)

    log("--- CONSORT Flow Diagram Data Generation ---")

    # 1. Load Cleaned Data (all rows preserved, including duplicates)
    if not INPUT_DATA.exists():
        log(f"Error: Data file not found at {INPUT_DATA}")
        return

    df = pd.read_csv(INPUT_DATA, low_memory=False)
    raw_n = len(df)

    # 2. Exclusions using pre-computed flags
    n_duplicates = (df["duplicate"] == 1).sum()

    # ITT: non-duplicates (keeps all outcomes including TO, NC, N/A)
    df_itt = df[df["duplicate"] == 0].copy()
    itt_n = len(df_itt)

    # Further exclusions from ITT
    n_mdto = (df_itt["treatmentoutcome"] == "TO").sum()
    n_na = df_itt["treatmentoutcome"].isna().sum()
    n_nc = (df_itt["treatmentoutcome"] == "NC").sum()
    n_excluded_from_itt = n_mdto + n_na + n_nc

    # mITT
    df_mitt = df[df["MITT"] == 1].copy()
    mitt_n = len(df_mitt)

    # Group Breakdown
    groups = ["Control Group", "SMS Reminder Group", "SBCC Group", "Keheala Group"]
    group_counts_itt = df_itt["treatment_group"].value_counts().to_dict()
    group_counts_mitt = df_mitt["treatment_group"].value_counts().to_dict()

    # --- Generate Text Diagram ---
    log("\n")
    log(f"                         Assessed for Eligibility")
    log(f"                                (N={raw_n:,})")
    log(f"                                   |")
    log(f"                                   v")
    log(f"                         Excluded (n={n_duplicates:,})")
    log(f"                           |-- Duplicates/Test/MT4: {n_duplicates:,}")
    log(f"                                   |")
    log(f"                                   v")
    log(f"                         Randomized (ITT N={itt_n:,})")
    log(f"                                   |")
    log(f"                                   v")
    log(f"                         Excluded (n={n_excluded_from_itt:,})")
    log(f"                           |-- Transferred Out:     {n_mdto:,}")
    log(f"                           |-- Missing Outcome/NA:  {n_na:,}")
    log(f"                           |-- Not Evaluated (NC):  {n_nc:,}")
    log(f"                                   |")
    log(f"                                   v")
    log(f"                         Analyzed (mITT N={mitt_n:,})")
    log(f"                                   |")
    log(f"          ---------------------------------------------------")

    log("\nAllocation & Analysis:")

    row_fmt = "{:<22} {:<20} {:<20} {:<20}"
    log(row_fmt.format("Control", "SMS Reminder", "Platform (SBCC)", "Keheala"))
    log(row_fmt.format("-" * 20, "-" * 18, "-" * 18, "-" * 18))

    # Allocated (ITT)
    allocated = [group_counts_itt.get(g, 0) for g in groups]
    log(row_fmt.format(*[f"Allocated: {n:,}" for n in allocated]))

    # Analyzed (mITT)
    analyzed = [group_counts_mitt.get(g, 0) for g in groups]
    log(row_fmt.format(*[f"Analyzed:  {n:,}" for n in analyzed]))

    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = OUTPUT_DIR / "consort_flow.log"
    with open(out_file, "w") as f:
        f.write("\n".join(log_buffer))
    print(f"\n[Saved log to {out_file}]")

if __name__ == "__main__":
    generate_consort()
