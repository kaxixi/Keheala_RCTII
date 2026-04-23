
"""
Keheala Study 2 - CONSORT Flow Diagram Generator
================================================

Purpose:
    Generates the data points for the CONSORT flow diagram, plus a by-arm
    follow-up-duration summary used in the Patient Characteristics section
    of the manuscript.
    Outputs a structured text log (output/consort_flow.log) visualizing the flow:
    Assessed -> Excluded -> Randomized (MITT) -> Allocation -> Analysis (MITT),
    followed by follow-up-duration statistics by arm.

Logic:
    1. Loads cleaned (de-identified) study data which preserves all rows
       including duplicates, TO, NC, and missing-outcome cases.
    2. Uses pre-computed flags (duplicate, MITT) and treatmentoutcome values
       to derive exclusion counts.
    3. Computes follow-up duration (days between dateregistered_formatted
       and treatmentoutcomedate_formatted) for the mITT sample, dropping
       records with implausible date pairs (negative or >5 years).

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

    # --- Follow-up Duration (mITT) ---
    log("\n\n--- Follow-up Duration (mITT) ---")
    log("Days from enrolment (dateregistered_formatted) to outcome recording")
    log("(treatmentoutcomedate_formatted).  Records with implausible date pairs")
    log("(negative follow-up, or > 5 years) are excluded.")

    fu = df_mitt.copy()
    fu["_enroll"] = pd.to_datetime(fu["dateregistered_formatted"], errors="coerce")
    fu["_outcome"] = pd.to_datetime(fu["treatmentoutcomedate_formatted"], errors="coerce")
    fu["_followup_days"] = (fu["_outcome"] - fu["_enroll"]).dt.days

    valid = fu[(fu["_followup_days"] >= 0) & (fu["_followup_days"] <= 1825)]
    n_dropped = len(fu) - len(valid)
    log(f"\nValid follow-up records: {len(valid):,} of {len(fu):,} mITT "
        f"({n_dropped:,} dropped for implausible dates)")

    fu_fmt = "  {:<22} {:>6} {:>8} {:>8} {:>8} {:>8}"
    log("")
    log(fu_fmt.format("Arm", "N", "Median", "Q1", "Q3", "Max"))
    log(fu_fmt.format("-" * 22, "-" * 6, "-" * 8, "-" * 8, "-" * 8, "-" * 8))

    def _row(label, d):
        if len(d) == 0:
            return fu_fmt.format(label, "0", "-", "-", "-", "-")
        return fu_fmt.format(
            label,
            f"{len(d):,}",
            f"{d.median():.0f}",
            f"{d.quantile(0.25):.0f}",
            f"{d.quantile(0.75):.0f}",
            f"{d.max():.0f}",
        )

    for g in groups:
        log(_row(g, valid.loc[valid["treatment_group"] == g, "_followup_days"]))
    log(_row("All mITT", valid["_followup_days"]))
    log("  (follow-up expressed in days)")

    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = OUTPUT_DIR / "consort_flow.log"
    with open(out_file, "w") as f:
        f.write("\n".join(log_buffer))
    print(f"\n[Saved log to {out_file}]")

if __name__ == "__main__":
    generate_consort()
