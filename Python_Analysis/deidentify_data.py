"""
Keheala Study 2 - De-identification Pipeline
=============================================

Creates de-identified datasets at two levels for public sharing:
  Level 1: Names and phone numbers removed; IDs replaced with anonymous integers.
  Level 2: Names, phone, AND location identifiers removed; facility/county replaced with numeric IDs.

Inputs (must exist before running):
  - Python_Analysis/output/study2_cleaned_python.csv
  - TIBU_data/output/TIBU_firstnm_deidentified.csv
  - original_data/Urine_Test_Results.csv
  - original_data/DQA_*.csv (7 county files)

Outputs:
  deidentified_data/level1/
  deidentified_data/level2/

ID mappings are generated deterministically (sorted unique values -> sequential integers)
but are NOT saved to disk.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STUDY_DATA = os.path.join(ROOT_DIR, "Python_Analysis/output/study2_cleaned_python.csv")
TIBU_DATA = os.path.join(ROOT_DIR, "TIBU_data/output/TIBU_firstnm_deidentified.csv")
URINE_DATA = os.path.join(ROOT_DIR, "original_data/Urine_Test_Results.csv")
ORIGINAL_DATA_DIR = os.path.join(ROOT_DIR, "original_data")

OUTPUT_BASE = os.path.join(ROOT_DIR, "deidentified_data")
LEVEL1_DIR = os.path.join(OUTPUT_BASE, "level1")
LEVEL2_DIR = os.path.join(OUTPUT_BASE, "level2")

DQA_COUNTIES = ["Kakamega", "Kiambu", "Kisumu", "Machakos", "Mombasa", "Nairobi", "Turkana"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_id_map(series):
    """Sort unique non-null values and assign sequential integer IDs (1-based)."""
    unique_vals = sorted(series.dropna().unique(), key=str)
    return {v: i + 1 for i, v in enumerate(unique_vals)}


def apply_id_map(df, col, mapping, new_col):
    """Replace column values using mapping dict, rename column."""
    df[col] = df[col].map(mapping)
    if new_col != col:
        df = df.rename(columns={col: new_col})
    return df


def make_scrn_key(scrn_series, subcounty_series):
    """Create composite key from SCRN + subcounty for cross-county uniqueness.

    SCRN (subcounty registration number) is only unique within a subcounty,
    not globally. Using (scrn, subcounty) as composite key prevents merge
    errors when the same SCRN string appears in different subcounties.
    """
    scrn_str = scrn_series.astype(str).str.strip().str.lower()
    sc_str = subcounty_series.fillna("").astype(str).str.strip().str.lower()
    key = scrn_str + "|" + sc_str
    # Preserve NaN where scrn is missing
    key = key.where(scrn_series.notna(), other=np.nan)
    return key


def drop_cols(df, cols):
    """Drop columns if they exist."""
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Create output directories
    os.makedirs(LEVEL1_DIR, exist_ok=True)
    os.makedirs(LEVEL2_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load study data and build shared ID mappings
    # ------------------------------------------------------------------
    print("Loading study data...")
    if not os.path.exists(STUDY_DATA):
        print(f"Error: {STUDY_DATA} not found. Run prepare_study_data.py first.")
        return
    df_study = pd.read_csv(STUDY_DATA, low_memory=False)
    print(f"  Study data: {len(df_study):,} rows, {len(df_study.columns)} cols")

    # Build PatientID mapping (shared with Urine data)
    patient_id_map = build_id_map(df_study["PatientID"])
    print(f"  PatientID map: {len(patient_id_map):,} unique values")

    # Build SCRN mapping using (scrn, subcounty) composite key.
    # SCRN is only unique within a subcounty -- the same SCRN string can appear
    # in different counties (e.g., 13/1/1/2019 exists in both Kiambu and Nairobi).
    study_scrn_key = make_scrn_key(df_study["scrn"], df_study["syssubcounty"])
    scrn_map = build_id_map(study_scrn_key)
    print(f"  SCRN composite map: {len(scrn_map):,} unique (scrn, subcounty) pairs")

    # Build location mappings for Level 2
    county_map = build_id_map(df_study["syscounty"]) if "syscounty" in df_study.columns else {}
    subcounty_map = build_id_map(df_study["syssubcounty"]) if "syssubcounty" in df_study.columns else {}

    # Also include county/subcounty from non-sys columns if present
    if "county" in df_study.columns:
        for v in df_study["county"].dropna().unique():
            if v not in county_map:
                county_map[v] = max(county_map.values(), default=0) + 1
    if "subcounty" in df_study.columns:
        for v in df_study["subcounty"].dropna().unique():
            if v not in subcounty_map:
                subcounty_map[v] = max(subcounty_map.values(), default=0) + 1

    # Get list of participating clinics (for TIBU is_participating_clinic flag)
    participating_clinics = set()
    if "healthfacility" in df_study.columns:
        participating_clinics = set(
            df_study["healthfacility"].dropna().astype(str).str.lower().str.strip().unique()
        )
    elif "clinic" in df_study.columns:
        participating_clinics = set(
            df_study["clinic"].dropna().astype(str).str.lower().str.strip().unique()
        )
    print(f"  Participating clinics: {len(participating_clinics)}")

    # ------------------------------------------------------------------
    # 2. Load TIBU data
    # ------------------------------------------------------------------
    print("\nLoading TIBU data...")
    if not os.path.exists(TIBU_DATA):
        print(f"Warning: {TIBU_DATA} not found. Skipping TIBU de-identification.")
        df_tibu = None
    else:
        df_tibu = pd.read_csv(TIBU_DATA, low_memory=False)
        print(f"  TIBU data: {len(df_tibu):,} rows")

    # Build independent TIBU SCRN mapping
    if df_tibu is not None and "subcountyregistrationnumber" in df_tibu.columns:
        tibu_scrn_map = build_id_map(df_tibu["subcountyregistrationnumber"])
        print(f"  TIBU SCRN map: {len(tibu_scrn_map):,} unique values")
    else:
        tibu_scrn_map = {}

    # Build TIBU location mappings for Level 2
    tibu_province_map = {}
    tibu_county_map = {}
    tibu_subcounty_map = {}
    if df_tibu is not None:
        if "province" in df_tibu.columns:
            tibu_province_map = build_id_map(df_tibu["province"])
        if "county" in df_tibu.columns:
            tibu_county_map = build_id_map(df_tibu["county"])
        if "subcounty" in df_tibu.columns:
            tibu_subcounty_map = build_id_map(df_tibu["subcounty"])

    # ------------------------------------------------------------------
    # 3. Load Urine data
    # ------------------------------------------------------------------
    print("\nLoading Urine data...")
    if not os.path.exists(URINE_DATA):
        print(f"Warning: {URINE_DATA} not found. Skipping Urine de-identification.")
        df_urine = None
    else:
        df_urine = pd.read_csv(URINE_DATA)
        # Handle BOM in column names
        df_urine.columns = [c.lstrip('\ufeff') for c in df_urine.columns]
        print(f"  Urine data: {len(df_urine):,} rows")

    # ------------------------------------------------------------------
    # 4. Load DQA data
    # ------------------------------------------------------------------
    print("\nLoading DQA data...")
    dqa_dfs = {}
    for county in DQA_COUNTIES:
        path = os.path.join(ORIGINAL_DATA_DIR, f"DQA_{county}.csv")
        if os.path.exists(path):
            dqa_dfs[county] = pd.read_csv(path, dtype=str)
            print(f"  DQA_{county}: {len(dqa_dfs[county]):,} rows")
        else:
            print(f"  Warning: DQA_{county}.csv not found")

    # ==================================================================
    # LEVEL 1: Names removed
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 1: Names removed")
    print("=" * 60)

    # --- Study data Level 1 ---
    df_s1 = df_study.copy()
    df_s1 = drop_cols(df_s1, ["patientname", "phone"])
    df_s1 = apply_id_map(df_s1, "PatientID", patient_id_map, "anon_patient_id")
    # Map scrn using composite key (scrn + subcounty)
    df_s1["scrn"] = make_scrn_key(df_s1["scrn"], df_s1["syssubcounty"]).map(scrn_map)
    df_s1 = df_s1.rename(columns={"scrn": "anon_scrn"})
    if "subcountyregistrationnumber" in df_s1.columns:
        df_s1["subcountyregistrationnumber"] = make_scrn_key(
            df_s1["subcountyregistrationnumber"], df_s1["syssubcounty"]
        ).map(scrn_map)
        df_s1 = df_s1.rename(columns={"subcountyregistrationnumber": "anon_scrn_orig"})

    out_path = os.path.join(LEVEL1_DIR, "study2_cleaned.csv")
    df_s1.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df_s1):,} rows)")

    # --- TIBU data Level 1 ---
    if df_tibu is not None:
        df_t1 = df_tibu.copy()
        df_t1 = drop_cols(df_t1, ["patientname"])
        df_t1 = apply_id_map(df_t1, "subcountyregistrationnumber", tibu_scrn_map, "anon_scrn_tibu")

        out_path = os.path.join(LEVEL1_DIR, "TIBU_firstnm_deidentified.csv")
        df_t1.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_t1):,} rows)")

    # --- Urine data Level 1 ---
    if df_urine is not None:
        df_u1 = df_urine.copy()
        df_u1 = drop_cols(df_u1, ["Support_Sponsor"])
        df_u1 = apply_id_map(df_u1, "Patient ID", patient_id_map, "anon_patient_id")

        out_path = os.path.join(LEVEL1_DIR, "Urine_Test_Results.csv")
        df_u1.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_u1):,} rows)")

    # --- DQA data Level 1 ---
    for county, df_dqa in dqa_dfs.items():
        df_d1 = df_dqa.copy()
        df_d1 = drop_cols(df_d1, ["Name"])

        # Map SCRN using composite key (scrn + subcounty) -- same as study data
        scrn_col = "SCRN" if "SCRN" in df_d1.columns else "Patient ID"
        subcounty_col = "Sys Sub County"
        dqa_scrn_key = make_scrn_key(df_d1[scrn_col], df_d1[subcounty_col])
        df_d1[scrn_col] = dqa_scrn_key.map(scrn_map)
        df_d1 = df_d1.rename(columns={scrn_col: "anon_scrn"})

        out_path = os.path.join(LEVEL1_DIR, f"DQA_{county}.csv")
        df_d1.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_d1):,} rows)")

    # ==================================================================
    # LEVEL 2: Names AND locations removed
    # ==================================================================
    print("\n" + "=" * 60)
    print("LEVEL 2: Names AND locations removed")
    print("=" * 60)

    # --- Study data Level 2 ---
    df_s2 = df_study.copy()
    df_s2 = drop_cols(df_s2, [
        "patientname", "phone",
        "clinic", "healthfacility", "zone", "province", "serialnumber"
    ])
    df_s2 = apply_id_map(df_s2, "PatientID", patient_id_map, "anon_patient_id")
    # Map scrn using composite key (must happen before syssubcounty is replaced)
    df_s2["scrn"] = make_scrn_key(df_s2["scrn"], df_s2["syssubcounty"]).map(scrn_map)
    df_s2 = df_s2.rename(columns={"scrn": "anon_scrn"})
    if "subcountyregistrationnumber" in df_s2.columns:
        df_s2["subcountyregistrationnumber"] = make_scrn_key(
            df_s2["subcountyregistrationnumber"], df_s2["syssubcounty"]
        ).map(scrn_map)
        df_s2 = df_s2.rename(columns={"subcountyregistrationnumber": "anon_scrn_orig"})

    # Replace location columns with IDs
    if "syscounty" in df_s2.columns:
        df_s2 = apply_id_map(df_s2, "syscounty", county_map, "county_id")
    if "syssubcounty" in df_s2.columns:
        df_s2 = apply_id_map(df_s2, "syssubcounty", subcounty_map, "subcounty_id")
    if "county" in df_s2.columns:
        df_s2 = apply_id_map(df_s2, "county", county_map, "county_id")
    if "subcounty" in df_s2.columns:
        df_s2 = apply_id_map(df_s2, "subcounty", subcounty_map, "subcounty_id")

    out_path = os.path.join(LEVEL2_DIR, "study2_cleaned.csv")
    df_s2.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df_s2):,} rows)")

    # --- TIBU data Level 2 ---
    if df_tibu is not None:
        df_t2 = df_tibu.copy()

        # Add is_participating_clinic flag BEFORE stripping facility names
        if "healthfacility" in df_t2.columns:
            df_t2["is_participating_clinic"] = (
                df_t2["healthfacility"]
                .astype(str).str.lower().str.strip()
                .isin(participating_clinics)
            ).astype(int)
        else:
            df_t2["is_participating_clinic"] = 0

        df_t2 = drop_cols(df_t2, ["patientname", "healthfacility", "zone"])
        df_t2 = apply_id_map(df_t2, "subcountyregistrationnumber", tibu_scrn_map, "anon_scrn_tibu")

        # Replace location columns with IDs
        if "province" in df_t2.columns:
            df_t2 = apply_id_map(df_t2, "province", tibu_province_map, "province_id")
        if "county" in df_t2.columns:
            df_t2 = apply_id_map(df_t2, "county", tibu_county_map, "county_id")
        if "subcounty" in df_t2.columns:
            df_t2 = apply_id_map(df_t2, "subcounty", tibu_subcounty_map, "subcounty_id")

        out_path = os.path.join(LEVEL2_DIR, "TIBU_firstnm_deidentified.csv")
        df_t2.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_t2):,} rows)")

    # --- Urine data Level 2 ---
    if df_urine is not None:
        df_u2 = df_urine.copy()
        df_u2 = drop_cols(df_u2, ["Support_Sponsor", "Facility_name", "County"])
        df_u2 = apply_id_map(df_u2, "Patient ID", patient_id_map, "anon_patient_id")

        out_path = os.path.join(LEVEL2_DIR, "Urine_Test_Results.csv")
        df_u2.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_u2):,} rows)")

    # --- DQA data Level 2 (combined into single file to remove county from filename) ---
    dqa_l2_parts = []
    for county, df_dqa in dqa_dfs.items():
        df_d2 = df_dqa.copy()

        # Map SCRN using composite key BEFORE dropping Sys Sub County
        scrn_col = "SCRN" if "SCRN" in df_d2.columns else "Patient ID"
        dqa_scrn_key = make_scrn_key(df_d2[scrn_col], df_d2["Sys Sub County"])
        df_d2[scrn_col] = dqa_scrn_key.map(scrn_map)
        df_d2 = df_d2.rename(columns={scrn_col: "anon_scrn"})

        df_d2 = drop_cols(df_d2, ["Name", "Clinic", "Sys County", "Sys Sub County"])

        dqa_l2_parts.append(df_d2)

    if dqa_l2_parts:
        df_dqa_l2 = pd.concat(dqa_l2_parts, ignore_index=True)
        out_path = os.path.join(LEVEL2_DIR, "DQA_combined.csv")
        df_dqa_l2.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_dqa_l2):,} rows)")

    # ==================================================================
    # Verification
    # ==================================================================
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check no PII columns remain
    pii_cols = {"patientname", "phone", "Name", "Support_Sponsor"}
    for level_name, level_dir in [("Level 1", LEVEL1_DIR), ("Level 2", LEVEL2_DIR)]:
        print(f"\n  {level_name}:")
        for fname in sorted(os.listdir(level_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(level_dir, fname)
            cols = set(pd.read_csv(fpath, nrows=0).columns)
            found_pii = cols & pii_cols
            status = "CLEAN" if not found_pii else f"PII FOUND: {found_pii}"
            print(f"    {fname}: {status}")

    # Cross-file ID consistency: Study <-> Urine on anon_patient_id
    if df_urine is not None:
        l1_study = pd.read_csv(os.path.join(LEVEL1_DIR, "study2_cleaned.csv"), usecols=["anon_patient_id"], low_memory=False)
        l1_urine = pd.read_csv(os.path.join(LEVEL1_DIR, "Urine_Test_Results.csv"), usecols=["anon_patient_id"])
        l1_urine_dedup = l1_urine.drop_duplicates(subset=["anon_patient_id"])
        merge_count = pd.merge(l1_study, l1_urine_dedup, on="anon_patient_id", how="inner").shape[0]

        # Original merge count
        orig_urine = pd.read_csv(URINE_DATA)
        orig_urine.columns = [c.lstrip('\ufeff') for c in orig_urine.columns]
        orig_urine = orig_urine.dropna(subset=["Patient ID"]).drop_duplicates(subset=["Patient ID"])
        orig_merge = pd.merge(df_study, orig_urine, left_on="PatientID", right_on="Patient ID", how="inner").shape[0]
        match_str = "MATCH" if merge_count == orig_merge else f"MISMATCH ({merge_count} vs {orig_merge})"
        print(f"\n  Study<->Urine merge count: {match_str}")

    # Cross-file ID consistency: Study <-> DQA on anon_scrn
    if dqa_dfs:
        l1_study = pd.read_csv(os.path.join(LEVEL1_DIR, "study2_cleaned.csv"), usecols=["anon_scrn"], low_memory=False)
        l1_dqa_parts = []
        for county in DQA_COUNTIES:
            fpath = os.path.join(LEVEL1_DIR, f"DQA_{county}.csv")
            if os.path.exists(fpath):
                l1_dqa_parts.append(pd.read_csv(fpath, usecols=["anon_scrn"]))
        if l1_dqa_parts:
            l1_dqa = pd.concat(l1_dqa_parts, ignore_index=True)
            l1_dqa_dedup = l1_dqa.drop_duplicates(subset=["anon_scrn"])
            deid_merge = pd.merge(l1_study, l1_dqa_dedup, on="anon_scrn", how="inner").shape[0]

            # Original merge count (using composite key for apples-to-apples)
            orig_dqa_parts = []
            for county in DQA_COUNTIES:
                path = os.path.join(ORIGINAL_DATA_DIR, f"DQA_{county}.csv")
                if os.path.exists(path):
                    d = pd.read_csv(path, dtype=str)
                    d.columns = [c.strip().lower().replace(" ", "") for c in d.columns]
                    if "scrn" not in d.columns and "patientid" in d.columns:
                        d = d.rename(columns={"patientid": "scrn"})
                    orig_dqa_parts.append(d)
            if orig_dqa_parts:
                orig_dqa = pd.concat(orig_dqa_parts, ignore_index=True)
                # Use composite key for original merge too
                orig_dqa["_key"] = make_scrn_key(orig_dqa["scrn"], orig_dqa.get("syssubcounty"))
                orig_dqa_dedup = orig_dqa.drop_duplicates(subset=["_key"])
                df_study_keyed = df_study.copy()
                df_study_keyed["_key"] = make_scrn_key(df_study_keyed["scrn"], df_study_keyed["syssubcounty"])
                orig_merge_dqa = pd.merge(df_study_keyed, orig_dqa_dedup[["_key"]], on="_key", how="inner").shape[0]
                match_str = "MATCH" if deid_merge == orig_merge_dqa else f"MISMATCH ({deid_merge} vs {orig_merge_dqa})"
                print(f"  Study<->DQA merge count: {match_str}")

    print("\nDe-identification complete.")


if __name__ == "__main__":
    main()
