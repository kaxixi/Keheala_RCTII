from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import os

# Paths - aligned with sibling directory structure (Data_Analysis/TIBU_data)
ROOT = Path(__file__).resolve().parent.parent / "TIBU_data"
DR_FOLDER = ROOT / "original_data/Drug_Resistant"
DS_FOLDER = ROOT / "original_data/Drug_Sensitive"
OUTPUT_FOLDER = ROOT / "output"
CSV_FILE = OUTPUT_FOLDER / "TIBU_firstnm_deidentified.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clean column names
def clean_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name)
    name = name.strip("_")
    if re.match(r"^\d", name):
        name = "x_" + name
    return name

# Column renaming map
RENAME_MAP = {
    "sub_county_registration_number": "subcountyregistrationnumber",
    "sub_county_registration_no": "subcountyregistrationnumber",
    "sub_county": "subcounty",
    "health_facility": "healthfacility",
    "zone": "zone",
    "county": "county",
    "province": "province",
    "patient_name": "patientname",
    "patient name": "patientname",
    "patientname": "patientname",
    "type_of_patient": "typeofpatient",
    "registration_group": "registrationgroup",  # keep this name separate for logic
    "sex_m/f": "sexmf",
    "sex_m_f_": "sexmf",
    "sex_m_f": "sexmf",
    "age_on_registration": "ageonregistration",
    "date_of_registration": "dateofregistration",
    "dateofreg": "dateofregistration",
    "hiv_status": "hivstatus",
    "comorbidity": "comorbidity",
    "type_of_tb_(p/ep)": "typeoftbpep",
    "type_of_tb(p/ep)": "typeoftbpep",
    "type_of_tb_p_ep": "typeoftbpep",
    "typeofpatient": "typeofpatient",
    "registration_group": "typeofpatient",
    "registrationgroup": "typeofpatient",
    "nutrition_support": "nutritionsupport",
    "resistance_pattern": "resistancepattern",
    "sputum_smear_examination_0th_month_result": "sputumsmearexamination0thmon",
    "gen_expert": "genexpert",
    "treatment_outcome": "treatmentoutcome",
    "treatment_outcome_date": "treatmentoutcomedate",
    "source_file": "source_file"
}

# Variables to keep
vars_to_keep = [
    "subcountyregistrationnumber", "patientname", "treatmentoutcome", "treatmentoutcomedate",
    "sexmf", "ageonregistration", "hivstatus", "comorbidity", "typeoftbpep", "typeofpatient",
    "nutritionsupport", "resistancepattern", "sputumsmearexamination0thmon", "genexpert",
    "province", "county", "subcounty", "zone", "healthfacility", "dateofregistration",
    "clinic_id", "source_file"
]

CLINIC_ID_MAP_FILE = OUTPUT_FOLDER / "clinic_id_mapping.csv"

# Load Excel files
def load_excel_files(folder, tag):
    print(f"Reading {tag} files from {folder}")
    dfs = []
    if not folder.exists():
        print(f"Warning: Folder {folder} does not exist.")
        return []
        
    for file in sorted(folder.glob("*.xlsx")):
        try:
            df = pd.read_excel(file, dtype=str, engine="openpyxl")
            df.columns = [clean_column_name(col) for col in df.columns]
            df.rename(columns=RENAME_MAP, inplace=True)

            # Fallback: if typeofpatient missing but registrationgroup exists, use it
            if "typeofpatient" not in df.columns and "registrationgroup" in df.columns:
                df["typeofpatient"] = df["registrationgroup"]

            if "patientname" not in df.columns:
                print(f"⚠️ {tag} file {file.name} is missing 'patientname'. Columns are:\n{df.columns.tolist()}")

            df["source_file"] = file.name
            dfs.append(df)
        except Exception as e:
            print(f"❌ {tag} file {file.name} failed to load: {e}")
    return dfs

# Format breakdown
def log_date_patterns(series, label):
    print(f"\n🔍 Date format breakdown for {label}:")
    iso_mask = series.str.match(r"^\d{4}-\d{2}-\d{2}")
    dmy_mask = series.str.match(r"^\d{1,2} [A-Za-z]+ \d{4}")
    empty_mask = series.isna() | (series.str.strip() == "")
    other_mask = ~(iso_mask | dmy_mask | empty_mask)

    print(f"  ISO format       : {iso_mask.sum():,}")
    print(f"  DMY format       : {dmy_mask.sum():,}")
    print(f"  Empty or missing : {empty_mask.sum():,}")
    print(f"  Other formats    : {other_mask.sum():,}")

# Parse and convert to Stata dates
def parse_and_convert_to_stata_days(series, label="(unnamed)"):
    series = series.fillna("").astype(str)
    series_clean = series.str.replace(r"[\r\n]", " ", regex=True).str.strip()

    stata_epoch = datetime(1960, 1, 1)
    parsed_dates = pd.to_datetime(series_clean, format='mixed', dayfirst=True, errors="coerce")

    failed_mask = parsed_dates.isna() & series_clean.ne("")
    failed_count = failed_mask.sum()
    total = len(series)
    parsed_nonnull = parsed_dates.dropna()

    print(f"\n📋 {label}: {failed_count:,} of {total:,} values could not be parsed ({failed_count/total:.1%} missing)")
    if not parsed_nonnull.empty:
        print(f"  ✅ Parsed date range: {parsed_nonnull.min().date()} to {parsed_nonnull.max().date()}")
    if failed_count > 0:
        failed_examples = series_clean[failed_mask].drop_duplicates().head(5).tolist()
        print(f"  ❓ Sample unparsed values: {failed_examples}")

    return (parsed_dates - stata_epoch).dt.days

def main():
    # Load and combine
    dr_dfs = load_excel_files(DR_FOLDER, "DR")
    ds_dfs = load_excel_files(DS_FOLDER, "DS")

    print("\nCombining data...")
    all_dfs = [df for df in dr_dfs + ds_dfs if not df.empty]
    if not all_dfs:
        print("No data found!")
        return
        
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)

    # Log total rows
    print(f"\n📦 Total rows before processing: {df_all.shape[0]:,}")

    # -------------------------------------------------------------------------
    # Deduplicate using trust-based ranking (matches Stata merge_and_deidentify_TIBU.do)
    # -------------------------------------------------------------------------
    print("Deduplicating with trust-based ranking (per Stata logic)...")

    # Create anonymous ID for grouping (SCRN + patient name)
    if "subcountyregistrationnumber" in df_all.columns and "patientname" in df_all.columns:
        df_all["anon_id"] = df_all["subcountyregistrationnumber"].astype(str) + "_" + df_all["patientname"].astype(str)
    elif "subcountyregistrationnumber" in df_all.columns:
        df_all["anon_id"] = df_all["subcountyregistrationnumber"].astype(str)
    else:
        df_all["anon_id"] = range(len(df_all))

    # Treatment outcome ranking (higher = more trusted, per Stata lines 57-70)
    treatment_rank = {
        "TC": 10, "C": 9, "D": 8, "F": 7, "NC": 6,
        "MT4": 5, "TO": 4, "NTB": 3, "LTFU": 2
    }
    if "treatmentoutcome" in df_all.columns:
        df_all["treatmentrank"] = df_all["treatmentoutcome"].astype(str).str.strip().str.upper().map(treatment_rank).fillna(1)
    else:
        df_all["treatmentrank"] = 1

    # HIV status ranking (Pos=4, Neg=3, D=2, ND=1)
    hiv_rank = {"Pos": 4, "Neg": 3, "D": 2, "ND": 1}
    if "hivstatus" in df_all.columns:
        df_all["hivstatus_rank"] = df_all["hivstatus"].astype(str).str.strip().map(hiv_rank).fillna(0)
    else:
        df_all["hivstatus_rank"] = 0

    # Type of TB ranking (EP=2, P=1)
    tbtype_rank = {"EP": 2, "P": 1}
    if "typeoftbpep" in df_all.columns:
        df_all["typeoftbpep_rank"] = df_all["typeoftbpep"].astype(str).str.strip().str.upper().map(tbtype_rank).fillna(0)
    else:
        df_all["typeoftbpep_rank"] = 0

    # Sort by anon_id and ranking criteria (descending for ranks = higher values first)
    # Stata: gsort anon_id -treatmentoutcomedate -treatmentrank -dateofregistration ...
    sort_cols = ["anon_id"]
    sort_ascending = [True]

    if "treatmentoutcomedate" in df_all.columns:
        # Convert to numeric for sorting (will be converted to Stata days later)
        df_all["tod_sort"] = pd.to_datetime(df_all["treatmentoutcomedate"], format='mixed', dayfirst=True, errors="coerce")
        sort_cols.append("tod_sort")
        sort_ascending.append(False)  # Most recent first

    sort_cols.extend(["treatmentrank", "hivstatus_rank", "typeoftbpep_rank"])
    sort_ascending.extend([False, False, False])  # Higher ranks first

    df_all = df_all.sort_values(sort_cols, ascending=sort_ascending, na_position='last')

    # Collapse: keep first non-null value for each field within each anon_id
    # This matches Stata's collapse (firstnm)
    before_dedup = len(df_all)
    df_all = df_all.groupby("anon_id", as_index=False).first()
    print(f"Collapsed {before_dedup:,} rows to {len(df_all):,} unique patients")

    # Clean up ranking columns
    drop_cols = ["anon_id", "treatmentrank", "hivstatus_rank", "typeoftbpep_rank", "tod_sort"]
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors='ignore')
    
    # Log date formats
    if "dateofregistration" in df_all.columns:
        log_date_patterns(df_all["dateofregistration"].astype(str), "dateofregistration")
    if "treatmentoutcomedate" in df_all.columns:
        log_date_patterns(df_all["treatmentoutcomedate"].astype(str), "treatmentoutcomedate")

    # Parse to Stata date format
    print("\n📅 Parsing and converting dates to Stata format...")
    if "dateofregistration" in df_all.columns:
        df_all["dateofregistration"] = parse_and_convert_to_stata_days(df_all["dateofregistration"], "dateofregistration")
    if "treatmentoutcomedate" in df_all.columns:
        df_all["treatmentoutcomedate"] = parse_and_convert_to_stata_days(df_all["treatmentoutcomedate"], "treatmentoutcomedate")

    # -------------------------------------------------------------------------
    # 4. Filter by Date (Stata: 13 Apr 2018 - 20 Dec 2019)
    # -------------------------------------------------------------------------
    start_ts = pd.Timestamp("2018-04-13")
    end_ts = pd.Timestamp("2019-12-20")
    stata_epoch = pd.Timestamp("1960-01-01")
    
    start_days = (start_ts - stata_epoch).days
    end_days = (end_ts - stata_epoch).days
    
    print(f"\nFiltering dateofregistration: {start_ts.date()} ({start_days}) to {end_ts.date()} ({end_days})")
    
    # Ensure column is numeric (handle NaNs if necessary)
    if "dateofregistration" in df_all.columns:
         # Fill NaNs with a value outside range or drop. Stata script: keep if inrange(...)
         # In Stata, missing (.) is > any number usually, but inrange handles it.
         # Stata: keep if inrange(field, min, max). If field is missing, it's dropped.
         
         df_all = df_all[
            (df_all["dateofregistration"].notna()) &
            (df_all["dateofregistration"] >= start_days) & 
            (df_all["dateofregistration"] <= end_days)
         ].copy()
         print(f"Rows after date filter: {len(df_all)}")
    
    # -------------------------------------------------------------------------
    # 5. Apply Exclusions (Match Stata _study2_TIBU_summarystats.do)
    # -------------------------------------------------------------------------
    print("Applying exclusions...")
    initial_n = len(df_all)
    
    # 1. Regex Exclusions on SCRN
    if "subcountyregistrationnumber" in df_all.columns:
        scrn_str = df_all["subcountyregistrationnumber"].astype(str)
        mask_dup = (
            scrn_str.str.contains("Double", case=False, na=False) |
            scrn_str.str.contains("Duplicate", case=False, na=False) |
            scrn_str.str.contains("Test", case=False, na=False)
        )
        df_all = df_all[~mask_dup]
        print(f"Dropped {initial_n - len(df_all)} rows due to Double/Duplicate/Test SCRN")
    
    # 2. Outcome Exclusions
    # Only exclude MT4 (duplicate marker) and NTB (not TB - misdiagnosed)
    # Keep TO, N/A, NAN, NONE, NC for Table 1 comparisons
    if "treatmentoutcome" in df_all.columns:
        df_all["outcome_norm"] = df_all["treatmentoutcome"].astype(str).str.strip().str.upper()
        mask_exclude = df_all["outcome_norm"].isin(["MT4", "NTB"])
        df_all = df_all[~mask_exclude]
        print(f"Rows after outcome exclusions (MT4, NTB only): {len(df_all)}")

    # -------------------------------------------------------------------------
    # 6. Assign clinic_id based on (healthfacility, subcounty) pairs
    # -------------------------------------------------------------------------
    print("Assigning clinic_id...")
    df_all["hf_lower"] = df_all["healthfacility"].astype(str).str.lower().str.strip()
    df_all["sc_lower"] = df_all["subcounty"].astype(str).str.lower().str.strip()
    df_all["clinic_id"] = df_all.groupby(["hf_lower", "sc_lower"]).ngroup()

    # Save the mapping for use by prepare_study_data.py
    clinic_map = (
        df_all.groupby(["hf_lower", "sc_lower", "clinic_id"])
        .agg(
            healthfacility=("healthfacility", "first"),
            subcounty=("subcounty", "first"),
            county=("county", "first"),
            n_patients=("clinic_id", "size"),
        )
        .reset_index()
        [["clinic_id", "healthfacility", "subcounty", "county", "hf_lower", "sc_lower", "n_patients"]]
        .sort_values("clinic_id")
    )
    clinic_map.to_csv(CLINIC_ID_MAP_FILE, index=False)
    print(f"  Saved clinic_id mapping ({len(clinic_map)} clinics) to {CLINIC_ID_MAP_FILE}")

    df_all = df_all.drop(columns=["hf_lower", "sc_lower"])

    # Keep selected vars
    df_all = df_all[[col for col in vars_to_keep if col in df_all.columns]]

    # Final diagnostics
    print("\nColumn presence in combined dataset:")
    for col in vars_to_keep:
        if col not in df_all.columns:
            print(f"❌ {col}: column not found")
        else:
            print(f"✓ {col}: {df_all[col].isna().sum():,} missing")

    print(f"\n📊 Final shape: {df_all.shape[0]:,} rows × {df_all.shape[1]} columns")

    # Save full file
    df_all.to_csv(CSV_FILE, index=False)
    print(f"📁 Also saved CSV to {CSV_FILE}")

    # Save 5% sample
    SAMPLE_FILE = OUTPUT_FOLDER / "TIBU_combined_sample.csv"
    if "subcountyregistrationnumber" in df_all.columns:
        unique_scrns = df_all["subcountyregistrationnumber"].dropna().unique()
        sampled_scrns = pd.Series(unique_scrns).sample(frac=0.05, random_state=42)
        df_sample = df_all[df_all["subcountyregistrationnumber"].isin(sampled_scrns)]
        df_sample.to_csv(SAMPLE_FILE, index=False)
        print(f"🧪 Saved 5% SCRN-based sample CSV to {SAMPLE_FILE}")
    else:
        print("⚠️ Cannot generate SCRN-based sample: 'subcountyregistrationnumber' not in dataset")

if __name__ == "__main__":
    main()
