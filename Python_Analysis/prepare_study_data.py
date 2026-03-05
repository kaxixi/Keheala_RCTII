import pandas as pd
import numpy as np
import os
import re

# Set Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_DATA_DIR = os.path.join(ROOT_DIR, "original_data")
STATA_ANALYSIS_DIR = os.path.join(ROOT_DIR, "Stata_Analysis") # Found here
OUTPUT_DIR = os.path.join(ROOT_DIR, "Python_Analysis/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_data():
    # 1. Load Data
    # insheet using "`original_data'/keheala_study2_outcome_data-ORIGINAL.csv", comma names clear
    # Keep "N/A" as string to match Stata behavior (it treats only empty as missing usually, unless configured)
    # Pandas defaults treat "N/A" as NaN.
    # We set keep_default_na=False and specify empty string as NaN.
    df = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "keheala_study2_outcome_data-ORIGINAL.csv"), 
                     low_memory=False, keep_default_na=False, na_values=[''])

    # Normalization of headers to match Stata's insheet behavior:
    # Lowercase, remove spaces and special characters.
    # Also truncate to 32 characters as Stata does (older versions or compatibility).
    def normalize_col(col):
        # Remove anything that isn't a-z, 0-9
        normalized = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
        return normalized[:32]
    
    df.columns = [normalize_col(c) for c in df.columns]

    # 2. Rename
    # rename sysgroup treatment_group
    df = df.rename(columns={"sysgroup": "treatment_group"})

    # 3. Drop Misdiagnosed, Transfer-Outs, and Known Duplicates
    # gen duplicate = 0 -> replace if regexm(scrn, "Double")...
    duplicate_mask = df["scrn"].str.contains("Double|double|Duplicate|Test", regex=True, na=False) | (df["treatmentoutcome"] == "MT4")
    df["duplicate"] = duplicate_mask.astype(int)

    # gen MDTO = 0 -> replace if TO
    df["MDTO"] = (df["treatmentoutcome"] == "TO").astype(int)

    # gen prereg_exclusion = 0 -> replace if duplicate | MDTO | N/A | missing | NC
    prereg_mask = (df["duplicate"] == 1) | (df["MDTO"] == 1) | (df["treatmentoutcome"] == "N/A") | df["treatmentoutcome"].isna() | (df["treatmentoutcome"] == "NC")
    df["prereg_exclusion"] = prereg_mask.astype(int)

    # 4. Prepare Outcomes
    # gen unsuccessful_outcome
    df["unsuccessful_outcome"] = np.nan
    df.loc[df["treatmentoutcome"].isin(["C", "TC"]), "unsuccessful_outcome"] = 0
    df.loc[df["treatmentoutcome"].isin(["D", "F", "LTFU", "NC"]), "unsuccessful_outcome"] = 1

    # gen died
    df["died"] = np.nan
    df.loc[df["treatmentoutcome"].isin(["C", "TC", "F", "LTFU", "NC"]), "died"] = 0
    df.loc[df["treatmentoutcome"] == "D", "died"] = 1
    
    # LTFU
    df["LTFU"] = np.nan
    df.loc[df["treatmentoutcome"].notna(), "LTFU"] = 0
    df.loc[df["treatmentoutcome"] == "LTFU", "LTFU"] = 1

    # gen failed
    # Stata: replace failed=0; replace failed=. if missing; replace failed=1 if F
    # With N/A preserved as string, it is NOT missing (isna() is False). So failed stays 0.
    df["failed"] = 0
    df.loc[df["treatmentoutcome"].isna(), "failed"] = np.nan
    df.loc[df["treatmentoutcome"] == "F", "failed"] = 1

    # 5. Prepare IVs (Encoding)
    df["treatment_group_str"] = df["treatment_group"] 
    
    # language
    # regen gender
    # replace sexmf = "M" if sexmf == "Male"
    df["sexmf"] = df["sexmf"].replace("Male", "M")
    
    # gen male
    df["male"] = np.nan
    df.loc[df["sexmf"] == "M", "male"] = 1
    df.loc[df["sexmf"] == "F", "male"] = 0
    
    # 6. Age Parsing
    # destring ageonregistration
    # It seems ageonregistration is a string like "34Y" or "2M".
    # Stata logic lines 97-102 iterates year 0-120.
    
    # We need to emulate the exact behavior. 
    # Creating age_year/age_month columns first
    # emulating Stata logic
    df["age_year"] = df["ageonregistration"].astype(str).str.extract(r'(\d+)Y').astype(float)
    df["age_month"] = df["ageonregistration"].astype(str).str.extract(r'(\d+)M').astype(float).fillna(0)
    
    # If no "Y" found, Stata `destring ageonregistration` might have converted it if it was pure number.
    # In pandas, if extract fails, we get NaN.
    # Try just numeric conversion for those that failed extraction but are numbers?
    # Stata: `destring ageonregistration, replace`.
    # If `ageonregistration` == "34", Stata makes it int 34.
    # Then loop: `regexm(ageonregistration, "0Y")` will likely fail if it's a number?
    # Actually regexm in Stata on a numeric variable coerces to string?
    # Regardless, we can likely assume 34 -> 34 years.
    
    # Fallback for plain numbers
    mask_numeric = df["ageonregistration"].astype(str).str.isdigit()
    df.loc[mask_numeric & df["age_year"].isna(), "age_year"] = df.loc[mask_numeric & df["age_year"].isna(), "ageonregistration"].astype(float)
    
    # Stata Line 103: replace age_year = 12 if age_year ==120
    df.loc[df["age_year"] == 120, "age_year"] = 12

    df["age_in_months"] = df["age_year"] * 12 + df["age_month"]
    
    # 7. Dates
    # clock(..., "YMD hms") -> dofc
    # Python: pd.to_datetime
    date_cols = ["dateoftreatmentstarted", "dateregistered", "treatmentoutcomedate"]
    for col in date_cols:
        # Check format. Stata uses "YMD hms".
        df[f"{col}_formatted"] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
        # Stata `dofc` keeps just the date part (floored to day).
        # We can keep timestamp or normalize to midnight.
        # df[f"{col}_formatted"] = df[f"{col}_formatted"].dt.normalize() # Already done above

    # 8. Treatment Month When Enrolled
    # (dateregistered - datetreatmentstarted) / 30.5
    df["treatment_month_when_enrolled"] = (df["dateregistered_formatted"] - df["dateoftreatmentstarted_formatted"]).dt.days / 30.5
    
    # Caps/Floors (Lines 115-117)
    # replace = 0 if < 0 & >= -2
    mask_near_zero = (df["treatment_month_when_enrolled"] < 0) & (df["treatment_month_when_enrolled"] >= -2)
    df.loc[mask_near_zero, "treatment_month_when_enrolled"] = 0
    
    # replace = . if < -2
    df.loc[df["treatment_month_when_enrolled"] < -2, "treatment_month_when_enrolled"] = np.nan
    
    # replace = . if > 10
    df.loc[df["treatment_month_when_enrolled"] > 10, "treatment_month_when_enrolled"] = np.nan

    # 9. Study Month When Enrolled (New logic)
    # mofd(dateregistered) - mofd(13apr2018)
    # study_start = pd.Timestamp("2018-04-13")
    # Python period arithmetic
    # df["study_month_enrolled"] = df["dateregistered_formatted"].dt.to_period('M').astype(int) - study_start.to_period('M').astype(int)
    # mofd(dateregistered) - mofd(13apr2018)
    study_start_mofd = (2018 * 12) + (4 - 1) # April is 4th month, index from 0? or 1?
    # (year - 2018)*12 + (month - 4)
    # 2018-04-13 -> 0
    # 2018-05-01 -> 1
    df["study_month_when_enrolled"] = (df["dateregistered_formatted"].dt.year - 2018) * 12 + (df["dateregistered_formatted"].dt.month - 4)

    # 10. Nutrition & Comorbidity
    # Note: Original data contains "N/A" (no support/comorbidity) AND "NaN" strings (truly missing)

    # nutritionsupport
    # FIX: Distinguish between missing (unknown) vs "N/A" (no support) vs documented support
    df["nutritionsupport_dummy"] = np.nan
    df.loc[df["nutritionsupport"] == "N/A", "nutritionsupport_dummy"] = 0  # Explicitly no support
    # Has support: not null, not "N/A", and not literal "NaN" string
    has_support_mask = (
        df["nutritionsupport"].notna() &
        (df["nutritionsupport"] != "N/A") &
        (df["nutritionsupport"] != "NaN")
    )
    df.loc[has_support_mask, "nutritionsupport_dummy"] = 1
    # Missing nutritionsupport (including "NaN" strings) remains NaN (unknown)

    # comorbidity
    # FIX: Distinguish between missing (unknown) vs "N/A" (no comorbidity) vs documented comorbidity
    df["comorbidity_dummy"] = np.nan
    df.loc[df["comorbidity"] == "N/A", "comorbidity_dummy"] = 0  # Explicitly no comorbidity
    # Has comorbidity: not null, not "N/A", and not literal "NaN" string
    has_comorbidity_mask = (
        df["comorbidity"].notna() &
        (df["comorbidity"] != "N/A") &
        (df["comorbidity"] != "NaN")
    )
    df.loc[has_comorbidity_mask, "comorbidity_dummy"] = 1
    # Missing comorbidity (including "NaN" strings) remains NaN (unknown)
    
    # 11. Regimen
    # Replacements
    df.loc[df["regimen"].isin(["2RHZ/4R H", "2RHZE/4 RH", "2RHZE/4RH"]), "regimen"] = "2RHZ/4RH"
    df.loc[df["regimen"] == "2RHZE/1RHZE/5RHE", "regimen"] = "2SRHZE/1RHZE/5RHE"
    
    # 12. Drug Resistant
    # FIX: Missing drug resistance status should be NaN, not 0
    df["drugresistant_dummy"] = np.nan
    df.loc[df["drugresistant"] == "DR", "drugresistant_dummy"] = 1
    # Set to 0 only for known drug-sensitive values (DS or non-DR documented values)
    df.loc[df["drugresistant"].notna() & (df["drugresistant"] != "DR") & (df["drugresistant"] != ""), "drugresistant_dummy"] = 0
    df = df.rename(columns={"drugresistant": "drugresistant_old", "drugresistant_dummy": "drugresistant"})

    # 13. English
    # FIX: Missing language should remain NaN, not be treated as non-English
    # In pandas, NaN != "English" evaluates to True, so we must explicitly check notna()
    unique_langs = sorted(df["language"].dropna().unique())
    df["English"] = np.nan
    df.loc[df["language"] == "English", "English"] = 1
    df.loc[(df["language"].notna()) & (df["language"] != "English"), "English"] = 0 

    # 14. Age in Years
    df["age_in_years"] = df["age_in_months"] / 12
    
    # 14b. HIV Positive (New)
    # FIX: Missing HIV status should be NaN, not 0 (which would treat unknown as negative)
    df["hiv_positive"] = np.nan
    df.loc[df["hivstatus"] == "Pos", "hiv_positive"] = 1
    # Set to 0 only for known negative values
    df.loc[df["hivstatus"].isin(["Neg", "Negative", "NEG"]), "hiv_positive"] = 0
    # Also check for other non-positive documented values (unknown status remains NaN)
    df.loc[df["hivstatus"].isin(["Unknown", "Not Done", "ND"]), "hiv_positive"] = np.nan

    # Clinic ID
    if "subcounty" in df.columns and "clinic" in df.columns:
        df["clinic_id"] = df.groupby(["subcounty", "clinic"]).ngroup()
    else:
        # Fallback if columns missing?
        print("Warning: subcounty or clinic missing for clinic_id creation")
        df["clinic_id"] = 0

    # 15. Bacteriologically Confirmed
    # FIX: Distinguish between "not tested" (NaN) vs "tested negative" (0) vs "confirmed" (1)
    df["bacteriologically_confirmed"] = np.nan
    # Note: Stata checks regexm(genexpert, "MTB").
    mask_bact = (df["sputumsmearexamination0thmonthre"] == "Pos") | (df["genexpert"].str.contains("MTB", na=False))
    df.loc[mask_bact, "bacteriologically_confirmed"] = 1
    # Set to 0 only if tested but not positive (has a result but not MTB/Pos)
    mask_tested_negative = (
        (df["sputumsmearexamination0thmonthre"].notna() & (df["sputumsmearexamination0thmonthre"] != "Pos")) |
        (df["genexpert"].notna() & ~df["genexpert"].str.contains("MTB", na=True))
    ) & ~mask_bact
    df.loc[mask_tested_negative, "bacteriologically_confirmed"] = 0

    # 16. Retreatment / Extrapulmonary
    # retreatment
    df["retreatment"] = np.nan
    df.loc[df["typeofpatient"] == "N", "retreatment"] = 0
    df.loc[df["typeofpatient"].isin(["F", "R", "TLF"]), "retreatment"] = 1
    
    # extrapulmonary
    df["extrapulmonary"] = np.nan
    df.loc[df["typeoftbpep"] == "P", "extrapulmonary"] = 0
    df.loc[df["typeoftbpep"] == "EP", "extrapulmonary"] = 1

    # 17. Rename PatientID
    df = df.rename(columns={"patientid": "PatientID"})

    # 18. Merges (Malfunction)
    # insheet `Patients_Affected_by_Call_Queue_Malfunction.csv`
    malf_path = os.path.join(ORIGINAL_DATA_DIR, "Patients_Affected_by_Call_Queue_Malfunction.csv")
    if os.path.exists(malf_path):
        malf_df = pd.read_csv(malf_path, keep_default_na=False, na_values=[''])
        # normalize header
        malf_df.columns = [normalize_col(c) for c in malf_df.columns]
        # Rename patientid -> PatientID for join
        malf_df = malf_df.rename(columns={"patientid": "PatientID"})
        
        malf_ids = malf_df["PatientID"].unique()
        df["affected_by_malfunction"] = 0
        df.loc[df["PatientID"].isin(malf_ids), "affected_by_malfunction"] = 1
    else:
        print("Warning: Malfunction CSV not found.")
        df["affected_by_malfunction"] = 0
    
    # 19. Merge Verification (Daily exports)
    # This part is complex (Lines 208-234). 
    # Load `Appended_Daily_Verification_Exports.dta`? Stata file.
    # pd.read_stata works.
    # We might need to implement this in a separate block or file if it's large.
    verif_path = os.path.join(STATA_ANALYSIS_DIR, "Appended_Daily_Verification_Exports.dta")
    if os.path.exists(verif_path):
        vdf = pd.read_stata(verif_path)
        
        # keep if TreatmentVerified (Assuming checking for True/1)
        # Check column type?
        if "TreatmentVerified" in vdf.columns:
            # Stata treated it as boolean likely
            pass
        
        # We need to replicate:
        # keep if TreatmentVerified
        vdf = vdf[vdf["TreatmentVerified"] != 0] # Assuming 0/1 or similar
        
        # sort PatientID Date
        # rename Date verification_date
        if "Date" in vdf.columns:
            vdf = vdf.rename(columns={"Date": "verification_date"})
        
        vdf = vdf.sort_values(["PatientID", "verification_date"])
        
        # by PatientID: egen verification_num = seq()
        vdf["verification_num"] = vdf.groupby("PatientID").cumcount() + 1
        
        # keep if verification_num <= 3
        vdf = vdf[vdf["verification_num"] <= 3]
        
        # reshape wide
        # keep PatientID verification_date verification_num
        vdf_wide = vdf.pivot(index="PatientID", columns="verification_num", values="verification_date")
        # Rename columns to verification_date1, verification_date2...
        vdf_wide.columns = [f"verification_date{c}" for c in vdf_wide.columns]
        vdf_wide = vdf_wide.reset_index()
        
        # Merge back to df
        # merge 1:1 PatientID
        df = pd.merge(df, vdf_wide, on="PatientID", how="left")
        
        # drop if _merge==2 (handled by left join)
    else:
        print(f"Warning: Verification DTA not found at {verif_path}")

    # 20. Define Populations
    # gen ITT = 1
    df["ITT"] = 1
    df.loc[df["duplicate"] == 1, "ITT"] = 0
    
    # gen MITT = 1
    df["MITT"] = 1
    df.loc[df["prereg_exclusion"] == 1, "MITT"] = 0
    
    # gen PP = 1 (Removed as vestige)
    # df["PP"] = 1 ...

    # Save Output
    output_path = os.path.join(OUTPUT_DIR, "study2_cleaned_python.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    clean_data()
