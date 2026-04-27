
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import re

# Set Paths - dynamically find Data_Analysis folder (Parent of Python_Analysis)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEIDENTIFIED_DIR = os.path.join(ROOT_DIR, "deidentified_data/level2")
INPUT_DATA = os.path.join(DEIDENTIFIED_DIR, "study2_cleaned.csv")
URINE_DATA = os.path.join(DEIDENTIFIED_DIR, "Urine_Test_Results.csv")

OUTPUT_TABLE1 = os.path.join(ROOT_DIR, "Python_Analysis/output/tbl1_TIBU_summarystats.tex")
OUTPUT_TABLE2 = os.path.join(ROOT_DIR, "Python_Analysis/output/tbl2_treatmentgroup_summary_stats.tex")
OUTPUT_TABLE3 = os.path.join(ROOT_DIR, "Python_Analysis/output/tbl3_primary_outcomes_ATEs.tex")
OUTPUT_TABLE4 = os.path.join(ROOT_DIR, "Python_Analysis/output/tbl4_medication_adherence.tex")
# --- Helper Functions ---

def verify_consort_flow(df):
    """Verify CONSORT flow diagram numbers using cleaned data flags."""
    print("\n--- CONSORT Flow Diagram Verification ---")

    raw_n = len(df)
    print(f"1. Assessed for Eligibility (Raw N): {raw_n:,}")

    # Exclusions using pre-computed flags from prepare_study_data.py
    n_duplicates = (df["duplicate"] == 1).sum()
    print(f"   - Excluded (Duplicates/Test/MT4): {n_duplicates:,}")

    # ITT: non-duplicates (keeps all outcomes including TO, NC, N/A)
    df_itt = df[df["duplicate"] == 0].copy()
    print(f"2. Randomized (ITT): {len(df_itt):,}")

    # Further exclusions from ITT to reach mITT
    n_mdto = (df_itt["treatmentoutcome"] == "TO").sum()
    print(f"   - Excluded (Transfer Out - TO): {n_mdto:,}")

    n_na = df_itt["treatmentoutcome"].isna().sum()
    print(f"   - Excluded (Missing Outcome/N/A): {n_na:,}")

    n_nc = (df_itt["treatmentoutcome"] == "NC").sum()
    print(f"   - Excluded (Not Evaluated/Consistent - NC): {n_nc:,}")

    df_mitt = df[df["MITT"] == 1].copy()
    mitt_n = len(df_mitt)
    print(f"3. Final Analytic Sample (mITT): {mitt_n:,}")

    print("   Group Breakdown:")
    counts = df_mitt["treatment_group"].value_counts().sort_index()
    print(counts)

    expected_total = 14962
    if mitt_n == expected_total:
         print("SUCCESS: CONSORT Flow matches expected mITT Total N.")
    else:
         print(f"WARNING: CONSORT Flow Mismatch! Expected {expected_total}, Got {mitt_n}")


def analyze_table2(df):
    print("\n--- Generating Table 2 (Descriptive Statistics) ---")
    
    # Filter for MITT
    df = df[df["MITT"] == 1].copy()
    print(f"MITT N: {len(df)}")
    
    group_map = {
        "Control Group": "Control",
        "SMS Reminder Group": "SMS",
        "SBCC Group": "Platform",
        "Keheala Group": "Keheala"
    }
    desired_order = ["Control", "SMS", "Platform", "Keheala"]
    
    # Variables definition (Same as analysis_table2.py)
    variables = [
        ("English", r"English (N, \%)", "binary"),
        ("male", "Male", "binary"),
        ("hiv_positive", "Living with HIV", "binary"),
        ("comorbidity_dummy", "At Least One Comorbidity", "binary"),
        ("extrapulmonary", "Extrapulmonary", "binary"),
        ("retreatment", "Retreatment", "binary"),
        ("nutritionsupport_dummy", "Nutrition Support", "binary"),
        ("bacteriologically_confirmed", "Bacteriologically Confirmed", "binary"),
        ("age_in_years", "Age in Years (Avg., Min.--Max.)", "continuous"),
        ("treatment_month_when_enrolled", "Month of Treatment at Start", "continuous"),
    ]
    
    stats = {}
    group_ns = {}
    
    for group_raw, group_label in group_map.items():
        sub = df[df["treatment_group"] == group_raw]
        group_ns[group_label] = len(sub)
        stats[group_label] = {}
        
        for var, label, vtype in variables:
            data = sub[var].dropna()
            if vtype == "binary":
                count = (data == 1).sum()
                total = len(data)
                pct = (count / total * 100) if total > 0 else 0
                stats[group_label][var] = (f"{count:,}", f"{pct:.1f}")
            elif vtype == "continuous":
                mean = data.mean()
                min_val = data.min()
                max_val = data.max()
                
                def fmt_range(mn, mx):
                    s_mn = f"{mn:.2f}".rstrip('0').rstrip('.')
                    if s_mn.startswith("0."): s_mn = s_mn[1:]
                    s_mx = f"{mx:.2f}".rstrip('0').rstrip('.')
                    return f"{s_mn}--{s_mx}"
                
                stats[group_label][var] = (f"{mean:.1f}", fmt_range(min_val, max_val))

    # LaTeX Generation
    latex = []
    latex.append(r"\scriptsize{\begin{tabular}{lrr|rr|rr|rr}")
    latex.append(r"\hline \hline & & & & & & & & \\[-8pt]")
    
    header = r"\rowcolor{yellow!15}	& "
    header_parts = []
    for g in desired_order:
        sep = "|" if g != desired_order[-1] else ""
        header_parts.append(fr"\multicolumn{{2}}{{c{sep}}}{{\textbf{{{g}}}}}")
    header += " & ".join(header_parts) + r"	 \\ \hline"
    latex.append(header)
    
    row_n = "N & "
    n_parts = []
    for g in desired_order:
        sep = "|" if g != desired_order[-1] else ""
        n_parts.append(fr"\multicolumn{{2}}{{c{sep}}}{{{group_ns[g]:,}}}")
    row_n += " & ".join(n_parts) + r"    \\"
    latex.append(row_n)
    
    for var, label, vtype in variables:
        row = f"{label} & "
        parts = []
        for g in desired_order:
            val1, val2 = stats[g][var]
            parts.append(f"{val1} & {val2}")
        row += " & ".join(parts) + r"  \\"
        latex.append(row)
        
    latex.append(r"\hline \hline")
    latex.append(r"\end{tabular}}")
    
    with open(OUTPUT_TABLE2, "w") as f:
        f.write("\n".join(latex))
    print(f"Table 2 saved to {OUTPUT_TABLE2}")



def analyze_table1(df_study_mitt):
    print("\n--- Generating Table 1 (Comparison TIBU) ---")
    
    # OUTPUTS:
    # 1. Kenya (Raw): TIBU filtered by Date & SCRN Dups ONLY. Keeps TO, NC, etc.
    # 2. Partic (Raw): TIBU filtered by Clinic. Keeps TO, NC, etc.
    # 3. Randomized (Raw): Original Study Data filtered by Dups. Keeps TO, NC, etc.
    # 4. mITT: Study Data filtered by Exclusions. (df_study_mitt)
    
    # A. Load TIBU Raw
    tibu_path = os.path.join(DEIDENTIFIED_DIR, "TIBU_firstnm_deidentified.csv")
    if not os.path.exists(tibu_path):
        print("TIBU data missing. Skipping.")
        return

    df_tibu = pd.read_csv(tibu_path, low_memory=False)

    # Filter Date (13 Apr 2018 - 20 Dec 2019)
    # Note: dateofregistration is in Stata days format (days since Jan 1, 1960)
    if 'dateofregistration' in df_tibu.columns:
        stata_epoch = pd.Timestamp("1960-01-01")
        df_tibu['date_reg'] = stata_epoch + pd.to_timedelta(df_tibu['dateofregistration'], unit='D')

    start_date = pd.Timestamp("2018-04-13")
    end_date = pd.Timestamp("2019-12-20")
    df_tibu = df_tibu[(df_tibu['date_reg'] >= start_date) & (df_tibu['date_reg'] <= end_date)].copy()
    
    # Filter SCRN Dups AND MT4 (Known Duplicates)
    # Stata: drop if regexm(scrn, "Double|Duplicate|Test") OR treatmentoutcome=="MT4"
    if "subcountyregistrationnumber" in df_tibu.columns:
        scrn_str = df_tibu["subcountyregistrationnumber"].astype(str)
        mask_regex = (
            scrn_str.str.contains("Double", case=False, na=False) |
            scrn_str.str.contains("Duplicate", case=False, na=False) |
            scrn_str.str.contains("Test", case=False, na=False)
        )
        # MT4 Mask
        mask_mt4 = pd.Series(False, index=df_tibu.index)
        if "treatmentoutcome" in df_tibu.columns:
             mask_mt4 = df_tibu["treatmentoutcome"].astype(str).str.strip().str.upper() == "MT4"
        
        mask_drop = mask_regex | mask_mt4
        df_tibu = df_tibu[~mask_drop].copy()
    
    # DEFINE POP 1: Kenya Raw
    df_kenya_raw = df_tibu.copy()
    print(f"Pop 1: Kenya Raw N={len(df_kenya_raw):,}")

    # B. Filter Participating Clinics
    if 'is_participating_clinic' in df_tibu.columns:
        df_part_raw = df_tibu[df_tibu['is_participating_clinic'] == 1].copy()
    elif 'healthfacility' in df_tibu.columns and 'healthfacility' in df_study_mitt.columns:
        # Level 1 fallback: match facility names directly
        clinics = df_study_mitt["healthfacility"].dropna().unique()
        clinics_lower = set(str(c).lower().strip() for c in clinics if str(c).upper() not in ["N/A", "NAN"])
        df_tibu['hf_lower'] = df_tibu['healthfacility'].astype(str).str.lower().str.strip()
        df_part_raw = df_tibu[df_tibu['hf_lower'].isin(clinics_lower)].copy()
    else:
        df_part_raw = pd.DataFrame()

    print(f"Pop 2: Participating Raw N={len(df_part_raw):,}")

    # C. Randomized Population (all rows except duplicates — keeps TO, NC, N/A)
    df_rand = df_study_mitt[df_study_mitt['duplicate'] == 0].copy()

    print(f"Pop 3A: Randomized N={len(df_rand):,}")
    
    # D. mITT (Passed Argument)
    # Ensure it's filtered
    df_mitt = df_study_mitt[df_study_mitt["MITT"] == 1].copy()
    print(f"Pop 3B: mITT N={len(df_mitt):,}")

    # -------------------------------------------------------------------------
    # TABLE 1 GENERATION
    # -------------------------------------------------------------------------
    # The table has 4 Columns:
    # 1. Kenya (Raw TIBU): Includes all outcomes (TO, NC, NaN) but excludes duplicates.
    # 2. Participating Clinics (Raw TIBU): Filtered by clinic list. Includes all outcomes.
    # 3. This Study (Randomized): Matches Raw Study Data. Includes all outcomes. 
    #    (Excludes Admin Duplicates: Regex 'Double|Test' AND Outcome='MT4')
    # 4. This Study (mITT): Final Analytic Sample (Excludes TO, NC, NaN, MT4).
    
    # --- Helper to deriving variables for Raw Datasets ---
    # TIBU Raw and Part Raw need 'male', 'hiv_positive', etc. derived again (same logic as before).
    # Study Raw (df_rand) needs variable mapping from Original format.
    
    # 1. Derive TIBU Vars (applied to df_kenya_raw and df_part_raw)
    def prep_tibu(df):
        if df.empty: return df
        # ... (Same logic as before) ...
        # Age
        def parse_age(x):
            x = str(x).upper()
            if 'Y' in x:
                try: return float(re.findall(r"(\d+)", x)[0])
                except: return np.nan
            elif 'M' in x:
                try: return float(re.findall(r"(\d+)", x)[0]) / 12.0
                except: return np.nan
            elif x.replace('.', '', 1).isdigit(): return float(x)
            return np.nan
        df['age_in_years'] = df['ageonregistration'].apply(parse_age)
        
        # Sex
        df['male'] = np.nan
        df.loc[df['sexmf'].astype(str).str.upper().str.startswith('M'), 'male'] = 1
        df.loc[df['sexmf'].astype(str).str.upper().str.startswith('F'), 'male'] = 0
        
        # HIV
        df['hiv_positive'] = 0
        df.loc[df['hivstatus'].astype(str).str.contains('Pos', case=False, na=False), 'hiv_positive'] = 1
        
        # Comorb
        df['comorbidity_dummy'] = 0
        df.loc[df['comorbidity'].notna() & (df['comorbidity'] != 'None') & (df['comorbidity'] != 'nan'), 'comorbidity_dummy'] = 1
        
        # Extra
        df['extrapulmonary'] = 0
        df.loc[df['typeoftbpep'].astype(str).str.strip().str.upper() == 'EP', 'extrapulmonary'] = 1
        
        # Retreatment
        df['retreatment'] = 0
        ret_codes = ['F', 'R', 'TLF', 'Relapse', 'Failure', 'Return']
        pat = '|'.join(ret_codes)
        df.loc[df['typeofpatient'].astype(str).str.contains(pat, case=False, na=False), 'retreatment'] = 1
        
        # Nutrition
        # Assuming nutritionsupport available?
        if 'nutritionsupport' in df.columns:
            df['nutritionsupport_dummy'] = 0
            # Check if value exists (not NaN) AND does not contain 'No' (case insensitive)
            has_val = df['nutritionsupport'].notna()
            is_no = df['nutritionsupport'].astype(str).str.contains('No', case=False, na=False)
            # Set to 1 only where we have a value that isn't essentially "No"
            df.loc[has_val & ~is_no, 'nutritionsupport_dummy'] = 1
        else:
             df['nutritionsupport_dummy'] = np.nan

        # Bact Confirmed
        # smear pos or geneXpert MTB
        df['bacteriologically_confirmed'] = 0
        mask_b = (
            df['sputumsmearexamination0thmon'].astype(str).str.contains('Pos', case=False, na=False) |
            df['genexpert'].astype(str).str.contains('MTB', case=False, na=False)
        )
        df.loc[mask_b, 'bacteriologically_confirmed'] = 1

        return df

    df_kenya_raw = prep_tibu(df_kenya_raw)
    df_part_raw = prep_tibu(df_part_raw)
    
    # 2. Derive OUTCOME Categories for Table 1 (MD/TO & Missing/NC)
    # Define codes for exclusion rows:
    # - MD/TO: Includes TO (Transfer Out), NTB (Not TB), and other administrative exits.
    #   *NOTE*: MT4 is excluded from this count because it is treated as a duplicate and dropped from the N.
    # - Missing/NC: Includes NC (Not Consistent), NaN (Missing), N/A, NONE.
    
    def prep_outcomes(df):
        if df.empty: return df
        # Normalize outcome column name
        target_col = None
        if 'treatmentoutcome' in df.columns: # TIBU / Cleaned
             target_col = 'treatmentoutcome'
        elif 'treatment outcome' in df.columns: # Study Raw
             target_col = 'treatment outcome'
             
        if target_col:
            val = df[target_col].astype(str).str.strip().str.upper()
            
            # TO or MT4: patients who left the DS-TB analytic cohort for a new record
            # elsewhere (TO = transferred to another facility; MT4 = moved to DR-TB
            # register). Grouped together because they are analytically equivalent.
            df['md_to_dummy'] = 0
            mask_mdto = val.isin(['TO', 'MT4', 'NTB', 'TRANSFER OUT', 'NOT TB', 'MDR-TB'])
            df.loc[mask_mdto, 'md_to_dummy'] = 1
            
            # Missing/NC
            # NC, NAN, N/A, NONE
            df['missing_nc_dummy'] = 0
            mask_miss = val.isin(['NC', 'NAN', 'N/A', 'NONE']) | df[target_col].isna()
            df.loc[mask_miss, 'missing_nc_dummy'] = 1
        else:
            df['md_to_dummy'] = 0
            df['missing_nc_dummy'] = 0
        return df

    df_kenya_raw = prep_outcomes(df_kenya_raw)
    df_part_raw = prep_outcomes(df_part_raw)
    df_rand = prep_outcomes(df_rand)
    df_mitt = prep_outcomes(df_mitt) # Should be 0 for both

    # --- Generate Table ---
    # Cols: Kenya(1), Partic(2), Rand(3A), mITT(3B)
    cols_data = [df_kenya_raw, df_part_raw, df_rand, df_mitt]
    
    metrics = [
        ("N", "N", "count"),
        ("Male (N, \\%)", "male", "binary"),
        ("Living with HIV (N, \\%)", "hiv_positive", "binary"),
        ("At Least One Comorbidity (N, \\%)", "comorbidity_dummy", "binary"),
        ("Extrapulmonary (N, \\%)", "extrapulmonary", "binary"),
        ("Retreatment (N, \\%)", "retreatment", "binary"),
        ("Nutrition Support (N, \\%)", "nutritionsupport_dummy", "binary"),
        ("Bacteriologically Confirmed (N, \\%)", "bacteriologically_confirmed", "binary"),
        ("Age in Years (Avg., Min--Max)", "age_in_years", "mean_minmax"),
        # Exclusion rows (with hline before)
        ("Transferred Out (TO or MT4) (N, \\%)", "md_to_dummy", "binary"),
        ("Missing Outcome or New Case (N, \\%)", "missing_nc_dummy", "binary"),
    ]
    
    rows = []
    rows.append(r"\scriptsize{")
    rows.append(r"\begin{tabular}{lrr|rr|rr|rr}") 
    rows.append(r"\hline \hline & & & & & & & & \\[-8pt]")
    rows.append(r"\rowcolor{yellow!15}")
    # Header 1
    rows.append(r"& \multicolumn{2}{c|}{\textbf{Kenya}} & \multicolumn{2}{c|}{\textbf{Participating Clinics}} & \multicolumn{4}{c}{\textbf{This Study}} \\")
    # Header 2 - REMOVED (All) subheadings, kept Randomized/mITT
    rows.append(r"\rowcolor{yellow!15}")
    rows.append(r"& \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{Randomized} & \multicolumn{2}{c}{mITT} \\") # Empty cells for 1/2
    rows.append(r"\hline")
    
    for label, var, mtype in metrics:
        # Add hline before the exclusion rows (MD/TO and Missing)
        if var == "md_to_dummy":
            rows.append(r"\hline")

        row_str = f"{label}"
        
        for i, df in enumerate(cols_data):
            sep = "|" if i != 3 else "" # last col no pipe
            
            if var == "N":
                val = f"{len(df):,}"
                row_str += fr" & \multicolumn{{2}}{{c{sep}}}{{{val}}}"
            else:
                 val1, val2 = "-", "-"
                 
                 # Special handling for outcome rows in mITT column (Index 3)
                 if i == 3 and var in ["md_to_dummy", "missing_nc_dummy"]:
                     val1, val2 = "0", "-" # Explicitly 0 or - as requested
                 
                 elif not df.empty and var in df.columns:
                     if mtype == "mean_minmax":
                         d = df[var].dropna()
                         if not d.empty:
                             val1 = f"{d.mean():.1f}"
                             val2 = f"{int(d.min())}--{int(d.max())}"
                     elif mtype == "binary":
                         d = df[var].dropna()
                         if not d.empty:
                              cnt = (d==1).sum()
                              if len(d) > 0:
                                  pct = (cnt / len(d)) * 100 # Use total N (valid denominator?)
                                  # Or valid N? Usually N % uses total N for binary.
                                  # d is cleaned series. len(d) is non-nulls.
                                  # But for valid %, we usually want N=Total.
                                  # Let's count 1s.
                                  # Denominator: Total N in column? 
                                  # Standard Table 1 practice: N=Total.
                                  # But duplicates/missing handling?
                                  # Let's use mean() which is 1s / (1s+0s).
                                  # If NaNs exist in var, mean() ignores them (valid %), which is correct.
                                  pct = d.mean()*100
                                  val1 = f"{cnt:,}"
                                  val2 = f"{pct:.1f}"
                                  
                                  # Special case: check if 0
                                  if cnt == 0:
                                      val1, val2 = "0", "0.0"
                 
                 row_str += f" & {val1} & {val2}"
        
        row_str += r" \\"
        rows.append(row_str)

    rows.append(r"\hline \hline")
    rows.append(r"\end{tabular}")
    rows.append(r"}")
    
    # Save to NEW Name
    # Use script directory to avoid ROOT_DIR issues (which points to Data_Analysis)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "tbl1_TIBU_summarystats.tex")
    
    with open(out_path, "w") as f:
        f.write("\n".join(rows))
    print(f"Table 1 Saved to {out_path}")


def analyze_table3(df):
    print("\n--- Generating Table 3 (Main Outcomes) ---")
    
    df = df[df["MITT"] == 1].copy()
    
    outcomes = ["unsuccessful_outcome", "LTFU", "died"]
    
    # Combined List of Controls for Fully Adjusted Models
    full_controls = [
        "C(study_month_when_enrolled)", "C(clinic_id)", 
        "C(English)", "C(male)", "age_in_years", "C(hiv_positive)", 
        "C(comorbidity_dummy)", "C(extrapulmonary)", "C(retreatment)", 
        "C(nutritionsupport_dummy)", "C(bacteriologically_confirmed)",
        "C(drugresistant)", "treatment_month_when_enrolled"
    ]
    full_controls_formula = " + ".join(full_controls)
    
    results = {}
    abs_risks = {}
    groups = ["Control Group", "SMS Reminder Group", "SBCC Group", "Keheala Group"]
    group_labels = {"Control Group": "Control", "SMS Reminder Group": "SMS", "SBCC Group": "Platform", "Keheala Group": "Keheala"}
    
    for g in groups:
        sub = df[df["treatment_group"] == g]
        abs_risks[g] = {}
        abs_risks[g]['N'] = len(sub)
        for out in outcomes:
            count = sub[out].sum()
            pct = (count / len(sub)) * 100
            abs_risks[g][out] = (count, pct)
            
    model_types = ["Unadjusted", "Partial", "Fully"]
    
    for out in outcomes:
        results[out] = {}
        print(f"Outcome: {out}")
        for mtype in model_types:
            formula = f"{out} ~ C(treatment_group, Treatment(reference='Control Group'))"
            if mtype == "Partial":
                formula += " + C(study_month_when_enrolled) + C(clinic_id)"
            elif mtype == "Fully":
                # Uses the consolidated formula
                formula += " + " + full_controls_formula
            
            try:
                model = smf.ols(formula, data=df).fit()
                # Print N for comparison
                print(f"  [DEBUG] {out} - {mtype}: N={int(model.nobs)}")
                
                results[out][mtype] = {}
                for g in ["SMS Reminder Group", "SBCC Group", "Keheala Group"]:
                    term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{g}]"
                    if term in model.params:
                        c = model.params[term]
                        ci = model.conf_int().loc[term]
                        p = model.pvalues[term]
                        se = model.bse[term]
                        
                        # Print Coeff/SE for comparison (Keheala Only for check)
                        if g == "Keheala Group":
                             print(f"    [DEBUG] {out} - {mtype}: Coeff={c:.6f}, SE={se:.6f}")

                        val = -c * 100
                        ci_lower = -ci[1] * 100
                        ci_upper = -ci[0] * 100
                        
                        results[out][mtype][group_labels[g]] = {
                            "val": val, "ci": (ci_lower, ci_upper), "p": p
                        }
            except Exception as e:
                print(f"Error {out} {mtype}: {e}")

    # LaTeX (Condensed form)
    latex = []
    latex.append(r"\scriptsize{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    latex.append(r"\begin{tabular}{p{.05in}lcc|rcc|rcc|rcc}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r"\rowcolor{yellow!15}	& 		& & \textbf{Absolute} 		& \multicolumn{9}{c}{\textbf{Reduction in Risk Relative to the Control}} \\")
    latex.append(r"\rowcolor{yellow!15}	&		& & \textbf{Risk}		& \multicolumn{3}{c|}{\textbf{Unadjusted}}  & \multicolumn{3}{c|}{\textbf{Partially Adjusted}}	& \multicolumn{3}{c}{\textbf{Fully Adjusted}}  \\")
    latex.append(r"\rowcolor{yellow!15} & &  N & \% & \% & 95\% C.I. & p-value & \% & 95\% C.I. & p-value & \% & 95\% C.I. & p-value \\ \hline")
    
    desired_order = ["Control", "SMS", "Platform", "Keheala"]
    raw_group_order = ["Control Group", "SMS Reminder Group", "SBCC Group", "Keheala Group"]
    outcome_labels = {"unsuccessful_outcome": "Unsuccessful Outcomes", "LTFU": "LTFU", "died": "Death"}
    
    for i, g_label in enumerate(desired_order):
        g_raw = raw_group_order[i]
        latex.append(fr"\multicolumn{{4}}{{l|}}{{\textbf{{{g_label} (N={abs_risks[g_raw]['N']:,})}}}}  					& &  &					& &	&				& & & 		\\")
        
        for out in ["unsuccessful_outcome", "LTFU", "died"]:
            out_lbl = outcome_labels[out]
            count, pct = abs_risks[g_raw][out]
            abs_str = f"{count} & {pct:.1f}"
            
            mod_strs = []
            if g_label == "Control":
                mod_strs = ["& & " for _ in range(3)]
            else:
                for mtype in model_types:
                    res = results[out][mtype].get(g_label)
                    if res:
                        val = res['val']
                        ci_l, ci_u = res['ci']
                        p = res['p']
                        p_str = f"{p:.3f}".lstrip('0') if p < 1 else "1.000"
                        if p < 0.001: p_str = "<.001"
                        mod_strs.append(f"{val:.1f} & ({ci_l:.1f}--{ci_u:.1f}) & {p_str}")
                    else:
                        mod_strs.append(" & & ")
            mods_latex = "& ".join(mod_strs)
            latex.append(f"& {out_lbl} & {abs_str} & {mods_latex} \\\\")
        
        if i < len(desired_order) - 1:
            latex.append(r"[1em]")
            
    latex.append(r"\hline \hline")
    latex.append(r"\end{tabular}}")
    
    with open(OUTPUT_TABLE3, "w") as f:
        f.write("\n".join(latex))
    print(f"Table 3 saved to {OUTPUT_TABLE3}")


def analyze_table4(df_clean):
    print("\n--- Generating Table 4 (Urine Verifications) ---")
    
    if not os.path.exists(URINE_DATA):
        print("Urine Data not found. Skipping Table 4.")
        return

    df_urine = pd.read_csv(URINE_DATA)
    # Handle BOM in column names
    df_urine.columns = [c.lstrip('\ufeff') for c in df_urine.columns]
    # Clean Urine - handle both original and de-identified column names
    if "Patient ID" in df_urine.columns:
        df_urine = df_urine.dropna(subset=["Patient ID"])
        df_urine = df_urine.drop_duplicates(subset=["Patient ID"], keep="first")
        df_urine = df_urine.rename(columns={"Patient ID": "PatientID"})
    elif "anon_patient_id" in df_urine.columns:
        df_urine = df_urine.dropna(subset=["anon_patient_id"])
        df_urine = df_urine.drop_duplicates(subset=["anon_patient_id"], keep="first")
        df_urine = df_urine.rename(columns={"anon_patient_id": "PatientID"})
    # Stata Logic: Uses 'group' from Urine Data for assignment, but 'treatment_group' filtering from Main?
    # To match N=744 (C=203, K=541):
    # Sample = Merge(Inner) -> Filter Main Group in {C, K} -> Filter Outcome not (NC, NaN).
    # Group Assignment = Urine Group.
    
    if 'group' in df_urine.columns:
        df_urine = df_urine.rename(columns={'group': 'urine_group'})

    # Fix Column Names if needed
    if "PatientID" not in df_urine.columns and "Patient ID" in df_urine.columns:
        df_urine = df_urine.rename(columns={"Patient ID": "PatientID"})
    if "PatientID" not in df_urine.columns and "patient_id" in df_urine.columns:
        df_urine = df_urine.rename(columns={"patient_id": "PatientID"})
    
    df = pd.merge(df_urine, df_clean, on="PatientID", how="inner")
    
    # Debug N Counts Details
    print(f"Table 4 Raw Merge N: {len(df)}")
    
    # Ensure TO exists
    if "TO" not in df.columns:
        if "treatmentoutcome" in df.columns:
            df["TO"] = (df["treatmentoutcome"] == "TO").astype(int)
        else:
            print("Warning: TO and treatmentoutcome missing. Assuming TO=0.")
            df["TO"] = 0
    
    # Apply Filters (Rigorous Method: N=731)
    # logic: Use Main Data 'treatment_group', Filter Prereg Exclusion & TO.
    # See 'table4_n_mismatch_explanation.md' for details on Stata N=744 diff.
    
    # 1. Filter by Main Treatment Group
    df_sub = df[df["treatment_group"].isin(["Control Group", "Keheala Group"])].copy()
    
    # 2. Strict Exclusions
    # Use prereg_exclusion (includes Dup, Test, MT4, etc.) and TO
    df_sub = df_sub[
        (df_sub["prereg_exclusion"] == 0) & 
        (df_sub["TO"] == 0)
    ].copy()
    
    # 3. Treatment Group for Analysis
    # Use the verified Main Data group
    df_sub['treatment_group_analysis'] = df_sub['treatment_group']
    
    print(f"Table 4 Adjusted N (Rigorous): {len(df_sub)}")
    
    # Counts Debug
    print("  Analysis Groups Counts:")
    print(df_sub['treatment_group_analysis'].value_counts())

    # Outcome
    df_sub["non_adherence"] = 0
    df_sub.loc[df_sub["Results Interpretation"] == "Poor Adherence", "non_adherence"] = 1
    
    model_types = ["Unadjusted", "Partial", "Fully"]
    results = {}
    
    full_controls = [
        "C(study_month_when_enrolled)", "C(clinic_id)", 
        "C(English)", "C(male)", "age_in_years", "C(hiv_positive)", 
        "C(comorbidity_dummy)", "C(extrapulmonary)", "C(retreatment)", 
        "C(nutritionsupport_dummy)", "C(bacteriologically_confirmed)",
        "C(drugresistant)", "treatment_month_when_enrolled"
    ]
    full_controls_formula = " + ".join(full_controls)
    
    for mtype in model_types:
        # Use treatment_group_analysis
        formula = "non_adherence ~ C(treatment_group_analysis, Treatment(reference='Control Group'))"
        
        if mtype == "Partial":
            formula += " + C(study_month_when_enrolled) + C(clinic_id)"
        elif mtype == "Fully":
            formula += " + " + full_controls_formula
            
        try:
            # Handle Missing Data for Clustering/Fully
            vars_to_check = ["non_adherence", "treatment_group_analysis", "clinic_id"]
            if mtype == "Partial":
                vars_to_check += ["study_month_when_enrolled"]
            elif mtype == "Fully":
                vars_to_check += full_controls
            
            clean_vars = set()
            import re
            for v in vars_to_check:
                v_clean = re.sub(r'C\((.*?)\)', r'\1', v)
                if "treatment_group" in v: v_clean = "treatment_group_analysis" 
                clean_vars.add(v_clean)
                
            # Note: treatment_group_analysis needs to be in df
            
            valid_cols = [c for c in clean_vars if c in df_sub.columns]
            df_curr = df_sub.dropna(subset=valid_cols).copy()
            
            model = smf.ols(formula, data=df_curr).fit(cov_type='cluster', cov_kwds={'groups': df_curr['clinic_id']})
            
            # Print for Comparison
            print(f"  [DEBUG] {mtype}: N={int(model.nobs)}")
            
            term = "C(treatment_group_analysis, Treatment(reference='Control Group'))[T.Keheala Group]"
            if term in model.params:
                c = model.params[term]
                ci = model.conf_int().loc[term]
                p = model.pvalues[term]
                
                # Invert for "Reduction" (if outcome is "non_adherence", neg coef = reduction)
                val = -c * 100
                ci_l = -ci[1] * 100
                ci_u = -ci[0] * 100
                
                results[mtype] = {"val": val, "ci": (ci_l, ci_u), "p": p}
        except Exception as e:
            print(f"Error Table 4 {mtype}: {e}")

    # LaTeX Generation
    latex = []
    latex.append(r"\scriptsize{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    latex.append(r"\begin{tabular}{lrc|rcc|rcc|rcc}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r"\rowcolor{yellow!15}				 	&	& \textbf{Absolute} 		& \multicolumn{9}{c}{\textbf{Reduction in Risk Relative to the Control}} \\")
    latex.append(r"\rowcolor{yellow!15}					&	& \textbf{Risk}		& \multicolumn{3}{c}{\textbf{Unadjusted}} & \multicolumn{3}{c}{\textbf{Partially Adjusted}}	& \multicolumn{3}{c}{\textbf{Fully Adjusted}}  \\")
    latex.append(r"\rowcolor{yellow!15}	  & N &  \% & \% & 95\% C.I. & p-value & \% & 95\% C.I. & p-value & \% & 95\% C.I. & p-value \\ \hline")
    
    # Calculate Abs Risk
    abs_risks = {}
    for g in ["Control Group", "Keheala Group"]:
        sub = df_sub[df_sub["treatment_group_analysis"] == g]
        cnt = sub["non_adherence"].sum()
        pct = sub["non_adherence"].mean() * 100
        abs_risks[g] = (len(sub), pct)

    # Control Row
    n_c, pct_c = abs_risks["Control Group"]
    latex.append(fr"\textbf{{Control (N={n_c})}} & {int(n_c * pct_c/100)}  & {pct_c:.1f}  &    & & & & & &  \\")
    
    # Keheala Row
    n_k, pct_k = abs_risks["Keheala Group"]
    res_str = ""
    # No term finding here, just iteration
    
    for mtype in ["Unadjusted", "Partial", "Fully"]:
        res_dict = results.get(mtype)
        if res_dict:
            val = res_dict['val']
            l, u = res_dict['ci']
            p = res_dict['p']
            
            p_str = f"{p:.3f}".lstrip('0') if p < 1 else "1.000"
            if p < 0.001: p_str = "<.001"
            
            sep = "&" if mtype != "Fully" else ""
            res_str += fr"{val:.1f} & ({l:.1f}--{u:.1f}) & {p_str} {sep} "
        else:
            sep = "&" if mtype != "Fully" else ""
            res_str += f" & & {sep} "
            
    latex.append(fr"\textbf{{Keheala (N={n_k})}} & {int(n_k * pct_k/100)} &  {pct_k:.1f}  & {res_str} \\")
    
    latex.append(r"\hline \hline")
    latex.append(r"\end{tabular}}")
    
    with open(OUTPUT_TABLE4, "w") as f:
        f.write("\n".join(latex))
    print(f"Table 4 saved to {OUTPUT_TABLE4}")



def analyze_verification_rates(df):
    print("\n--- Generating Verification Rates Text ---")

    sub = df[(df["treatment_group"].isin(["SBCC Group", "Keheala Group"])) & (df["MITT"] == 1)].copy()
    print(f"mITT subset N: {len(sub)}")

    means = sub.groupby("treatment_group")["compliancescore"].mean()
    n_platform = int((sub["treatment_group"] == "SBCC Group").sum())
    n_keheala = int((sub["treatment_group"] == "Keheala Group").sum())
    mean_platform = means.get("SBCC Group", 0) * 100
    mean_keheala = means.get("Keheala Group", 0) * 100

    print(f"Platform (mITT) N: {n_platform}, Mean: {mean_platform:.1f}%")
    print(f"Keheala  (mITT) N: {n_keheala}, Mean: {mean_keheala:.1f}%")

    formula = "compliancescore ~ C(treatment_group, Treatment(reference='SBCC Group'))"

    model = smf.ols(formula, data=sub).fit()

    term = "C(treatment_group, Treatment(reference='SBCC Group'))[T.Keheala Group]"

    if term in model.params:
        diff_val = model.params[term] * 100
        ci = model.conf_int().loc[term] * 100
        p = model.pvalues[term]

        p_str = f"{p:.3f}".lstrip('0')
        if p < 0.001: p_str = "<.001"
        else: p_str = f"p={p_str}"

        text = (
            f"Verification rates were {mean_platform:.1f}\\% on average for individuals in the platform group "
            f"(N={n_platform:,}) and {mean_keheala:.1f}\\% in the Keheala group (N={n_keheala:,}), "
            f"a difference of {diff_val:.1f} percentage points "
            f"(95\\% C.I.: {ci[0]:.1f}-{ci[1]:.1f}; {p_str})."
        )

        print("\nGenerated Text:")
        print(text)

        out_path = os.path.join(ROOT_DIR, "Python_Analysis/output/verificationrates.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
        print(f"Saved to {out_path}")
    else:
        print("Error: Could not find regression term for difference.")

def main():
    if not os.path.exists(INPUT_DATA):
        print(f"Error: {INPUT_DATA} not found. Run prepare_data.py first.")
        return

    print("Loading Cleaned Data...")
    df = pd.read_csv(INPUT_DATA, low_memory=False)

    # Handle de-identified column names (anon_patient_id -> PatientID, anon_scrn -> scrn)
    if "anon_patient_id" in df.columns and "PatientID" not in df.columns:
        df = df.rename(columns={"anon_patient_id": "PatientID"})
    if "anon_scrn" in df.columns and "scrn" not in df.columns:
        df = df.rename(columns={"anon_scrn": "scrn"})

    # Run Analyses
    verify_consort_flow(df)
    analyze_table1(df)
    analyze_table2(df)
    analyze_table3(df)
    analyze_table4(df)
    analyze_verification_rates(df)

if __name__ == "__main__":
    main()
