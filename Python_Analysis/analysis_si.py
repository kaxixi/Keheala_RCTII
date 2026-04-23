
"""
Keheala Study 2 - Supplementary Information Analysis Script
===========================================================

Purpose:
    Replicates the Supplementary Information (SI) tables (SI5 - SI13) for the Keheala Study 2 RCT.
    
    Sensitivity Analyses:
    - SI7: Missing outcomes (NA) excluded.
    - SI8: Transfer Out (TO) and Missing outcomes (NA) excluded.
    - SI7+SI8: Combined Side-by-Side comparison of these sensitivities.
    
    Subsets:
    - SI9: Bacteriologically Confirmed cases only.
    - SI10: Pulmonary TB cases only (Extrapulmonary == 0).
    - SI11: New cases only (Retreatment == 0).
    
    Time Splits:
    - SI12: Unsuccessful Outcome split by time to enrollment (<=1 mo vs >1 mo).
    - SI13: LTFU split by time to enrollment.
    
    Other:
    - SI5: Cost Effectiveness (Intervention comparison).
    - SI6: Exclusion counts and Missing Data.

Input:
    - deidentified_data/level2/study2_cleaned.csv

Output:
    - output/tblSI_*.tex

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
import os
import re

# Constants - dynamically find Data_Analysis folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEIDENTIFIED_DIR = os.path.join(ROOT_DIR, "deidentified_data/level2")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Python_Analysis/output")
INPUT_DATA_CLEAN = os.path.join(DEIDENTIFIED_DIR, "study2_cleaned.csv")

def preprocess_data(df):
    """
    Pre-processes cleaned data for SI analysis specifics.
    Recreates specific exclusion flags and outcome definitions used in SI tables.
    """
    # Exclusions (Specific breakdowns for SI6/MDTO)
    # Note: 'treatmentoutcome' is already cleaned/normalized in input
    
    df['MD'] = (df['treatmentoutcome'] == "MT4").astype(int)
    df['TO'] = (df['treatmentoutcome'] == "TO").astype(int)
    df['NA'] = ((df['treatmentoutcome'] == "N/A") | (df['treatmentoutcome'].isna())).astype(int)
    
    # Outcomes
    # LTFU_outcome (Variable used in SI5, SI9-13)
    # Note: cleaned data has 'LTFU' column, but SI scripts use 'LTFU_outcome'
    df['LTFU_outcome'] = 0
    df.loc[df['treatmentoutcome'] == 'LTFU', 'LTFU_outcome'] = 1
    
    # Ensure clinic_id exists (should be in cleaned data)
    if 'clinic_id' not in df.columns:
        print("Warning: clinic_id not found in cleaned data. Attempting to recreate.")
        if 'syssubcounty' in df.columns and 'clinic' in df.columns:
            df['clinic_str'] = df['syssubcounty'].astype(str) + "_" + df['clinic'].astype(str)
            df['clinic_id'] = df['clinic_str'].astype('category').cat.codes
            
    return df

def run_ols(df, formula, label):
    # Extract potential columns
    terms = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula)
    cols = [c for c in terms if c in df.columns]
    if 'clinic_id' in df.columns and 'clinic_id' not in cols:
        cols.append('clinic_id')
        
    df_reg = df.dropna(subset=cols).copy()
    
    model = smf.ols(formula, data=df_reg).fit(cov_type='cluster', cov_kwds={'groups': df_reg['clinic_id']})
    
    try:
        coef = model.params["C(treatment_group, Treatment(reference='Control Group'))[T.Keheala Group]"]
        se = model.bse["C(treatment_group, Treatment(reference='Control Group'))[T.Keheala Group]"]
        pval = model.pvalues["C(treatment_group, Treatment(reference='Control Group'))[T.Keheala Group]"]
        n = model.nobs
        print(f"[{label}] N={int(n)}, Keheala Coef={coef:.4f}, SE={se:.4f}, P={pval:.4f}")
    except:
        print(f"[{label}] Keheala coef not found (Check levels: {df_reg['treatment_group'].unique()})")

    return model

def analyze_table_si5(df):
    """
    Generates Table SI5: Comparison of Interventions (Unsuccessful Outcomes, LTFU).
    """
    print("\n=== Generating Table SI5 (Comparison of Interventions) ===")
    
    outcomes = [
        ("Unsuccessful Outcomes", "unsuccessful_outcome"),
        ("LTFU", "LTFU_outcome")
    ]
    
    comparisons = [
        ("SMS", "SMS Reminder Group"),
        ("Platform", "SBCC Group")
    ]
    
    results = {outcome_label: {} for outcome_label, _ in outcomes}
    
    for outcome_label, outcome_var in outcomes:
        for comp_label, ref_group in comparisons:
            formula = f"{outcome_var} ~ C(treatment_group, Treatment(reference='{ref_group}'))"
            try:
                mitt = df[df['prereg_exclusion'] == 0].copy()
                model = smf.ols(formula=formula, data=mitt).fit()
                term = f"C(treatment_group, Treatment(reference='{ref_group}'))[T.Keheala Group]"
                if term in model.params:
                    c = model.params[term]
                    ci = model.conf_int().loc[term]
                    p = model.pvalues[term]
                    val = -c * 100
                    ci_l = -ci[1] * 100
                    ci_u = -ci[0] * 100
                    results[outcome_label][comp_label] = (val, ci_l, ci_u, p)
                    print(f"  [SI5] {outcome_label} vs {comp_label}: Red={val:.2f}% p={p:.3f}")
            except Exception as e:
                print(f"  Error SI5 {outcome_label} {comp_label}: {e}")

    latex = []
    latex.append(r"\scriptsize{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    latex.append(r"\begin{tabular}{l|rcc|rcc}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r"\rowcolor{yellow!15}	& \multicolumn{6}{c}{\textbf{Reduction in Risk}} \\")
    latex.append(r"\rowcolor{yellow!15}	& \multicolumn{3}{c|}{\textbf{Relative to SMS}}  & \multicolumn{3}{c}{\textbf{Relative to Platform}}  \\")
    latex.append(r"\rowcolor{yellow!15} & \% & 95\% C.I. & p-value & \% & 95\% C.I. & p-value \\ \hline")
    
    for outcome_label, _ in outcomes:
        row = f"{outcome_label} & "
        if "SMS" in results[outcome_label]:
            val, l, u, p = results[outcome_label]["SMS"]
            p_str = f"{p:.3f}".lstrip('0') if p < 1 else "1.000"
            if p < 0.001: p_str = "<.001"
            row += f"{val:.1f} & {l:.1f}--{u:.1f} & {p_str} & "
        else:
            row += " & & & "
        if "Platform" in results[outcome_label]:
            val, l, u, p = results[outcome_label]["Platform"]
            p_str = f"{p:.3f}".lstrip('0') if p < 1 else "1.000"
            if p < 0.001: p_str = "<.001"
            row += f"{val:.1f} & {l:.1f}--{u:.1f} & {p_str} \\\\"
        else:
            row += " & & \\\\"
        latex.append(row)
        
    latex.append(r"\hline \hline")
    latex.append(r"\end{tabular}}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_comparison_of_interventions.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_comparison_of_interventions.tex")

def analyze_table_si6(df):
    """
    Generates Table SI6: NC Counts (and Rates).
    """
    print("\n=== Generating Table SI6 (NC Counts) ===")
    
    sub = df[df["duplicate"] == 0].copy()
    total_n = sub["treatment_group"].value_counts()
    
    nc_df = sub[sub["treatmentoutcome"] == "NC"]
    nc_counts = nc_df["treatment_group"].value_counts()
    
    groups = ["Control Group", "SMS Reminder Group", "SBCC Group", "Keheala Group"]
    labels = {"Control Group": "Control", "SMS Reminder Group": "SMS", "SBCC Group": "Platform", "Keheala Group": "Keheala"}
    
    latex = []
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\hline")
    latex.append(r"Treatment Group & Total N & NC Count & \% NC \\")
    latex.append(r"\hline")
    
    for g in groups:
        n = total_n.get(g, 0)
        nc = nc_counts.get(g, 0)
        pct = (nc / n * 100) if n > 0 else 0
        latex.append(f"{labels[g]} & {n} & {nc} & {pct:.1f}\% \\\\")
        
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    
    with open(os.path.join(OUTPUT_DIR, "tbl6_NC_counts.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tbl6_NC_counts.tex")

def analyze_mdto_table(df):
    """
    Generates Table SI6 (tblSI_MDTO.tex): post-randomization exclusion
    categories (TO, MT4, N/A, NC) by intervention group.  Shows counts
    and percentages of the randomized denominator.
    """
    print("\n=== Generating Table SI6 (post-randomization exclusions) ===")

    # Randomized = all records except SCRN-flagged duplicates.
    # MT4 records ARE randomized (they received treatment assignment); they are
    # post-randomization exclusions grouped with TO in the main narrative.
    rand = df[df['duplicate'] == 0].copy()

    groups = ["Keheala Group", "SBCC Group", "SMS Reminder Group", "Control Group"]
    labels = {"Keheala Group": "Keheala", "SBCC Group": "Platform",
              "SMS Reminder Group": "SMS", "Control Group": "Control"}

    def counts(sub):
        na = sub['treatmentoutcome'].isna().sum()
        nc = (sub['treatmentoutcome'] == 'NC').sum()
        return {
            "TO":  (sub['treatmentoutcome'] == 'TO').sum(),
            "MT4": (sub['treatmentoutcome'] == 'MT4').sum(),
            "N/A": na,
            "NC":  nc,
            "Missing": na + nc,
        }

    latex = [
        r"\scriptsize{\begin{tabular}{lccccc}",
        r"\hline \hline \\[-8pt]",
        r"\rowcolor{yellow!15}            & TO             & MT4            & N/A or Blank   & NC             & Missing (Total) \\",
        r"\rowcolor{yellow!15}            & n (\%)         & n (\%)         & n (\%)         & n (\%)         & n (\%)         \\ \hline \\[-8pt]",
    ]

    col_order = ["TO", "MT4", "N/A", "NC", "Missing"]

    for i, g in enumerate(groups):
        grp = rand[rand['treatment_group'] == g]
        n = len(grp)
        c = counts(grp)
        row = f"{labels[g]:<10s} "
        for m in col_order:
            cnt = c[m]
            pct = (cnt / n * 100) if n > 0 else 0
            row += f"& {cnt:,} ({pct:.1f})   "
        row += r"\\"
        latex.append(row)
        if i < len(groups) - 1:
            latex.append(r"[0.4em]")

    latex.append(r"\hline")
    n_total = len(rand)
    c_total = counts(rand)
    row = "Total      "
    for m in col_order:
        cnt = c_total[m]
        pct = (cnt / n_total * 100) if n_total > 0 else 0
        row += f"& {cnt:,} ({pct:.1f})   "
    row += r"\\ \hline \hline"
    latex.append(row)
    latex.append(r"\end{tabular}}")

    # Chi-squared test: does the proportion of missing outcomes (N/A + NC)
    # differ across intervention arms?  Cited in Results alongside Table MDTO.
    from scipy.stats import chi2_contingency
    contingency = []
    for g in groups:
        grp = rand[rand['treatment_group'] == g]
        c = counts(grp)
        contingency.append([c["Missing"], len(grp) - c["Missing"]])
    chi2, p_missing, dof, _ = chi2_contingency(contingency)
    log_lines = [
        "--- Missing-outcome rate across intervention arms (N/A + NC) ---",
        "",
        "Per-arm missing totals (N/A + NC) and denominators:",
    ]
    for g, row in zip(groups, contingency):
        log_lines.append(f"  {labels[g]:<10s} missing = {row[0]:>5d} / {row[0]+row[1]:>5d} "
                         f"({row[0]/(row[0]+row[1])*100:.1f}%)")
    log_lines += [
        "",
        f"Chi-squared test of independence (2x{len(groups)} table):",
        f"  chi2({dof}) = {chi2:.3f}, p = {p_missing:.3f}",
    ]
    log_text = "\n".join(log_lines) + "\n"
    with open(os.path.join(OUTPUT_DIR, "missing_outcome_by_arm.log"), "w") as f:
        f.write(log_text)
    print(log_text)

    with open(os.path.join(OUTPUT_DIR, "tblSI_MDTO.tex"), "w") as f:
        f.write("\n".join(latex))
    print("Saved tblSI_MDTO.tex")

def analyze_sensitivity_outcome_coding(df):
    """
    Generates Table SI7 (tblSI_missing.tex): a 9-column sensitivity
    table showing how the primary mITT estimates change under alternative
    treatments of records with missing or non-standard outcomes.

      - Cols 1-3: Primary mITT (MD/TO and Missing dropped from sample).
                  Reproduces the unsuccessful-outcome estimates in Table 3.
      - Cols 4-6: Missing (N/A + NC) coded as unsuccessful; MD/TO dropped.
      - Cols 7-9: MD/TO (MT4 + TO) and Missing both coded as unsuccessful.

    Each block uses the same three specifications: unadjusted (1,4,7);
    + clinic & month dummies (2,5,8); + individual characteristics (3,6,9).
    """
    print("\n=== Generating Sensitivity Table (9 cols) ===")

    specs = [
        ("", ""),
        (" + treatment_month_when_enrolled + C(study_month_when_enrolled)", " + month"),
        (" + treatment_month_when_enrolled + C(study_month_when_enrolled) + age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(extrapulmonary) + C(retreatment) + C(nutritionsupport_dummy) + C(bacteriologically_confirmed) + C(drugresistant)", " + controls"),
    ]

    # Block 1 — Primary mITT: drops MD/TO and Missing from sample; outcome unchanged.
    sub_primary = df[df['MITT'] == 1].copy()
    sub_primary['outcome_primary'] = sub_primary['unsuccessful_outcome']
    models_primary = [
        run_ols(sub_primary,
                f"outcome_primary ~ C(treatment_group, Treatment(reference='Control Group')){spec}",
                f"Primary({i+1}){label}")
        for i, (spec, label) in enumerate(specs)
    ]

    # Block 2 — Missing coded as unsuccessful; MD/TO still dropped.
    sub_missing = df[df['duplicate'] == 0].copy()
    sub_missing = sub_missing[~sub_missing['treatmentoutcome'].isin(['MT4', 'TO'])]
    sub_missing['outcome_missing'] = sub_missing['unsuccessful_outcome']
    sub_missing.loc[
        sub_missing['treatmentoutcome'].isin(['N/A', 'NC']) | sub_missing['treatmentoutcome'].isna(),
        'outcome_missing'
    ] = 1
    models_missing = [
        run_ols(sub_missing,
                f"outcome_missing ~ C(treatment_group, Treatment(reference='Control Group')){spec}",
                f"Missing({i+1}){label}")
        for i, (spec, label) in enumerate(specs)
    ]

    # Block 3 — MD/TO and Missing both coded as unsuccessful.
    sub_mdto = df[df['duplicate'] == 0].copy()
    sub_mdto['outcome_mdto'] = sub_mdto['unsuccessful_outcome']
    sub_mdto.loc[
        sub_mdto['treatmentoutcome'].isin(['TO', 'N/A', 'NC', 'MT4']) | sub_mdto['treatmentoutcome'].isna(),
        'outcome_mdto'
    ] = 1
    models_mdto = [
        run_ols(sub_mdto,
                f"outcome_mdto ~ C(treatment_group, Treatment(reference='Control Group')){spec}",
                f"MDTO({i+1}){label}")
        for i, (spec, label) in enumerate(specs)
    ]

    all_models = models_primary + models_missing + models_mdto

    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in all_models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells

    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{9}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{\shortstack{mITT \\ Analysis}}} & \multicolumn{3}{c}{\textbf{\shortstack{Missing coded \\ as unsuccessful}}} & \multicolumn{3}{c}{\textbf{\shortstack{MD/TO and Missing \\ coded as unsuccessful}}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}&\multicolumn{1}{c}{(7)}&\multicolumn{1}{c}{(8)}&\multicolumn{1}{c}{(9)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in all_models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")

    # Manuscript references this table via \input{tblSI_missing.tex}.
    with open(os.path.join(OUTPUT_DIR, "tblSI_missing.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_missing.tex")

def analyze_si9(df):
    """
    Generates Table SI9: Regressions for Bacteriologically Confirmed (tblSI_bacteriologically_confirmed.tex).
    Side-by-side: Unsuccessful Outcome (Cols 1-3) | LTFU (Cols 4-6).
    """
    print("\n=== Generating Table SI9 (Bacteriologically Confirmed) ===")
    
    sub = df[(df['prereg_exclusion'] == 0) & (df['bacteriologically_confirmed'] == 1)].copy()
    time_controls = "treatment_month_when_enrolled + C(study_month_when_enrolled)"
    full_controls = "age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(extrapulmonary) + C(retreatment) + C(nutritionsupport_dummy) + C(drugresistant)"
    
    models = []
    models.append(run_ols(sub, "unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI9 UO(1)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI9 UO(2)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI9 UO(3)"))
    
    models.append(run_ols(sub, "LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI9 LTFU(4)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI9 LTFU(5)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI9 LTFU(6)"))
    
    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells
        
    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{6}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{Unsuccessful Outcome}} & \multicolumn{3}{c}{\textbf{LTFU}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_bacteriologically_confirmed.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_bacteriologically_confirmed.tex")

def analyze_si10(df):
    """
    Generates Table SI10: Regressions for Pulmonary TB (tblSI_pulmonary.tex).
    """
    print("\n=== Generating Table SI10 (Pulmonary TB) ===")
    
    sub = df[(df['prereg_exclusion'] == 0) & (df['extrapulmonary'] == 0)].copy()
    time_controls = "treatment_month_when_enrolled + C(study_month_when_enrolled)"
    full_controls = "age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(retreatment) + C(nutritionsupport_dummy) + C(bacteriologically_confirmed) + C(drugresistant)"
    
    models = []
    models.append(run_ols(sub, "unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI10 UO(1)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI10 UO(2)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI10 UO(3)"))
    models.append(run_ols(sub, "LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI10 LTFU(4)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI10 LTFU(5)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI10 LTFU(6)"))
    
    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells
        
    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{6}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{Unsuccessful Outcome}} & \multicolumn{3}{c}{\textbf{LTFU}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_pulmonary.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_pulmonary.tex")
    
def analyze_si11(df):
    """
    Generates Table SI11: Regressions for New/Relapse Patients (tblSI_not_retreatment.tex).
    """
    print("\n=== Generating Table SI11 (Not Retreatment) ===")
    
    sub = df[(df['prereg_exclusion'] == 0) & (df['retreatment'] == 0)].copy()
    time_controls = "treatment_month_when_enrolled + C(study_month_when_enrolled)"
    full_controls = "age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(extrapulmonary) + C(nutritionsupport_dummy) + C(bacteriologically_confirmed) + C(drugresistant)"
    
    models = []
    models.append(run_ols(sub, "unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI11 UO(1)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI11 UO(2)"))
    models.append(run_ols(sub, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI11 UO(3)"))
    models.append(run_ols(sub, "LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI11 LTFU(4)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI11 LTFU(5)"))
    models.append(run_ols(sub, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI11 LTFU(6)"))
    
    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells
        
    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{6}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{Unsuccessful Outcome}} & \multicolumn{3}{c}{\textbf{LTFU}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_not_retreatment.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_not_retreatment.tex")
    
def analyze_si12_uo(df):
    """
    Generates Table SI12: Regressions of Unsuccessful Outcome by Time to Treatment (tblSI_time_UO.tex).
    Side-by-side: <= 1 month (Cols 1-3) | > 1 month (Cols 4-6).
    """
    print("\n=== Generating Table SI12 (Time UO) ===")
    
    mitt = df[df['prereg_exclusion'] == 0].copy()
    early = mitt[mitt['treatment_month_when_enrolled'] <= 1].copy()
    late = mitt[mitt['treatment_month_when_enrolled'] > 1].copy()
    
    time_controls = "treatment_month_when_enrolled + C(study_month_when_enrolled)"
    full_controls = "age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(extrapulmonary) + C(retreatment) + C(nutritionsupport_dummy) + C(bacteriologically_confirmed) + C(drugresistant)"
    
    models = []
    # Early
    models.append(run_ols(early, "unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI12 Early(1)"))
    models.append(run_ols(early, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI12 Early(2)"))
    models.append(run_ols(early, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI12 Early(3)"))
    # Late
    models.append(run_ols(late, "unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI12 Late(4)"))
    models.append(run_ols(late, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI12 Late(5)"))
    models.append(run_ols(late, f"unsuccessful_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI12 Late(6)"))
    
    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells
        
    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{6}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{$\leq$1 month}} & \multicolumn{3}{c}{\textbf{$>$1 month}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_time_UO.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_time_UO.tex")

def analyze_si13_ltfu(df):
    """
    Generates Table SI13: Regressions of LTFU Outcome by Time to Treatment (tblSI_time_LTFU.tex).
    Side-by-side: <= 1 month (Cols 1-3) | > 1 month (Cols 4-6).
    """
    print("\n=== Generating Table SI13 (Time LTFU) ===")
    
    mitt = df[df['prereg_exclusion'] == 0].copy()
    early = mitt[mitt['treatment_month_when_enrolled'] <= 1].copy()
    late = mitt[mitt['treatment_month_when_enrolled'] > 1].copy()
    
    time_controls = "treatment_month_when_enrolled + C(study_month_when_enrolled)"
    full_controls = "age_in_years + C(male) + C(English) + C(hiv_positive) + C(comorbidity_dummy) + C(extrapulmonary) + C(retreatment) + C(nutritionsupport_dummy) + C(bacteriologically_confirmed) + C(drugresistant)"
    
    models = []
    # Early
    models.append(run_ols(early, "LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI13 Early(1)"))
    models.append(run_ols(early, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI13 Early(2)"))
    models.append(run_ols(early, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI13 Early(3)"))
    # Late
    models.append(run_ols(late, "LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group'))", "SI13 Late(4)"))
    models.append(run_ols(late, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls}", "SI13 Late(5)"))
    models.append(run_ols(late, f"LTFU_outcome ~ C(treatment_group, Treatment(reference='Control Group')) + {time_controls} + {full_controls}", "SI13 Late(6)"))
    
    rows_data = {}
    for label, group in [("Keheala", "Keheala Group"), ("Platform", "SBCC Group"), ("SMS", "SMS Reminder Group")]:
        row_cells = []
        term = f"C(treatment_group, Treatment(reference='Control Group'))[T.{group}]"
        for m in models:
            if term in m.params:
                coef = m.params[term]
                se = m.bse[term]
                pval = m.pvalues[term]
                star = ""
                if pval < 0.001: star = r"\sym{***}"
                elif pval < 0.01: star = r"\sym{**}"
                elif pval < 0.05: star = r"\sym{*}"
                row_cells.append((f"{coef*100:.1f}{star}", f"({se*100:.1f})"))
            else:
                row_cells.append(("", ""))
        rows_data[label] = row_cells
        
    latex = []
    latex.append(r"\scriptsize{")
    latex.append(r"\begin{tabular}{l*{6}{c}}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & \multicolumn{3}{c}{\textbf{$\leq$1 month}} & \multicolumn{3}{c}{\textbf{$>$1 month}} \\")
    latex.append(r"\rowcolor{yellow!15}            &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}&\multicolumn{1}{c}{(5)}&\multicolumn{1}{c}{(6)}\\")
    latex.append(r"\hline \\[-8pt]")
    for label in ["Keheala", "Platform", "SMS"]:
        line = f"{label} & " + " & ".join([c[0] for c in rows_data[label]]) + r" \\"
        latex.append(line)
        line = "            &   " + " &   ".join([c[1] for c in rows_data[label]]) + r"         \\"
        latex.append(line)
        if label != "SMS": latex.append(r"[0.4em]")
    latex.append(r"[0.4em]")
    latex.append(r"\hline")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"Clinic \& Month Dummies & & Yes & Yes & & Yes & Yes \\")
    latex.append(r"Individual Chars. & & & Yes & & & Yes \\")
    line = r"\(N\)       "
    for m in models:
        line += f"&       {int(m.nobs)}         "
    line += r"\\"
    latex.append(line)
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"}")
    
    with open(os.path.join(OUTPUT_DIR, "tblSI_time_LTFU.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Saved tblSI_time_LTFU.tex")

def analyze_time_unadjusted_proportions(df):
    """
    Computes unadjusted proportions tests for the first-month subsample.
    This produces the numbers cited in manuscript lines 216 and 529:
    - Raw percentages of unsuccessful outcomes by treatment group
    - Two-sample Z-test comparing each intervention to control
    - 95% CIs and p-values
    """
    print("\n=== Computing Unadjusted Proportions for First-Month Subsample ===")

    mitt = df[df['prereg_exclusion'] == 0].copy()
    early = mitt[mitt['treatment_month_when_enrolled'] <= 1].copy()

    groups = {
        'Control': 'Control Group',
        'SMS': 'SMS Reminder Group',
        'Platform': 'SBCC Group',
        'Keheala': 'Keheala Group'
    }

    # Calculate raw percentages
    print("\n--- Unsuccessful Outcomes (First Month Subsample) ---")
    results = {}
    for label, group_name in groups.items():
        grp = early[early['treatment_group'] == group_name]
        n = len(grp)
        n_uo = grp['unsuccessful_outcome'].sum()
        pct = (n_uo / n * 100) if n > 0 else 0
        results[label] = {'n': n, 'n_uo': n_uo, 'pct': pct}
        print(f"  {label}: {n_uo}/{n} = {pct:.1f}%")

    # Z-tests comparing each intervention to control
    print("\n--- Reductions vs Control (Z-test of proportions) ---")
    ctrl = results['Control']

    for label in ['Keheala', 'Platform', 'SMS']:
        trt = results[label]

        # Two-sample Z-test
        count = np.array([ctrl['n_uo'], trt['n_uo']])
        nobs = np.array([ctrl['n'], trt['n']])

        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')

        # Calculate difference and CI
        p1 = ctrl['n_uo'] / ctrl['n']
        p2 = trt['n_uo'] / trt['n']
        diff = (p1 - p2) * 100  # Control - Treatment (reduction)

        # Standard error for difference of proportions
        se = np.sqrt(p1*(1-p1)/ctrl['n'] + p2*(1-p2)/trt['n']) * 100
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        p_str = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"

        print(f"  {label}: {diff:.1f} pp (95% CI: {ci_low:.1f}--{ci_high:.1f}; p={p_str})")

    # Also do LTFU
    print("\n--- LTFU (First Month Subsample) ---")
    results_ltfu = {}
    for label, group_name in groups.items():
        grp = early[early['treatment_group'] == group_name]
        n = len(grp)
        n_ltfu = grp['LTFU'].sum() if 'LTFU' in grp.columns else grp['LTFU_outcome'].sum()
        pct = (n_ltfu / n * 100) if n > 0 else 0
        results_ltfu[label] = {'n': n, 'n_ltfu': n_ltfu, 'pct': pct}
        print(f"  {label}: {n_ltfu}/{n} = {pct:.1f}%")

    print("\n--- LTFU Reductions vs Control (Z-test of proportions) ---")
    ctrl_ltfu = results_ltfu['Control']

    for label in ['Keheala', 'Platform', 'SMS']:
        trt = results_ltfu[label]

        count = np.array([ctrl_ltfu['n_ltfu'], trt['n_ltfu']])
        nobs = np.array([ctrl_ltfu['n'], trt['n']])

        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')

        p1 = ctrl_ltfu['n_ltfu'] / ctrl_ltfu['n']
        p2 = trt['n_ltfu'] / trt['n']
        diff = (p1 - p2) * 100

        se = np.sqrt(p1*(1-p1)/ctrl_ltfu['n'] + p2*(1-p2)/trt['n']) * 100
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        p_str = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"

        print(f"  {label}: {diff:.1f} pp (95% CI: {ci_low:.1f}--{ci_high:.1f}; p={p_str})")

    # Deaths
    print("\n--- Deaths (First Month Subsample) ---")
    results_death = {}
    for label, group_name in groups.items():
        grp = early[early['treatment_group'] == group_name]
        n = len(grp)
        n_death = grp['died'].sum()
        pct = (n_death / n * 100) if n > 0 else 0
        results_death[label] = {'n': n, 'n_death': n_death, 'pct': pct}
        print(f"  {label}: {n_death}/{n} = {pct:.1f}%")

    print("\n--- Death Reductions vs Control (Z-test of proportions) ---")
    ctrl_death = results_death['Control']

    for label in ['Keheala', 'Platform', 'SMS']:
        trt = results_death[label]

        count = np.array([ctrl_death['n_death'], trt['n_death']])
        nobs = np.array([ctrl_death['n'], trt['n']])

        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')

        p1 = ctrl_death['n_death'] / ctrl_death['n']
        p2 = trt['n_death'] / trt['n']
        diff = (p1 - p2) * 100

        se = np.sqrt(p1*(1-p1)/ctrl_death['n'] + p2*(1-p2)/trt['n']) * 100
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        p_str = f"{pval:.3f}".lstrip('0') if pval >= 0.001 else "<.001"

        print(f"  {label}: {diff:.1f} pp (95% CI: {ci_low:.1f}--{ci_high:.1f}; p={p_str})")


def main():
    if not os.path.exists(INPUT_DATA_CLEAN):
        print(f"Error: {INPUT_DATA_CLEAN} not found. Please run prepare_study_data.py first.")
        return

    print(f"Loading cleaned data from {INPUT_DATA_CLEAN}...")
    df_clean = pd.read_csv(INPUT_DATA_CLEAN, low_memory=False)

    # Handle de-identified column names
    if "anon_scrn" in df_clean.columns and "scrn" not in df_clean.columns:
        df_clean = df_clean.rename(columns={"anon_scrn": "scrn"})
    if "anon_patient_id" in df_clean.columns and "PatientID" not in df_clean.columns:
        df_clean = df_clean.rename(columns={"anon_patient_id": "PatientID"})

    df = preprocess_data(df_clean)

    analyze_table_si5(df)
    analyze_table_si6(df)
    analyze_mdto_table(df)
    
    analyze_sensitivity_outcome_coding(df)
    
    analyze_si9(df)
    analyze_si10(df)
    analyze_si11(df)
    analyze_si12_uo(df)
    analyze_si13_ltfu(df)
    analyze_time_unadjusted_proportions(df)

if __name__ == "__main__":
    main()
