# Keheala Study 2 - Python Analysis Replication

This directory contains the Python scripts used to replicate the Stata analysis for the Keheala Study 2 RCT (Randomized Controlled Trial) evaluating digital adherence technologies for TB treatment in Kenya.

## Directory Structure

```
Data_Analysis/
├── original_data/             # READ-ONLY: Raw study data (not public, contains PII)
├── TIBU_data/                 # National TB registry raw data
├── deidentified_data/         # De-identified datasets (public)
│   ├── level1/                #   Names removed, locations preserved
│   └── level2/                #   Names AND locations removed (default for analysis)
├── Python_Analysis/
│   ├── prepare_study_data.py  # Data Cleaning [requires original data]
│   ├── prepare_TIBU_data.py   # TIBU Aggregation [requires original data]
│   ├── deidentify_data.py     # De-identification Pipeline
│   ├── analysis_manuscript.py # Manuscript Tables (Tables 1-4)
│   ├── analysis_si.py         # Supplementary Information (Tables SI5-SI13)
│   ├── analysis_dqa.py        # Data Quality Assessment (SI14)
│   ├── generate_consort.py    # CONSORT Diagram Numbers
│   ├── output/                # Generated LaTeX tables
│   │   ├── tbl1_TIBU_summarystats.tex
│   │   ├── tbl2_treatmentgroup_summary_stats.tex
│   │   ├── tbl3_primary_outcomes_ATEs.tex
│   │   ├── tbl4_medication_adherence.tex
│   │   ├── tblSI_*.tex
│   │   └── ...
│   └── README.md
```

## Replication with De-identified Data

The raw study data cannot be shared publicly because it contains personally identifiable information (patient names, phone numbers, facility names). De-identified datasets are provided in `deidentified_data/level2/`.

**All analysis scripts read from `deidentified_data/level2/` by default.** No file copying is needed.

### De-identification

Patient names, phone numbers, and sponsor names are removed. Patient IDs and registration numbers are replaced with anonymous sequential integers. Facility names, zones, provinces, and serial numbers are removed. County and sub-county names are replaced with numeric IDs. The TIBU dataset includes an `is_participating_clinic` boolean flag so Table 1 can filter participating clinics without facility names.

### Replication Instructions

Run the analysis scripts (in any order):

```bash
python3 Python_Analysis/analysis_manuscript.py    # Tables 1-4
python3 Python_Analysis/analysis_si.py            # SI Tables 5-13
python3 Python_Analysis/analysis_dqa.py           # DQA Table (SI14)
python3 Python_Analysis/generate_consort.py       # CONSORT flow diagram
```

Output: LaTeX `.tex` files are written to `Python_Analysis/output/`.

### Column Name Changes in De-identified Data

The de-identification script renames identifier columns:

| Original Column | De-identified Column |
|---|---|
| `PatientID` | `anon_patient_id` |
| `scrn` | `anon_scrn` |
| `subcountyregistrationnumber` | `anon_scrn_tibu` (TIBU) / `anon_scrn_orig` (study) |

**Note on SCRN uniqueness:** SCRN (subcounty registration number) is only unique within a subcounty, not globally. The de-identification script uses a composite key of `(scrn, syssubcounty)` to generate `anon_scrn`, ensuring patients with the same SCRN in different subcounties receive distinct anonymous IDs. This same composite key is applied to both the study data and DQA data so cross-file merges remain correct.

The analysis scripts handle both naming conventions automatically.

## Full Pipeline (Authors Only)

The complete pipeline requires the original (non-public) data. These scripts are provided for transparency.

1. **Data Preparation** (requires raw data in `original_data/` and `TIBU_data/`):
   ```bash
   python3 Python_Analysis/prepare_study_data.py     # Raw -> output/study2_cleaned_python.csv
   python3 Python_Analysis/prepare_TIBU_data.py      # TIBU Excel -> TIBU_data/output/TIBU_firstnm_deidentified.csv
   ```

2. **De-identification** (reads from step 1 outputs + `original_data/`):
   ```bash
   python3 Python_Analysis/deidentify_data.py        # -> deidentified_data/level1/ and level2/
   ```

3. **Analysis** (reads from `deidentified_data/level2/`):
   ```bash
   python3 Python_Analysis/analysis_manuscript.py
   python3 Python_Analysis/analysis_si.py
   python3 Python_Analysis/analysis_dqa.py
   ```

## Analysis Scripts

#### `analysis_manuscript.py`
*   **Purpose**: Generates the main manuscript tables.
    *   **Table 1**: Baseline Characteristics (Study Sample vs National TIBU).
    *   **Table 2**: Descriptive Statistics by Treatment Group.
    *   **Table 3**: Main Outcomes (Unsuccessful, LTFU, Death) - Unadjusted, Partially Adjusted, Fully Adjusted models.
    *   **Table 4**: Urine Test Verification Analysis.
*   **Input**: `deidentified_data/level2/` (study data, TIBU data, Urine results)
*   **Output**: `output/tbl{1-4}_*.tex`

#### `analysis_si.py`
*   **Purpose**: Generates Supplementary Information Tables SI5-SI13.
*   **Includes**: Comparison of Interventions (SI5), Exclusion/NC Tables (SI6), Missing Data Sensitivity (SI7-SI8), Subgroup Analyses (SI9-SI11), Time-Split Analyses (SI12-SI13).
*   **Input**: `deidentified_data/level2/study2_cleaned.csv`
*   **Output**: `output/tblSI_*.tex`

#### `analysis_dqa.py`
*   **Purpose**: Performs Data Quality Assessment (DQA) analysis (SI14).
*   **Input**: `deidentified_data/level2/` (study data + DQA county files)
*   **Output**: `output/tblSI_DQA_*.tex`

#### `generate_consort.py`
*   **Purpose**: Calculates numbers for the CONSORT flow diagram.
*   **Input**: `deidentified_data/level2/study2_cleaned.csv`
*   **Output**: `output/consort_flow.log`

## Requirements

*   Python 3.x
*   pandas
*   numpy
*   statsmodels
