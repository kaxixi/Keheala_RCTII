# Keheala Study 2 — Data Analysis in Python

Data and analysis code for a cluster-randomized controlled trial (RCT) evaluating digital adherence technologies for tuberculosis (TB) treatment in Kenya, conducted in partnership with USAID.

The study randomized patients across four arms — Control, SMS Reminders, a digital platform (SBCC), and Keheala — and measured treatment outcomes including unsuccessful outcomes, loss to follow-up (LTFU), and death.

## Quick Start

**Requirements:** Python 3 with `pandas`, `numpy`, and `statsmodels`.

```bash
pip install pandas numpy statsmodels
```

**Run the analysis** (scripts can be run in any order):

```bash
python3 Python_Analysis/analysis_manuscript.py    # Tables 1–4
python3 Python_Analysis/analysis_si.py            # Supplementary Tables SI5–SI13
python3 Python_Analysis/analysis_dqa.py           # Data Quality Assessment (SI14)
python3 Python_Analysis/generate_consort.py       # CONSORT flow diagram numbers
```

Output: LaTeX `.tex` files are written to `Python_Analysis/output/`.

## Repository Structure

```
Keheala_RCTII/
├── README.md
├── deidentified_data/level2/               # De-identified datasets for analysis
│   ├── study2_cleaned.csv                  # Study data (N=17,160 rows; mITT N=14,962)
│   ├── TIBU_firstnm_deidentified.csv       # Kenya national TB registry
│   ├── Urine_Test_Results.csv              # Urine verification data
│   └── DQA_combined.csv                    # Data quality assessment records
├── Python_Analysis/
│   ├── analysis_manuscript.py              # Tables 1–4 (main manuscript)
│   ├── analysis_si.py                      # Supplementary Tables SI5–SI13
│   ├── analysis_dqa.py                     # Data Quality Assessment (SI14)
│   ├── generate_consort.py                 # CONSORT flow diagram
│   ├── prepare_study_data.py               # Data cleaning [reference only]
│   ├── prepare_TIBU_data.py                # TIBU aggregation [reference only]
│   ├── deidentify_data.py                  # De-identification pipeline [reference only]
│   └── output/                             # Generated LaTeX tables
```

## Analysis Scripts

The following scripts constitute the analysis pipeline. Each reads from `deidentified_data/level2/` and writes LaTeX tables to `Python_Analysis/output/`. They can be run in any order.

#### `analysis_manuscript.py`
*   **Table 1**: Baseline Characteristics — Study Sample vs Kenya National TIBU Registry.
*   **Table 2**: Descriptive Statistics by Treatment Group.
*   **Table 3**: Primary Outcomes (Unsuccessful, LTFU, Death) — Unadjusted, Partially Adjusted, and Fully Adjusted models.
*   **Table 4**: Urine Test Verification Analysis.

#### `analysis_si.py`
*   Supplementary Tables SI5–SI13: Comparison of Interventions (SI5), Exclusion/NC Counts (SI6), Missing Data Sensitivity (SI7–SI8), Subgroup Analyses for bacteriologically confirmed, pulmonary, and non-retreatment patients (SI9–SI11), Time-Split Analyses (SI12–SI13).

#### `analysis_dqa.py`
*   Data Quality Assessment (SI14): Platform-level outcome comparisons and county-level crosstabs.

#### `generate_consort.py`
*   Calculates numbers for the CONSORT flow diagram.

## Data Preparation Scripts (Reference Only)

Three additional scripts document the full pipeline from raw data to the de-identified datasets provided in this repository. They require the original study data, which contains personally identifiable information and cannot be shared publicly. They are included for transparency and auditability.

| Script | Purpose |
|--------|---------|
| `prepare_study_data.py` | Cleans raw study data, creates outcome variables and exclusion flags, outputs `study2_cleaned_python.csv` |
| `prepare_TIBU_data.py` | Aggregates Kenya's national TIBU registry with trust-based deduplication |
| `deidentify_data.py` | Produces de-identified datasets at two levels from the cleaned data |

The de-identification pipeline generates datasets at two levels: Level 1 (names removed, locations preserved) and Level 2 (names and locations removed). Only Level 2 is included in this public repository. Both levels produce identical statistical results.

## Data

### De-identified Study Data (`study2_cleaned.csv`)

The study dataset contains 17,160 patient records. Each row includes treatment assignment, demographic covariates, and treatment outcomes. Pre-computed flags indicate analysis inclusion:

- `duplicate` — 1 if the record is a duplicate/test entry or misdiagnosis (excluded from all analyses)
- `MITT` — 1 if the record is in the modified intent-to-treat sample (N=14,962)
- `prereg_exclusion` — 1 if excluded from mITT (duplicates, transfers out, missing outcomes, not evaluated)
- `unsuccessful_outcome`, `died`, `LTFU`, `failed` — binary outcome variables

Patient identifiers have been replaced with anonymous integers (`anon_patient_id`, `anon_scrn`). Facility names and county names have been replaced with numeric IDs (`clinic_id`, `county_id`, `subcounty_id`).

### National TB Registry (`TIBU_firstnm_deidentified.csv`)

Kenya's Treatment Information from Basic Units (TIBU) registry, used in Table 1 to compare the study sample against the national population. Patient names have been removed; facility and location names have been replaced with numeric IDs. An `is_participating_clinic` flag identifies clinics that participated in the study.

### De-identification Details

The de-identification process (documented in `deidentify_data.py`) removes patient names, phone numbers, and sponsor names, and replaces all identifiers (patient IDs, registration numbers, facility names, county names) with anonymous sequential integers. The SCRN (subcounty registration number) mapping uses a composite key of `(scrn, subcounty)` to ensure cross-subcounty uniqueness. The same mappings are applied consistently across all datasets so that cross-file merges remain valid.

The de-identification script renames identifier columns as follows:

| Original Column | De-identified Column |
|---|---|
| `PatientID` | `anon_patient_id` |
| `scrn` | `anon_scrn` |
| `subcountyregistrationnumber` | `anon_scrn_tibu` (TIBU) / `anon_scrn_orig` (study) |

## Statistical Methods

The analysis uses linear probability models (OLS) estimated with `statsmodels`:

- **Unadjusted**: Treatment group indicators only
- **Partially adjusted**: Adds study month and clinic fixed effects
- **Fully adjusted**: Adds demographic and clinical covariates (sex, age, HIV status, comorbidities, extrapulmonary TB, retreatment status, nutrition support, bacteriological confirmation, drug resistance, treatment month at enrollment)

Treatment effects are reported as reductions in risk (percentage points) relative to the control group. Fully adjusted models use complete cases (N=13,485) due to missing covariate values.

## License

[To be determined]
