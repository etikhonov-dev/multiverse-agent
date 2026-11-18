# CDC NHANES 2007-2010 + 2019 Linked Mortality Cohort

## Summary

- **Output file:** `data/nhanes_2007_2010_mort.parquet` (186,093 bytes)
- **Output SHA-256:** `e968d9a9752de372b27f1870cf325593ff07fbc81fab4a78a9f93a73a21f3865`
- **Included adults:** `11,055` (`SEQN` unique, `RIDSTATR == 2`, `ELIGSTAT == 1`, `RIDAGEYR >= 20`, observed `LBXCRP > 0`, non-null `PERMTH_EXM` & `MORTSTAT`)
- **All-cause deaths (`event == 1`):** `1,697` (15.35%)
- **Month-0 continuity (`PERMTH_EXM == 0 -> 0.5`):** `2` participants
- **Follow-up range (`duration_months`):** `0.5` to `159.0` months
- **`UCOD_LEADING`:** Intentionally excluded from `nhanes_2007_2010_mort.parquet` so cause-of-death nullability can never leak mortality status into outcome-blind schemas.

## Inclusion Funnel

| Cycle | DEMO Total | MEC (`RIDSTATR==2`) | Eligible (`ELIGSTAT==1`) | Age >= 20 | Non-null Mortality | Observed `LBXCRP` |
|---|---:|---:|---:|---:|---:|---:|
| 2007 | 10,149 | 9,762 | 5,989 | 5,701 | 5,701 | 5,337 |
| 2009 | 10,537 | 10,253 | 6,346 | 6,045 | 6,045 | 5,718 |

## Recoding & Leakage Guardrails

- Refused (`7`, `77`, `777`) and Don't Know (`9`, `99`, `999`) questionnaire responses are recoded to `NaN`.
- `DIQ010` (diabetes): `1 -> 1.0`, `2` or `3` (borderline) `-> 0.0`.
- `SMQ020`, `MCQ220`, `MCQ160B-F`, `HUQ071`: `1 -> 1.0`, `2 -> 0.0`.
- `prevalent_cvd`: `1.0` if any of `MCQ160B`..`MCQ160F` is `1.0`, `0.0` if all 5 items are observed and `0.0`, else `NaN`.
- `sbp_mmhg`: Mean of non-null `BPXSY1`..`BPXSY4`.
- `log_crp_z`: `log(LBXCRP)` standardized to exact cohort mean `0.0` and population SD `1.0` (`ddof=0`); primary exposure is never imputed.
- `duration_months`: `0.5` when `PERMTH_EXM == 0` (half-month actuarial continuity correction), otherwise `float(PERMTH_EXM)`.

## Column Codebook & Missingness

| Column | Missing Count | Missing Rate |
|---|---:|---:|
| `log_crp_z` | 0 | 0.00% |
| `age_years` | 0 | 0.00% |
| `sex_male` | 0 | 0.00% |
| `bmi` | 137 | 1.24% |
| `sbp_mmhg` | 442 | 4.00% |
| `total_chol_mg_dl` | 46 | 0.42% |
| `hdl_chol_mg_dl` | 46 | 0.42% |
| `smoking_ever` | 6 | 0.05% |
| `prevalent_cvd` | 59 | 0.53% |
| `prevalent_cancer` | 16 | 0.14% |
| `diabetes_dx` | 9 | 0.08% |
| `hospitalized_past_year` | 2 | 0.02% |
| `duration_months` | 0 | 0.00% |
| `event` | 0 | 0.00% |

## Underlying Cause of Death Breakdown (Deceased Only; Excluded from Parquet)

| `UCOD_LEADING` | Cause Category | Deaths |
|---:|---|---:|
| 1 | Diseases of heart (I00-I09, I11, I13, I20-I51) | 392 |
| 2 | Malignant neoplasms (C00-C97) | 405 |
| 3 | Chronic lower respiratory diseases (J40-J47) | 104 |
| 4 | Accidents / unintentional injuries (V01-X59, Y85-Y86) | 47 |
| 5 | Cerebrovascular diseases (I60-I69) | 108 |
| 6 | Alzheimer's disease (G30) | 57 |
| 7 | Diabetes mellitus (E10-E14) | 58 |
| 8 | Influenza and pneumonia (J09-J18) | 42 |
| 9 | Nephritis, nephrotic syndrome and nephrosis (N00-N07, N17-N19, N25-N27) | 28 |
| 10 | All other causes (residual) | 456 |

## Upstream Source Artifacts & SHA-256 Checksums

| File | Source URL | SHA-256 | Bytes |
|---|---|---|---:|
| `DEMO_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.xpt` | `02b8e97703d04bc99769c751881bece51ac9ba68cf18a2992a45d3bdfb856147` | 3,498,080 |
| `CRP_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CRP_E.xpt` | `5cf3fc87d8c5eb8b8c053e5a8ea183b62dd29b417c4aeaaa53dc4b354ef611ab` | 140,480 |
| `BMX_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.xpt` | `b67b50dbd12c9510b49b6348b595b9fd073c422c0c6eac182f90ac57b491a97d` | 1,800,240 |
| `BPX_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPX_E.xpt` | `5a0303e68e0a4d7ebdbd2bc82f87852c31e44f1733782e107219bed9303baed5` | 2,113,200 |
| `TCHOL_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/TCHOL_E.xpt` | `1e137eb4fcccd0c0e104b4b1849f00b09e8a9077e2143e844a09bb673507fff1` | 196,400 |
| `HDL_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HDL_E.xpt` | `3c7021fb4411990551c3ea3f86c5b870c325a23b2b1656054d1ec10883bee19c` | 196,400 |
| `SMQ_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SMQ_E.xpt` | `881865ee05f4b2be0c03c761f00ad8e355f2cc5e7d3be41666edef6f3dca5862` | 2,449,520 |
| `DIQ_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DIQ_E.xpt` | `fb4d698fbec26bc8a09b2cd0f3019f815114742dd2c37ccac1aaf5e0123394c5` | 2,789,600 |
| `MCQ_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/MCQ_E.xpt` | `dfbe11c6b036ec4e79aeba7805833fc4951c74662a9510066c3a93932f6990e6` | 7,127,840 |
| `HUQ_E.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/HUQ_E.xpt` | `6280e8f2062b2e8aa717848e4cc5a635dec01b871ef0efff5287fdb126abcc49` | 814,080 |
| `NHANES_2007_2008_MORT_2019_PUBLIC.dat` | `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2007_2008_MORT_2019_PUBLIC.dat` | `75f89bd545a13ba726a7eb56de4d62f6dfc8f0f76d505dc18ced9752d5e2c77d` | 498,376 |
| `DEMO_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.xpt` | `78c4d4a6e5eafa9da3bbb24a90d140fef5d8c0cda932a7ff8ee609288d4fae07` | 3,631,600 |
| `CRP_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/CRP_F.xpt` | `049ab627054bda1b540353b0e80bfd9c1ef2ef67b22cb0f1a84f22a42404fc1f` | 148,480 |
| `BMX_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BMX_F.xpt` | `1fa42e07764037674c3f56e1e76f990e10c32245897597de3d731c57b91d869a` | 1,890,560 |
| `BPX_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPX_F.xpt` | `d09af545c732e5fb72c437e870823ec5d9efaed1ef3522210dcc1d8f47b9be57` | 2,219,280 |
| `TCHOL_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/TCHOL_F.xpt` | `87a237c6bcf8fe623db0ace18c5629bc6e8fddd98d4202a5a2d449958d13c4db` | 207,440 |
| `HDL_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HDL_F.xpt` | `d813b41ddf57bfd791a354463b5d90e9276ed3c2464b7f68bef4d007719c7a50` | 207,440 |
| `SMQ_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SMQ_F.xpt` | `ee85c82ea55046eb7bf491088b5bb9efecbea8aaf66ae8c89123bad23a5f1901` | 2,580,560 |
| `DIQ_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DIQ_F.xpt` | `7e90371cbc98e77a9684eff32ffa0e4ee3e8fc04d97581a8773e19cb9fe901fa` | 1,620,960 |
| `MCQ_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/MCQ_F.xpt` | `c2f0f9223c6a9e38e5bf6d26501c4b11d3cfd97b82db46e1c9c266f24761c457` | 6,562,720 |
| `HUQ_F.xpt` | `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/HUQ_F.xpt` | `cd710fb631d5918b3557428811b7f2cb1332c305163ea5563143ac45ff7bd4b2` | 845,120 |
| `NHANES_2009_2010_MORT_2019_PUBLIC.dat` | `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2009_2010_MORT_2019_PUBLIC.dat` | `eabf074331e20d3f3cbb8ddf21e3a43e517f5694187c12b755741f4472894786` | 517,751 |
