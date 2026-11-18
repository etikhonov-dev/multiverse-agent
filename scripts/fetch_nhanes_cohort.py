#!/usr/bin/env python3
"""Fetch and build the CDC NHANES 2007-2010 + 2019 Linked Mortality parquet cohort.

Sources:
- CDC NHANES 2007-2008 (_E) and 2009-2010 (_F) public XPT files
- NCHS 2019 Public-Use Linked Mortality Files (.dat)

Outputs:
- data/nhanes_2007_2010_mort.parquet (without UCOD_LEADING)
- data/DATASETS.md (source URLs, SHA-256 checksums, recoding rules, codebook)
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import ssl
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PARQUET_PATH = REPO_ROOT / "data" / "nhanes_2007_2010_mort.parquet"
DEFAULT_DATASETS_MD_PATH = REPO_ROOT / "data" / "DATASETS.md"

NHANES_XPT_FILES: dict[str, tuple[str, ...]] = {
    "2007": (
        "DEMO_E",
        "CRP_E",
        "BMX_E",
        "BPX_E",
        "TCHOL_E",
        "HDL_E",
        "SMQ_E",
        "DIQ_E",
        "MCQ_E",
        "HUQ_E",
    ),
    "2009": (
        "DEMO_F",
        "CRP_F",
        "BMX_F",
        "BPX_F",
        "TCHOL_F",
        "HDL_F",
        "SMQ_F",
        "DIQ_F",
        "MCQ_F",
        "HUQ_F",
    ),
}

LMF_FILES: dict[str, str] = {
    "2007": (
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
        "NHANES_2007_2008_MORT_2019_PUBLIC.dat"
    ),
    "2009": (
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
        "NHANES_2009_2010_MORT_2019_PUBLIC.dat"
    ),
}

UCOD_LABELS: dict[int, str] = {
    1: "Diseases of heart (I00-I09, I11, I13, I20-I51)",
    2: "Malignant neoplasms (C00-C97)",
    3: "Chronic lower respiratory diseases (J40-J47)",
    4: "Accidents / unintentional injuries (V01-X59, Y85-Y86)",
    5: "Cerebrovascular diseases (I60-I69)",
    6: "Alzheimer's disease (G30)",
    7: "Diabetes mellitus (E10-E14)",
    8: "Influenza and pneumonia (J09-J18)",
    9: "Nephritis, nephrotic syndrome and nephrosis (N00-N07, N17-N19, N25-N27)",
    10: "All other causes (residual)",
}

REFUSAL_CODES = frozenset({7.0, 9.0, 77.0, 99.0, 777.0, 999.0})
CVD_ITEMS = ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")


@dataclass(frozen=True)
class SourceArtifact:
    name: str
    url: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    if importlib.util.find_spec("certifi") is not None:
        certifi_mod = importlib.import_module("certifi")
        return ssl.create_default_context(cafile=str(certifi_mod.where()))
    macos_ca = Path("/etc/ssl/cert.pem")
    if macos_ca.exists():
        return ssl.create_default_context(cafile=str(macos_ca))
    return ssl.create_default_context()


def _https_get_bytes(url: str, max_redirects: int = 5) -> bytes:
    current = url
    ctx = _ssl_context()
    for _ in range(max_redirects):
        parsed = urllib.parse.urlsplit(current)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"Only HTTPS URLs with host are permitted: {current}")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=60, context=ctx)
        try:
            conn.request("GET", path, headers={"User-Agent": "multiverse-agent/0.1"})
            resp = conn.getresponse()
            if resp.status in {301, 302, 303, 307, 308}:
                location = resp.getheader("Location")
                if not location:
                    raise RuntimeError(f"Redirect without Location from {current}")
                current = urllib.parse.urljoin(current, location)
                continue
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} fetching {current}")
            return resp.read()
        finally:
            conn.close()
    raise RuntimeError(f"Too many redirects fetching {url}")


def download_file(url: str, dest: Path) -> SourceArtifact:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        payload = _https_get_bytes(url)
        dest.write_bytes(payload)
    return SourceArtifact(
        name=dest.name,
        url=url,
        sha256=sha256_file(dest),
        size_bytes=dest.stat().st_size,
    )


def _parse_lmf_dat(path: Path) -> pd.DataFrame:
    """Parse NCHS 2019 Public-Use Linked Mortality fixed-width ASCII file."""
    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.strip():
                continue
            seqn_str = line[0:6].strip()
            elig_str = line[14:15].strip()
            mort_str = line[15:16].strip()
            ucod_str = line[16:19].strip()
            permth_exm_str = line[45:48].strip()
            rows.append(
                {
                    "SEQN": int(seqn_str),
                    "ELIGSTAT": int(elig_str) if elig_str not in {"", "."} else np.nan,
                    "MORTSTAT": int(mort_str) if mort_str not in {"", "."} else np.nan,
                    "UCOD_LEADING": int(ucod_str) if ucod_str not in {"", "."} else np.nan,
                    "PERMTH_EXM": (
                        int(permth_exm_str) if permth_exm_str not in {"", "."} else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _recode_yes_no(series: pd.Series, *, zero_codes: tuple[float, ...] = (2.0,)) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out = out.mask(values == 1.0, 1.0)
    for code in zero_codes:
        out = out.mask(values == code, 0.0)
    out = out.mask(values.isin(REFUSAL_CODES), np.nan)
    return out


def _read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    df.columns = [str(col).upper() for col in df.columns]
    available = [col for col in columns if col in df.columns]
    out = df[available].copy()
    out["SEQN"] = pd.to_numeric(out["SEQN"], errors="raise").astype("int64")
    return out


def _build_cycle(
    cycle: str, raw_dir: Path
) -> tuple[pd.DataFrame, list[SourceArtifact], dict[str, int]]:
    artifacts: list[SourceArtifact] = []
    paths: dict[str, Path] = {}
    for stem in NHANES_XPT_FILES[cycle]:
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/DataFiles/{stem}.xpt"
        dest = raw_dir / f"{stem}.xpt"
        artifacts.append(download_file(url, dest))
        paths[stem.split("_")[0]] = dest

    lmf_dest = raw_dir / Path(LMF_FILES[cycle]).name
    artifacts.append(download_file(LMF_FILES[cycle], lmf_dest))

    demo = _read_xpt(paths["DEMO"], ["SEQN", "RIDSTATR", "RIDAGEYR", "RIAGENDR"])
    crp = _read_xpt(paths["CRP"], ["SEQN", "LBXCRP"])
    bmx = _read_xpt(paths["BMX"], ["SEQN", "BMXBMI"])
    bpx = _read_xpt(paths["BPX"], ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
    tchol = _read_xpt(paths["TCHOL"], ["SEQN", "LBXTC"])
    hdl = _read_xpt(paths["HDL"], ["SEQN", "LBDHDD"])
    smq = _read_xpt(paths["SMQ"], ["SEQN", "SMQ020"])
    diq = _read_xpt(paths["DIQ"], ["SEQN", "DIQ010"])
    mcq = _read_xpt(paths["MCQ"], ["SEQN", *CVD_ITEMS, "MCQ220"])
    huq = _read_xpt(paths["HUQ"], ["SEQN", "HUQ071"])
    lmf = _parse_lmf_dat(lmf_dest)

    merged = (
        demo.merge(lmf, on="SEQN", how="inner")
        .merge(crp, on="SEQN", how="left")
        .merge(bmx, on="SEQN", how="left")
        .merge(bpx, on="SEQN", how="left")
        .merge(tchol, on="SEQN", how="left")
        .merge(hdl, on="SEQN", how="left")
        .merge(smq, on="SEQN", how="left")
        .merge(diq, on="SEQN", how="left")
        .merge(mcq, on="SEQN", how="left")
        .merge(huq, on="SEQN", how="left")
    )

    funnel: dict[str, int] = {"demo_total": int(len(demo))}
    merged = merged[merged["RIDSTATR"] == 2].copy()
    funnel["mec_examined"] = int(len(merged))
    merged = merged[merged["ELIGSTAT"] == 1].copy()
    funnel["mortality_eligible"] = int(len(merged))
    merged = merged[merged["RIDAGEYR"] >= 20].copy()
    funnel["age_ge_20"] = int(len(merged))
    merged = merged[merged["PERMTH_EXM"].notna() & merged["MORTSTAT"].notna()].copy()
    funnel["non_null_mortality"] = int(len(merged))
    merged = merged[merged["LBXCRP"].notna() & (merged["LBXCRP"] > 0)].copy()
    funnel["observed_crp"] = int(len(merged))

    sbp_cols = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
    sbp_mean = merged[sbp_cols].mean(axis=1, skipna=True)

    cvd_items = pd.DataFrame({col: _recode_yes_no(merged[col]) for col in CVD_ITEMS})
    any_cvd = cvd_items.eq(1.0).any(axis=1)
    all_cvd_observed = cvd_items.notna().all(axis=1)
    prevalent_cvd = pd.Series(np.nan, index=merged.index, dtype="float64")
    prevalent_cvd = prevalent_cvd.mask(all_cvd_observed, 0.0)
    prevalent_cvd = prevalent_cvd.mask(any_cvd, 1.0)

    permth = merged["PERMTH_EXM"].astype("float64")
    duration_months = np.where(permth == 0.0, 0.5, permth)

    out = pd.DataFrame(
        {
            "SEQN": merged["SEQN"].astype("int64"),
            "log_crp_raw": np.log(merged["LBXCRP"].astype("float64")),
            "age_years": merged["RIDAGEYR"].astype("float64"),
            "sex_male": np.where(merged["RIAGENDR"] == 1, 1.0, 0.0).astype("float64"),
            "bmi": pd.to_numeric(merged["BMXBMI"], errors="coerce").astype("float64"),
            "sbp_mmhg": sbp_mean.astype("float64"),
            "total_chol_mg_dl": pd.to_numeric(merged["LBXTC"], errors="coerce").astype("float64"),
            "hdl_chol_mg_dl": pd.to_numeric(merged["LBDHDD"], errors="coerce").astype("float64"),
            "smoking_ever": _recode_yes_no(merged["SMQ020"]),
            "prevalent_cvd": prevalent_cvd,
            "prevalent_cancer": _recode_yes_no(merged["MCQ220"]),
            "diabetes_dx": _recode_yes_no(merged["DIQ010"], zero_codes=(2.0, 3.0)),
            "hospitalized_past_year": _recode_yes_no(merged["HUQ071"]),
            "duration_months": duration_months.astype("float64"),
            "event": merged["MORTSTAT"].astype("int64"),
            "_ucod_leading": merged["UCOD_LEADING"],
        }
    )
    return out, artifacts, funnel


def _render_datasets_md(
    df: pd.DataFrame,
    ucod_series: pd.Series,
    parquet_sha256: str,
    parquet_size: int,
    artifacts: list[SourceArtifact],
    funnels: dict[str, dict[str, int]],
    month_zero_count: int,
) -> str:
    n_rows = len(df)
    n_events = int(df["event"].sum())
    ucod_counts = ucod_series[df["event"] == 1].dropna().astype(int).value_counts().sort_index()
    ucod_lines = [
        f"| {code} | {UCOD_LABELS.get(int(code), 'Unknown')} | {int(cnt)} |"
        for code, cnt in ucod_counts.items()
    ]
    missing_lines = [
        f"| `{col}` | {int(df[col].isna().sum())} | {df[col].isna().mean() * 100:.2f}% |"
        for col in df.columns
        if col != "SEQN"
    ]
    source_lines = [
        f"| `{a.name}` | `{a.url}` | `{a.sha256}` | {a.size_bytes:,} |" for a in artifacts
    ]
    funnel_lines = [
        f"| {cycle} | {f['demo_total']:,} | {f['mec_examined']:,} | "
        f"{f['mortality_eligible']:,} | {f['age_ge_20']:,} | "
        f"{f['non_null_mortality']:,} | {f['observed_crp']:,} |"
        for cycle, f in funnels.items()
    ]

    dur_min = df["duration_months"].min()
    dur_max = df["duration_months"].max()
    death_pct = n_events / n_rows * 100

    return "\n".join(
        [
            "# CDC NHANES 2007-2010 + 2019 Linked Mortality Cohort",
            "",
            "## Summary",
            "",
            f"- **Output file:** `data/nhanes_2007_2010_mort.parquet` ({parquet_size:,} bytes)",
            f"- **Output SHA-256:** `{parquet_sha256}`",
            (
                f"- **Included adults:** `{n_rows:,}` (`SEQN` unique, `RIDSTATR == 2`, "
                "`ELIGSTAT == 1`, `RIDAGEYR >= 20`, observed `LBXCRP > 0`, "
                "non-null `PERMTH_EXM` & `MORTSTAT`)"
            ),
            f"- **All-cause deaths (`event == 1`):** `{n_events:,}` ({death_pct:.2f}%)",
            (
                f"- **Month-0 continuity (`PERMTH_EXM == 0 -> 0.5`):** "
                f"`{month_zero_count}` participants"
            ),
            f"- **Follow-up range (`duration_months`):** `{dur_min:.1f}` to `{dur_max:.1f}` months",
            (
                "- **`UCOD_LEADING`:** Intentionally excluded from "
                "`nhanes_2007_2010_mort.parquet` so cause-of-death nullability can never "
                "leak mortality status into outcome-blind schemas."
            ),
            "",
            "## Inclusion Funnel",
            "",
            (
                "| Cycle | DEMO Total | MEC (`RIDSTATR==2`) | Eligible (`ELIGSTAT==1`) "
                "| Age >= 20 | Non-null Mortality | Observed `LBXCRP` |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            *funnel_lines,
            "",
            "## Recoding & Leakage Guardrails",
            "",
            (
                "- Refused (`7`, `77`, `777`) and Don't Know (`9`, `99`, `999`) "
                "questionnaire responses are recoded to `NaN`."
            ),
            "- `DIQ010` (diabetes): `1 -> 1.0`, `2` or `3` (borderline) `-> 0.0`.",
            "- `SMQ020`, `MCQ220`, `MCQ160B-F`, `HUQ071`: `1 -> 1.0`, `2 -> 0.0`.",
            (
                "- `prevalent_cvd`: `1.0` if any of `MCQ160B`..`MCQ160F` is `1.0`, "
                "`0.0` if all 5 items are observed and `0.0`, else `NaN`."
            ),
            "- `sbp_mmhg`: Mean of non-null `BPXSY1`..`BPXSY4`.",
            (
                "- `log_crp_z`: `log(LBXCRP)` standardized to exact cohort mean `0.0` "
                "and population SD `1.0` (`ddof=0`); primary exposure is never imputed."
            ),
            (
                "- `duration_months`: `0.5` when `PERMTH_EXM == 0` (half-month actuarial "
                "continuity correction), otherwise `float(PERMTH_EXM)`."
            ),
            "",
            "## Column Codebook & Missingness",
            "",
            "| Column | Missing Count | Missing Rate |",
            "|---|---:|---:|",
            *missing_lines,
            "",
            "## Underlying Cause of Death Breakdown (Deceased Only; Excluded from Parquet)",
            "",
            "| `UCOD_LEADING` | Cause Category | Deaths |",
            "|---:|---|---:|",
            *ucod_lines,
            "",
            "## Upstream Source Artifacts & SHA-256 Checksums",
            "",
            "| File | Source URL | SHA-256 | Bytes |",
            "|---|---|---|---:|",
            *source_lines,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CDC NHANES 2007-2010 mortality cohort.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_DATASETS_MD_PATH)
    args = parser.parse_args(argv)

    cycles_df: list[pd.DataFrame] = []
    all_artifacts: list[SourceArtifact] = []
    funnels: dict[str, dict[str, int]] = {}
    for cycle in ("2007", "2009"):
        df_cycle, artifacts, funnel = _build_cycle(cycle, args.raw_dir)
        cycles_df.append(df_cycle)
        all_artifacts.extend(artifacts)
        funnels[cycle] = funnel

    combined = pd.concat(cycles_df, ignore_index=True)
    combined = combined.sort_values("SEQN", kind="mergesort").reset_index(drop=True)
    if combined["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN detected after merging cycles.")

    log_crp = combined.pop("log_crp_raw")
    combined.insert(1, "log_crp_z", (log_crp - log_crp.mean()) / log_crp.std(ddof=0))
    ucod_series = combined.pop("_ucod_leading")
    month_zero_count = int((combined["duration_months"] == 0.5).sum())

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out_parquet, index=False)

    parquet_sha = sha256_file(args.out_parquet)
    parquet_size = args.out_parquet.stat().st_size
    md_text = _render_datasets_md(
        df=combined,
        ucod_series=ucod_series,
        parquet_sha256=parquet_sha,
        parquet_size=parquet_size,
        artifacts=all_artifacts,
        funnels=funnels,
        month_zero_count=month_zero_count,
    )
    args.out_md.write_text(md_text, encoding="utf-8")

    print(
        f"Built {args.out_parquet} ({parquet_size:,} bytes, sha256={parquet_sha}): "
        f"N={len(combined):,}, events={int(combined['event'].sum()):,}, "
        f"month0={month_zero_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
