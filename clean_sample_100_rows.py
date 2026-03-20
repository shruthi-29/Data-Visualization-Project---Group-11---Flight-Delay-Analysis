from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from pickle import FALSE
from typing import Any, Optional

import numpy as np
import pandas as pd


INPUT_CSV = Path("flights_sample_3m.csv")
OUTPUT_CSV = INPUT_CSV.with_name("cleaned_flights_sample_3m.csv")
REPORT_JSON = INPUT_CSV.with_name("cleaning_report_flights_sample_3m.json")

CHUNKSIZE = 200_000
DROP_CANCELLED = False
DROP_COLS = ["AIRLINE_DOT", "ORIGIN_CITY", "DEST_CITY"]

TIME_COLS = [
    "CRS_DEP_TIME",
    "DEP_TIME",
    "WHEELS_OFF",
    "WHEELS_ON",
    "CRS_ARR_TIME",
    "ARR_TIME",
]

NUMERIC_COLS = [
    "DOT_CODE",
    "FL_NUMBER",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

DELAY_COMPONENT_COLS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

WINSOR_COLS = ["DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "ELAPSED_TIME"]
NONNEGATIVE_COLS = ["TAXI_OUT", "TAXI_IN", "AIR_TIME", "ELAPSED_TIME"]


@dataclass
class CleaningReport:
    input_rows: int
    output_rows: int
    dropped_cancelled_rows: int
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    winsor_bounds: dict[str, dict[str, float]]
    cancelled_rows: int
    diverted_rows: int


def hhmm_to_minutes(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        iv = int(round(float(value)))
    except Exception:
        return None

    if iv < 0:
        return None
    if iv == 2400:
        return 0

    s = str(iv).zfill(4)
    hh = int(s[:-2])
    mm = int(s[-2:])

    if hh > 24 or mm > 59:
        return None
    if hh == 24:
        hh = 0

    return hh * 60 + mm


def build_event_datetime(flight_date: pd.Series, minutes: pd.Series, ref_minutes: pd.Series | None = None) -> pd.Series:
    dt = pd.to_datetime(flight_date, errors="coerce")
    mins = minutes.astype("Int64")

    out = dt + pd.to_timedelta(mins.fillna(0), unit="m")
    out = out.where(~mins.isna(), pd.NaT)

    if ref_minutes is not None:
        ref = ref_minutes.astype("Int64")
        rollover = (~mins.isna()) & (~ref.isna()) & (mins < ref)
        out = out + pd.to_timedelta(rollover.astype(int) * 1440, unit="m")

    return out


def split_city_state(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = s.astype("string").str.split(",", n=1, expand=True)
    city = parts[0].str.strip().astype("string")
    if parts.shape[1] > 1:
        state = parts[1].str.strip().astype("string")
    else:
        state = pd.Series(pd.NA, index=s.index, dtype="string")
    return city, state


def compute_winsor_bounds(input_path: Path, chunksize: int = CHUNKSIZE, sample_target: int = 200_000) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(7)
    samples = {col: [] for col in WINSOR_COLS}
    collected = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        for col in WINSOR_COLS:
            if col not in chunk.columns:
                continue

            x = pd.to_numeric(chunk[col], errors="coerce").dropna().to_numpy(dtype=float)
            if x.size == 0:
                continue

            remaining = sample_target - collected
            if remaining <= 0:
                break

            take = min(remaining, max(1000, int(0.01 * x.size)))
            if take >= x.size:
                chosen = x
            else:
                idx = rng.choice(x.size, size=take, replace=False)
                chosen = x[idx]

            samples[col].append(chosen)
            collected += chosen.size

        if collected >= sample_target:
            break

    bounds = {}
    for col, arrays in samples.items():
        if not arrays:
            continue

        x = np.concatenate(arrays)
        q1 = float(np.percentile(x, 25))
        q3 = float(np.percentile(x, 75))
        iqr = q3 - q1

        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr

        if col in NONNEGATIVE_COLS:
            lower = max(0.0, lower)

        bounds[col] = {"lower": lower, "upper": upper}

    return bounds


def process_chunk(df: pd.DataFrame, winsor_bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = df.copy()

    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "CANCELLATION_CODE" in df.columns and "CANCELLED" in df.columns:
        df["CANCELLATION_CODE"] = df["CANCELLATION_CODE"].astype("string")
        mask = (df["CANCELLED"] == 1) & (df["CANCELLATION_CODE"].isna())
        df.loc[mask, "CANCELLATION_CODE"] = "UNKNOWN"

    for col in DELAY_COMPONENT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if "CANCELLED" in df.columns and "DIVERTED" in df.columns:
                mask = (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0) & (df[col].isna())
                df.loc[mask, col] = 0.0

    if "ORIGIN_CITY" in df.columns:
        df["ORIGIN_CITY_NAME"], df["ORIGIN_STATE"] = split_city_state(df["ORIGIN_CITY"])

    if "DEST_CITY" in df.columns:
        df["DEST_CITY_NAME"], df["DEST_STATE"] = split_city_state(df["DEST_CITY"])

    for col in TIME_COLS:
        if col in df.columns:
            df[f"{col}_MIN"] = df[col].apply(hhmm_to_minutes).astype("Int64")

    if "FL_DATE" in df.columns:
        df["CRS_DEP_DT"] = build_event_datetime(df["FL_DATE"], df["CRS_DEP_TIME_MIN"])
        df["CRS_ARR_DT"] = build_event_datetime(df["FL_DATE"], df["CRS_ARR_TIME_MIN"], df["CRS_DEP_TIME_MIN"])
        df["DEP_DT"] = build_event_datetime(df["FL_DATE"], df["DEP_TIME_MIN"])
        df["ARR_DT"] = build_event_datetime(df["FL_DATE"], df["ARR_TIME_MIN"], df["DEP_TIME_MIN"])

        df["SCHED_BLOCK_MINS"] = (df["CRS_ARR_DT"] - df["CRS_DEP_DT"]).dt.total_seconds() / 60
        df["ACTUAL_BLOCK_MINS"] = (df["ARR_DT"] - df["DEP_DT"]).dt.total_seconds() / 60

    for col, b in winsor_bounds.items():
        if col in df.columns:
            cleaned = pd.to_numeric(df[col], errors="coerce").clip(lower=b["lower"], upper=b["upper"])
            if col in NONNEGATIVE_COLS:
                cleaned = cleaned.clip(lower=0)
            df[f"{col}_CLEAN"] = cleaned

    if "CANCELLED" in df.columns:
        cancelled_mask = df["CANCELLED"] == 1
        for col in ["DEP_DT", "ARR_DT", "DEP_TIME", "ARR_TIME", "DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME"]:
            if col in df.columns:
                df.loc[cancelled_mask, col] = np.nan

    existing_drop = [col for col in DROP_COLS if col in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    if DROP_CANCELLED and "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]

    if DROP_CANCELLED and "DIVERTED" in df.columns:
        df = df[df["DIVERTED"] == 0]

    if DROP_CANCELLED:
        essential = [col for col in ["DEP_TIME", "ARR_TIME", "AIR_TIME", "ELAPSED_TIME", "ARR_DELAY"] if col in df.columns]
        if essential:
            df = df.dropna(subset=essential)

    if "CANCELLATION_CODE" in df.columns:
        df["CANCELLATION_CODE"] = df["CANCELLATION_CODE"].astype("string").fillna("NONE")

    return df


def main():
    winsor_bounds = compute_winsor_bounds(INPUT_CSV)

    missing_before = {}
    missing_after = {}
    input_rows = 0
    output_rows = 0
    cancelled_rows = 0
    diverted_rows = 0
    dropped_cancelled_rows = 0

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE):
        input_rows += len(chunk)

        na_counts = chunk.isna().sum().to_dict()
        for k, v in na_counts.items():
            missing_before[k] = missing_before.get(k, 0) + int(v)

        if "CANCELLED" in chunk.columns:
            cancelled_rows += int(pd.to_numeric(chunk["CANCELLED"], errors="coerce").fillna(0).astype(int).sum())

        if "DIVERTED" in chunk.columns:
            diverted_rows += int(pd.to_numeric(chunk["DIVERTED"], errors="coerce").fillna(0).astype(int).sum())

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    wrote_header = False

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE):
        if DROP_CANCELLED and "CANCELLED" in chunk.columns:
            dropped_cancelled_rows += int(pd.to_numeric(chunk["CANCELLED"], errors="coerce").fillna(0).astype(int).sum())

        cleaned = process_chunk(chunk, winsor_bounds)
        output_rows += len(cleaned)

        na_counts = cleaned.isna().sum().to_dict()
        for k, v in na_counts.items():
            missing_after[k] = missing_after.get(k, 0) + int(v)

        cleaned.to_csv(
            OUTPUT_CSV,
            index=False,
            mode="a",
            header=not wrote_header,
        )
        wrote_header = True

    report = CleaningReport(
        input_rows=int(input_rows),
        output_rows=int(output_rows),
        dropped_cancelled_rows=int(dropped_cancelled_rows) if DROP_CANCELLED else 0,
        missing_before={k: int(v) for k, v in missing_before.items()},
        missing_after={k: int(v) for k, v in missing_after.items()},
        winsor_bounds=winsor_bounds,
        cancelled_rows=int(cancelled_rows),
        diverted_rows=int(diverted_rows),
    )

    REPORT_JSON.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {REPORT_JSON}")


if __name__ == "__main__":
    main()