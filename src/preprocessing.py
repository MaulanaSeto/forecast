"""
preprocessing.py — Data preprocessing for IDX Stock Forecasting (TFT).

This module handles ALL data preprocessing:
  - Loading train/test CSVs and metadata
  - Extracting past features (100 target tickers' ret + vol + 4 market aggregates)
  - Engineering time-based future features (hour/dow sin/cos, sesi, menit_dalam_sesi)
  - Building static features from metadata (sektor_id, board_id, market_cap_bin)
  - Clipping outliers, handling inf/NaN, normalizing with StandardScaler
  - Returning neatly packaged numpy arrays ready for model consumption
"""

import os
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore

# 100 target tickers (alphabetical, as given in the competition spec)
TARGET_TICKERS: list[str] = [
    "AALI",
    "ABMM",
    "ACES",
    "ADMR",
    "ADRO",
    "AGRO",
    "AKRA",
    "AMRT",
    "ANTM",
    "ARCI",
    "ARNA",
    "ARTO",
    "ASII",
    "AVIA",
    "BANK",
    "BBCA",
    "BBHI",
    "BBKP",
    "BBNI",
    "BBRI",
    "BBTN",
    "BDMN",
    "BELI",
    "BFIN",
    "BJBR",
    "BJTM",
    "BMRI",
    "BNGA",
    "BRIS",
    "BRMS",
    "BRPT",
    "BSDE",
    "BSSR",
    "BTPS",
    "BUKA",
    "BUMI",
    "CARE",
    "CMRY",
    "CPIN",
    "CTRA",
    "DMAS",
    "DMMX",
    "EMTK",
    "ESSA",
    "EXCL",
    "FILM",
    "FREN",
    "GGRM",
    "GOTO",
    "HEAL",
    "HMSP",
    "ICBP",
    "INCO",
    "INDF",
    "INDY",
    "INKP",
    "INTP",
    "ISAT",
    "ITMG",
    "JPFA",
    "JSMR",
    "KLBF",
    "LPPF",
    "MAPI",
    "MBAP",
    "MCOL",
    "MDKA",
    "MEDC",
    "MIKA",
    "MNCN",
    "MORA",
    "MSIN",
    "MTEL",
    "MYOR",
    "PGAS",
    "PNBN",
    "PNLF",
    "POWR",
    "PRAY",
    "PTBA",
    "PWON",
    "SCMA",
    "SIDO",
    "SMGR",
    "SMRA",
    "SRTG",
    "SSMS",
    "STAA",
    "TAPG",
    "TBIG",
    "TCPI",
    "TINS",
    "TKIM",
    "TLKM",
    "TMAS",
    "TOWR",
    "TPIA",
    "UNTR",
    "UNVR",
    "WSKT",
]


# Helper: market-cap binning
def get_market_cap_bin(market_cap: float) -> int:
    """Bin a market-cap value into Small / Mid / Large.

    Parameters
    ----------
    market_cap : float
        Market capitalisation in IDR.

    Returns
    -------
    int
        0 = Small  (< 2 trillion IDR)
        1 = Mid    (2-20 trillion IDR)
        2 = Large  (> 20 trillion IDR)
    """
    if market_cap < 2e12:
        return 0  # Small
    if market_cap <= 20e12:
        return 1  # Mid
    return 2  # Large


# Internal helpers
def _discover_all_tickers(columns: list[str]) -> list[str]:
    """Return a sorted list of ALL unique ticker symbols found in the columns.

    Looks for columns matching ``{TICKER}_ret`` and extracts the ticker part.
    """
    tickers = sorted({col.rsplit("_", 1)[0] for col in columns if col.endswith("_ret")})
    return tickers


def _build_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """Create 6 known-future time features from a timestamp Series.

    Features
    --------
    hour_sin, hour_cos : float
        Sine / cosine encoding of the hour (period = 24 h).
    dow_sin, dow_cos : float
        Sine / cosine encoding of the day-of-week (period = 7 d).
    sesi : int
        Trading session indicator.
        0 = pre-open / outside hours,
        1 = session 1 (09:00-11:29),
        2 = session 2 (13:30-15:59).
    menit_dalam_sesi : float
        Minutes elapsed since the start of the current session (0 if sesi == 0).
    """
    ts = pd.to_datetime(timestamps)
    hour = ts.dt.hour + ts.dt.minute / 60.0  # fractional hour

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    dow = ts.dt.dayofweek.astype(float)  # Monday=0 … Friday=4
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # Session assignment
    total_minutes = ts.dt.hour * 60 + ts.dt.minute
    sesi = np.zeros(len(ts), dtype=np.int32)
    menit_dalam_sesi = np.zeros(len(ts), dtype=np.float64)

    # Session 1: 09:00 – 11:29  →  540 – 689 minutes
    mask_s1 = (total_minutes >= 540) & (total_minutes < 690)
    sesi[mask_s1] = 1
    menit_dalam_sesi[mask_s1] = (total_minutes[mask_s1] - 540).astype(float)

    # Session 2: 13:30 – 15:59  →  810 – 959 minutes
    mask_s2 = (total_minutes >= 810) & (total_minutes < 960)
    sesi[mask_s2] = 2
    menit_dalam_sesi[mask_s2] = (total_minutes[mask_s2] - 810).astype(float)

    return pd.DataFrame(
        {
            "hour_sin": hour_sin.values if hasattr(hour_sin, "values") else hour_sin,
            "hour_cos": hour_cos.values if hasattr(hour_cos, "values") else hour_cos,
            "dow_sin": dow_sin.values if hasattr(dow_sin, "values") else dow_sin,
            "dow_cos": dow_cos.values if hasattr(dow_cos, "values") else dow_cos,
            "sesi": sesi,
            "menit_dalam_sesi": menit_dalam_sesi,
        }
    )


def _build_static_features(
    metadata: pd.DataFrame,
    target_tickers: list[str],
) -> np.ndarray:
    """Build a (100, 3) static feature matrix from metadata.

    Features per ticker: sektor_id, board_id, market_cap_bin.
    """
    # Ensure metadata is indexed by Code for easy lookup
    meta = metadata.copy()
    meta["Code"] = meta["Code"].str.strip()
    meta = meta.set_index("Code")

    # Label-encode Sector
    sector_encoder = LabelEncoder()
    all_sectors = meta["Sector"].fillna("Unknown").values
    sector_encoder.fit(all_sectors)

    # Label-encode ListingBoard
    board_encoder = LabelEncoder()
    all_boards = meta["ListingBoard"].fillna("Unknown").values
    board_encoder.fit(all_boards)

    static = np.zeros((len(target_tickers), 3), dtype=np.float32)

    for i, ticker in enumerate(target_tickers):
        if ticker in meta.index:
            row = meta.loc[ticker]
            sector = row["Sector"] if pd.notna(row["Sector"]) else "Unknown"
            board = row["ListingBoard"] if pd.notna(row["ListingBoard"]) else "Unknown"
            mcap = row["MarketCap"] if pd.notna(row["MarketCap"]) else 0.0

            static[i, 0] = sector_encoder.transform([sector])[0]
            static[i, 1] = board_encoder.transform([board])[0]
            static[i, 2] = get_market_cap_bin(float(mcap))
        else:
            # Ticker not found in metadata — use zeros (unknown)
            static[i, :] = 0.0

    return static


def _compute_market_aggregates(
    df: pd.DataFrame,
    all_tickers: list[str],
) -> pd.DataFrame:
    """Compute 4 market-wide aggregate features from ALL 787 tickers.

    Features
    --------
    market_ret_mean : mean of all tickers' returns at each timestamp
    market_ret_std  : std of all tickers' returns at each timestamp
    market_vol_total: sum of all tickers' volumes at each timestamp
    n_aktif         : number of tickers with non-zero, non-NaN return
    """
    ret_cols = [f"{t}_ret" for t in all_tickers if f"{t}_ret" in df.columns]
    vol_cols = [f"{t}_vol" for t in all_tickers if f"{t}_vol" in df.columns]

    ret_block = df[ret_cols].values.astype(np.float64)
    vol_block = df[vol_cols].values.astype(np.float64)

    # Replace inf with NaN for safe aggregation
    ret_block[~np.isfinite(ret_block)] = np.nan
    vol_block[~np.isfinite(vol_block)] = np.nan

    market_ret_mean = np.nanmean(ret_block, axis=1)
    market_ret_std = np.nanstd(ret_block, axis=1)
    market_vol_total = np.nansum(vol_block, axis=1)
    n_aktif = np.sum(
        np.isfinite(ret_block) & (ret_block != 0.0),
        axis=1,
    ).astype(np.float64)

    return pd.DataFrame(
        {
            "market_ret_mean": market_ret_mean,
            "market_ret_std": market_ret_std,
            "market_vol_total": market_vol_total,
            "n_aktif": n_aktif,
        },
        index=df.index,
    )


# Main entry point
def preprocess_data(
    data_dir: str = "dataset",
    lookback: int = 60,
) -> dict[str, Any]:
    # pylint: disable=too-many-local-variables,too-many-statements,unused-argument
    """
    Perform end-to-end preprocessing of IDX stock data.

    This function loads train, test, and metadata files, extracts features,
    normalizes them, and packages them into a dictionary for the model.

    Args:
        data_dir: Path to the directory containing dataset files.
        lookback: Number of past timesteps (currently handled via data split).

    Returns:
        A dictionary containing all processed feature matrices and metadata.
    """
    # -- 1. Load raw CSVs -------------------------------------------------
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    # Sort chronologically to prevent rolling window mismatch
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    # -- 2. Discover tickers ----------------------------------------------
    all_tickers = _discover_all_tickers(train_df.columns.tolist())
    target_tickers = TARGET_TICKERS

    # -- 3. Extract target matrix -----------------------------------------
    target_cols = [f"{t}_target" for t in target_tickers]
    train_targets = train_df[target_cols].values.astype(np.float32)
    train_targets[~np.isfinite(train_targets)] = 0.0

    # -- 4. Compute market aggregates ------------------------------------
    train_market = _compute_market_aggregates(train_df, all_tickers)
    test_market = _compute_market_aggregates(test_df, all_tickers)

    # -- 5. Build past-feature matrices ----------------------------------
    ret_cols_target = [f"{t}_ret" for t in target_tickers]
    vol_cols_target = [f"{t}_vol" for t in target_tickers]
    past_ticker_cols = ret_cols_target + vol_cols_target
    train_past_ticker = train_df[past_ticker_cols].values.astype(np.float64)
    test_past_ticker = test_df[past_ticker_cols].values.astype(np.float64)
    train_past_ticker[~np.isfinite(train_past_ticker)] = np.nan
    test_past_ticker[~np.isfinite(test_past_ticker)] = np.nan

    # -- 6. Clip outliers ------------------------------------------------
    q_low = np.nanpercentile(train_past_ticker, 1, axis=0)
    q_high = np.nanpercentile(train_past_ticker, 99, axis=0)
    train_past_ticker = np.clip(train_past_ticker, q_low[None, :], q_high[None, :])
    test_past_ticker = np.clip(test_past_ticker, q_low[None, :], q_high[None, :])
    train_past_ticker = np.nan_to_num(train_past_ticker, nan=0.0)
    test_past_ticker = np.nan_to_num(test_past_ticker, nan=0.0)

    # -- 7. Normalise returns --------------------------------------------
    n_ret, n_vol = len(ret_cols_target), len(vol_cols_target)
    ret_scaler = StandardScaler()
    ret_scaler.fit(train_past_ticker[:, :n_ret])
    train_past_ticker[:, :n_ret] = ret_scaler.transform(train_past_ticker[:, :n_ret])
    test_past_ticker[:, :n_ret] = ret_scaler.transform(test_past_ticker[:, :n_ret])

    # -- 8. Transform volumes --------------------------------------------
    train_past_ticker[:, n_ret : n_ret + n_vol] = np.log1p(
        np.abs(train_past_ticker[:, n_ret : n_ret + n_vol])
    )
    test_past_ticker[:, n_ret : n_ret + n_vol] = np.log1p(
        np.abs(test_past_ticker[:, n_ret : n_ret + n_vol])
    )
    vol_scaler = StandardScaler()
    vol_scaler.fit(train_past_ticker[:, n_ret : n_ret + n_vol])
    train_past_ticker[:, n_ret : n_ret + n_vol] = vol_scaler.transform(
        train_past_ticker[:, n_ret : n_ret + n_vol]
    )
    test_past_ticker[:, n_ret : n_ret + n_vol] = vol_scaler.transform(
        test_past_ticker[:, n_ret : n_ret + n_vol]
    )

    # -- 9. Normalise market aggregates ----------------------------------
    market_cols_names = [
        "market_ret_mean",
        "market_ret_std",
        "market_vol_total",
        "n_aktif",
    ]
    train_mkt = train_market[market_cols_names].values.astype(np.float64)
    test_mkt = test_market[market_cols_names].values.astype(np.float64)
    train_mkt[~np.isfinite(train_mkt)] = np.nan
    test_mkt[~np.isfinite(test_mkt)] = np.nan
    m_q_low = np.nanpercentile(train_mkt, 1, axis=0)
    m_q_high = np.nanpercentile(train_mkt, 99, axis=0)
    train_mkt = np.clip(train_mkt, m_q_low[None, :], m_q_high[None, :])
    test_mkt = np.clip(test_mkt, m_q_low[None, :], m_q_high[None, :])
    train_mkt = np.nan_to_num(train_mkt, nan=0.0)
    test_mkt = np.nan_to_num(test_mkt, nan=0.0)
    train_mkt[:, 2] = np.log1p(np.abs(train_mkt[:, 2]))
    test_mkt[:, 2] = np.log1p(np.abs(test_mkt[:, 2]))
    m_scaler = StandardScaler()
    m_scaler.fit(train_mkt)
    train_mkt = m_scaler.transform(train_mkt)
    test_mkt = m_scaler.transform(test_mkt)

    # -- 10. Concatenate features ----------------------------------------
    train_past = np.concatenate([train_past_ticker, train_mkt], axis=1).astype(
        np.float32
    )
    test_past = np.concatenate([test_past_ticker, test_mkt], axis=1).astype(np.float32)
    past_cols = past_ticker_cols + market_cols_names

    # -- 11. Build time/static features ----------------------------------
    train_future = _build_time_features(
        cast(pd.Series, train_df["timestamp"])
    ).values.astype(np.float32)
    test_future = _build_time_features(
        cast(pd.Series, test_df["timestamp"])
    ).values.astype(np.float32)
    future_cols = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "sesi",
        "menit_dalam_sesi",
    ]
    static_matrix = _build_static_features(metadata, target_tickers)
    test_timestamps = pd.to_datetime(test_df["timestamp"]).values

    n_train, n_test = train_past.shape[0], test_past.shape[0]
    msg = (
        f"  Data: {n_train} train | {n_test} test | "
        f"{train_past.shape[1]} past, {len(future_cols)} future, "
        f"{static_matrix.shape[1]} static"
    )
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)

    return {
        "train_past": train_past,
        "train_future": train_future,
        "train_targets": train_targets,
        "test_past": test_past,
        "test_future": test_future,
        "static_matrix": static_matrix,
        "target_tickers": target_tickers,
        "past_cols": past_cols,
        "future_cols": future_cols,
        "n_train": n_train,
        "n_test": n_test,
        "test_timestamps": test_timestamps,
        "ret_scaler": ret_scaler,
        "vol_scaler": vol_scaler,
        "data_dir": data_dir,
    }


# Quick smoke test when run directly
if __name__ == "__main__":
    result = preprocess_data()
    print("\n=== Preprocessing result keys ===")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}  shape={v.shape}  dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k:20s}  len={len(v)}")
        else:
            print(f"  {k:20s}  {v}")
