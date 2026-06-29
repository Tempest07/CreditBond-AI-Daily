from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_dataset(out_path: str | Path, days: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    t = np.arange(days)

    pmi = 50 + 1.2 * np.sin(t / 70) + rng.normal(0, 0.35, days)
    cpi = 1.8 + 0.5 * np.sin(t / 130 + 1.2) + rng.normal(0, 0.08, days)
    social_financing = 10 + 1.5 * np.sin(t / 95 - 0.6) + rng.normal(0, 0.4, days)
    usd_index = 100 + np.cumsum(rng.normal(0, 0.08, days))
    treasury_1y_yield = 2.1 + np.cumsum(rng.normal(0, 0.012, days))

    risk_pressure = (
        0.35 * (treasury_1y_yield - pd.Series(treasury_1y_yield).rolling(20, min_periods=1).mean().to_numpy())
        - 0.025 * (pmi - 50)
        + 0.018 * (usd_index - pd.Series(usd_index).rolling(40, min_periods=1).mean().to_numpy())
        + rng.normal(0, 0.018, days)
    )
    credit_delta = 0.35 * risk_pressure + rng.normal(0, 0.018, days)
    credit_aaa_1y_yield = 2.8 + np.cumsum(credit_delta)
    spread_aa = 0.45 + 0.08 * np.sin(t / 110) + rng.normal(0, 0.025, days)
    credit_aa_1y_yield = credit_aaa_1y_yield + np.maximum(spread_aa, 0.1)
    credit_trade_count = 1200 + 180 * np.sin(t / 35) + 35 * np.abs(risk_pressure) + rng.normal(0, 50, days)

    df = pd.DataFrame(
        {
            "date": dates,
            "treasury_1y_yield": treasury_1y_yield,
            "credit_aaa_1y_yield": credit_aaa_1y_yield,
            "credit_aa_1y_yield": credit_aa_1y_yield,
            "pmi": pmi,
            "cpi": cpi,
            "social_financing": social_financing,
            "usd_index": usd_index,
            "credit_trade_count": credit_trade_count,
        }
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df
