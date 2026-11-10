"""趋势信号研究用的面板：全市场每个"股票-交易日"的指标 + 之后的结果。

有了它，任何信号定义都只是面板上的一个布尔条件，可以直接和"同一天全市场平均"比较。
    python -m ztfp.trend_lab        # 重建面板，存到 data/panel.parquet
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

from .db import DATA
from .trend import HOLD, STOP, TARGET, _roll, _shift
from .trend_data import connect

PANEL = DATA / "panel.parquet"


def _one(code, k, idx_ret120):
    d = k["date"].to_numpy()
    o, c, h, l, v = (k[x].to_numpy(float) for x in ("open", "close", "high", "low", "volume"))
    amt = k["amount"].to_numpy(float)
    amt = np.where(np.isnan(amt), c * v * 100, amt)
    n = len(c)
    if n < 320:
        return None
    lim = 0.2 if code[:2] in ("30", "68") else 0.1
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = {w: _roll(c, w, "mean") for w in (5, 10, 20, 60, 120, 250)}
        pc = _shift(c, 1)
        hi = {w: _shift(_roll(h, w, "max"), 1) for w in (20, 60, 120, 250)}
        hi500 = _shift(_roll(h, 500, "max", 250), 1)
        lo = {w: _shift(_roll(l, w, "min"), 1) for w in (20, 60, 120)}
        vma50 = _shift(_roll(v, 50, "mean"), 1)
        tr = (h - l) / c
        f = {
            "c": c, "day_pct": c / pc - 1,
            "open_gap": o / pc - 1,
            "close_pos": np.where(h > l, (c - l) / (h - l), 1.0),
            "amt20": _roll(amt, 20, "mean"),
            "vr": v / vma50,
            "v5_50": _roll(v, 5, "mean") / vma50,
            "v20_120": _roll(v, 20, "mean") / _shift(_roll(v, 120, "mean"), 20),
            "atr20": _roll(tr, 20, "mean"), "atr_ratio": _roll(tr, 20, "mean") / _roll(tr, 120, "mean"),
            "dd500": c / hi500 - 1,
            "lim_up": (c / pc - 1 >= lim - 0.003).astype(float),
        }
        for w in (20, 60, 120, 250):
            f[f"brk{w}"] = (c > hi[w]).astype(float)          # 收盘创 w 日新高
            f[f"nh{w}"] = c / hi[w] - 1                         # 相对前高的位置
            f[f"gap{w}"] = c / ma[w] - 1
            f[f"s{w}"] = ma[w] / _shift(ma[w], 20) - 1          # 均线 20 日斜率
            f[f"ret{w}"] = c / _shift(c, w) - 1
        for w in (20, 60, 120):
            f[f"rng{w}"] = (hi[w] - lo[w]) / lo[w]             # 前 w 日振幅（不含当天）
        f["pre120"] = pc / _shift(c, 121) - 1
        f["first120"] = ((c > hi[120]) & ~(_shift(_roll((c > hi[120]).astype(float), 20, "max", 1), 1) > 0)).astype(float)
        f["first250"] = ((c > hi[250]) & ~(_shift(_roll((c > hi[250]).astype(float), 20, "max", 1), 1) > 0)).astype(float)
        f["ma_bull"] = ((c > ma[20]) & (ma[20] > ma[60]) & (ma[60] > ma[120])).astype(float)
        f["x20_60"] = ((ma[20] > ma[60]) & (_shift(ma[20], 1) <= _shift(ma[60], 1))).astype(float)   # 20 日线上穿 60 日线
        f["x_c250"] = ((c > ma[250]) & (pc <= _shift(ma[250], 1))).astype(float)                      # 收盘上穿年线
        f["days_above60"] = _roll((c > ma[60]).astype(float), 60, "sum")
        f["rs120"] = c / _shift(c, 120) - 1 - idx_ret120.reindex(d).to_numpy()

        # 之后的结果：三重障碍 + 固定持有期 + "跌破 20 日线离场"的趋势跟随
        y = np.full(n, np.nan)
        if n > HOLD + 1:
            m = n - HOLD - 1                      # 有完整 60 日窗口的起点个数
            H, L = swv(h[1:], HOLD)[:m], swv(l[1:], HOLD)[:m]
            up = H >= (c[:m] * (1 + TARGET))[:, None]
            dn = L <= (c[:m] * (1 - STOP))[:, None]
            fu = np.where(up.any(1), up.argmax(1), HOLD + 1)
            fd = np.where(dn.any(1), dn.argmax(1), HOLD + 1)
            y[:m] = (fu < fd).astype(float)       # 同一天既到止盈又到止损按失败算
            f["mdd60"] = np.r_[L.min(1) / c[:m] - 1, np.full(n - m, np.nan)]
        f["y"] = y
        for w in (5, 20, 60, 120):
            f[f"f{w}"] = np.r_[c[w:] / c[:-w] - 1, np.full(w, np.nan)]
        # 次日开盘买入的 60 日收益（更接近实际能成交的价格）
        f["f60_open"] = np.r_[c[61:] / o[1:-60] - 1, np.full(61, np.nan)] if n > 61 else np.full(n, np.nan)
        # 趋势跟随离场：收盘跌破 20 日线的当天收盘卖出，最多持有 120 天
        below = c < ma[20]
        tf = np.full(n, np.nan)
        nxt = np.full(n, -1)
        last = -1
        for i in range(n - 1, -1, -1):            # nxt[i] = i 之后第一个跌破 20 日线的下标
            nxt[i] = last
            if below[i]:
                last = i
        for i in range(n):
            j = nxt[i]
            if j != -1 and j - i <= 120:
                tf[i] = c[j] / c[i] - 1
            elif i + 120 < n:
                tf[i] = c[i + 120] / c[i] - 1
        f["tf"] = tf

    df = pd.DataFrame({k_: np.asarray(v_, dtype="float32") for k_, v_ in f.items()})
    df.insert(0, "date", d)
    df.insert(0, "code", code)
    return df[(np.arange(n) >= 260) & (df.amt20 >= 5e7)]


def build(log=print):
    con = connect()
    kl = pd.read_sql("SELECT * FROM k ORDER BY code, date", con)
    stock = set(pd.read_sql("SELECT code FROM stock", con).code)
    con.close()
    idx = kl[kl.code == "IDX"].set_index("date")["close"]
    idx_ret120 = idx / idx.shift(120) - 1
    parts = [p for code, k in kl[kl.code != "IDX"].groupby("code", sort=False)
             if code in stock and (p := _one(code, k, idx_ret120)) is not None]
    df = pd.concat(parts, ignore_index=True)
    # 当天全市场强弱
    g = df.groupby("date")
    df["breadth"] = g["gap60"].transform(lambda s: (s > 0).mean()).astype("float32")
    df["rs_rank"] = g["ret120"].rank(pct=True).astype("float32")      # 120 日涨幅在全市场的分位
    df["idx_gap60"] = df.date.map(idx / idx.rolling(60).mean() - 1).astype("float32")
    # 同一天全市场的平均结果，用作基准
    for col in ("y", "f20", "f60", "f120", "tf", "f60_open"):
        df[f"m_{col}"] = g[col].transform("mean").astype("float32")
    df.to_parquet(PANEL)
    log(f"面板 {len(df)} 行，{df.code.nunique()} 只，{df.date.min()} ~ {df.date.max()}")
    return df


def load():
    return pd.read_parquet(PANEL)


def evaluate(df, mask, name=""):
    """一个信号相对"同一天全市场平均"的超额。返回一行统计。"""
    e = df[mask & df.y.notna()]
    if len(e) < 30:
        return {"信号": name, "n": len(e)}
    by_day = e.groupby("date").agg(x=("f60", "mean"), b=("m_f60", "mean"))
    d = by_day.x - by_day.b
    return {
        "信号": name, "n": len(e), "天数": len(by_day),
        "成功率": e.y.mean(), "基准": e.m_y.mean(), "超额": e.y.mean() - e.m_y.mean(),
        "60日": e.f60.mean(), "60日超额": (e.f60 - e.m_f60).mean(), "60日中位": e.f60.median(),
        "20日超额": (e.f20 - e.m_f20).mean(), "跌破20线离场": e.tf.mean(), "离场超额": (e.tf - e.m_tf).mean(),
        "t": d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 and d.std() > 0 else np.nan,   # 按天聚合后的 t 值
    }


if __name__ == "__main__":
    build()
