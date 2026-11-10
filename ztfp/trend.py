"""趋势突破复盘：长期横盘/阴跌之后，均线拐头并放量突破关键位的股票。

突破信号（某只股票在某天满足全部条件）
  1. 收盘价站上此前 120 个交易日的最高价（突破位），且之前 20 天内没有突破过 —— 是"第一次"突破
  2. 突破前是底部而不是已经涨了一大段：前 120 日涨幅 <= 30%，前 120 日振幅 <= 70%
  3. 均线拐头：收盘 > 20 日线 > 60 日线，60 日线向上，120 日线走平或向上
  4. 放量：当日成交量 >= 前 50 日均量的 1.2 倍
  5. 流动性：20 日日均成交额 >= 5000 万；上市满 250 个交易日；不含 ST
标签（用于回测）
  y = 突破后 60 个交易日内，先涨到 +25%（成功）还是先跌到 -10%（失败）；都没碰到算失败
每日候选 = 最近 30 个交易日内出过信号、还没触发 +25%/-10%、仍站在突破位和 60 日线之上、离 20 日线不太远的股票
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .db import DATA, ROOT
from .trend_data import connect

PIVOT_N = 120
TARGET, STOP, HOLD = 0.25, 0.10, 60
FRESH = 30            # 信号多少个交易日内算"还新鲜"
MAX_EXT = 0.15        # 离 20 日线超过这个比例算"已经拉开"，不再列为候选
STEP, MIN_DAYS = 20, 150

FEATURES = [
    "base_days", "range120", "pre120", "pre250", "dd500", "is250",
    "brk_pct", "day_pct", "close_pos", "vr", "v20_120", "atr20", "turnover",
    "gap20", "gap60", "gap250", "s20", "s60", "s120", "s250", "log_cap",
    "breadth", "idx_gap60", "idx_ret20", "n_ev",
]
FEATURE_CN = {
    "base_days": "底部天数", "range120": "突破前120日振幅", "pre120": "突破前120日涨幅", "pre250": "突破前250日涨幅",
    "dd500": "距两年高点", "is250": "同时创一年新高", "brk_pct": "超出突破位幅度", "day_pct": "突破日涨幅",
    "close_pos": "收盘在当日位置", "vr": "放量倍数", "v20_120": "近20日量/前120日量", "atr20": "20日平均振幅",
    "turnover": "换手率", "gap20": "偏离20日线", "gap60": "偏离60日线", "gap250": "偏离250日线",
    "s20": "20日线斜率", "s60": "60日线斜率", "s120": "120日线斜率", "s250": "250日线斜率", "log_cap": "流通市值",
    "breadth": "全市场站上60日线比例", "idx_gap60": "上证偏离60日线", "idx_ret20": "上证20日涨幅", "n_ev": "当日突破家数",
}


def _roll(a, n, f, minp=None):
    return getattr(pd.Series(a).rolling(n, min_periods=minp or n), f)().to_numpy()


def _shift(a, n):
    out = np.full(len(a), np.nan)
    out[n:] = a[:-n]
    return out


def _barrier(c, h, l, i):
    """从第 i 天收盘买入：60 日内先到 +25% 记 1，先到 -10% 或到期记 0，数据不够记 NaN。返回 (y, 出场下标)。"""
    n, entry = len(c), c[i]
    for t in range(i + 1, min(n, i + 1 + HOLD)):
        if l[t] <= entry * (1 - STOP):
            return 0.0, t
        if h[t] >= entry * (1 + TARGET):
            return 1.0, t
    return (0.0 if i + HOLD < n else np.nan), None


def _scan(code, k):
    """单只股票：返回 (全部历史信号, 当前候选或 None, 每天是否站上60日线, 随机买入基线样本)。"""
    d = k["date"].to_numpy()
    o, c, h, l, v = (k[x].to_numpy(float) for x in ("open", "close", "high", "low", "volume"))
    amt, tov = k["amount"].to_numpy(float), k["turnover"].to_numpy(float)
    amt = np.where(np.isnan(amt), c * v * 100, amt)
    n = len(c)
    ma20, ma60, ma120 = _roll(c, 20, "mean"), _roll(c, 60, "mean"), _roll(c, 120, "mean")
    ma250 = _roll(c, 250, "mean")
    above60 = pd.Series((c > ma60).astype(float), index=d)[~np.isnan(ma60)]
    if n < 260:
        return [], None, above60, []
    with np.errstate(invalid="ignore", divide="ignore"):
        pivot = _shift(_roll(h, PIVOT_N, "max"), 1)
        lo120 = _shift(_roll(l, PIVOT_N, "min"), 1)
        hi250 = _shift(_roll(h, 250, "max"), 1)
        hi500 = _shift(_roll(h, 500, "max", 250), 1)
        brk = c > pivot
        prior = _shift(_roll(brk.astype(float), 20, "max", 1), 1) > 0
        vr = v / _shift(_roll(v, 50, "mean"), 1)
        amt20 = _roll(amt, 20, "mean")
        pc = _shift(c, 1)
        pre120, pre250 = pc / _shift(c, 121) - 1, pc / _shift(c, 251) - 1
        range120 = (pivot - lo120) / lo120
        s20, s60, s120, s250 = (m / _shift(m, 20) - 1 for m in (ma20, ma60, ma120, ma250))
        ok = (brk & ~prior & (pre120 <= 0.30) & (range120 <= 0.70)
              & (c > ma20) & (ma20 > ma60) & (s60 > 0) & (s120 >= -0.02)
              & (vr >= 1.2) & (amt20 >= 5e7) & (np.arange(n) >= 250))
        v20_120 = _roll(v, 20, "mean") / _shift(_roll(v, 120, "mean"), 20)
        atr20 = _roll((h - l) / c, 20, "mean")

    events = []
    for i in np.flatnonzero(ok):
        j = i - 1
        while j >= 0 and c[j] < c[i] and i - j <= 500:   # 上一次收盘比今天高是多少天前
            j -= 1
        entry, fmax = c[i], np.nan
        y, exit_i = _barrier(c, h, l, i)
        # 回踩入场：突破后 2~30 天内、原信号还没出结果时，第一次回到突破位 ±3% 且仍在 60 日线上方
        pb = None
        for t in range(i + 2, min(n, i + 1 + FRESH)):
            if exit_i is not None and t >= exit_i:
                break
            if abs(c[t] / pivot[i] - 1) <= 0.03 and c[t] > ma60[t] and ma20[t] > ma60[t]:
                pb = t
                break
        pb_y = _barrier(c, h, l, pb)[0] if pb is not None else np.nan
        if i + 1 < n:
            fmax = h[i + 1:i + 1 + HOLD].max() / entry - 1
        events.append({
            "code": code, "date": d[i], "i": int(i), "entry": entry, "pivot": pivot[i], "day_low": l[i],
            "base_days": i - j - 1, "range120": range120[i], "pre120": pre120[i], "pre250": pre250[i],
            "dd500": c[i] / hi500[i] - 1, "is250": float(c[i] > hi250[i]),
            "brk_pct": c[i] / pivot[i] - 1, "day_pct": c[i] / pc[i] - 1,
            "close_pos": (c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 1.0,
            "vr": vr[i], "v20_120": v20_120[i], "atr20": atr20[i], "turnover": tov[i],
            "gap20": c[i] / ma20[i] - 1, "gap60": c[i] / ma60[i] - 1, "gap250": c[i] / ma250[i] - 1,
            "s20": s20[i], "s60": s60[i], "s120": s120[i], "s250": s250[i],
            "y": y, "fmax60": fmax, "resolved": exit_i is not None,
            "pb_date": d[pb] if pb is not None else None, "pb_y": pb_y,
            "pb_fret60": c[pb + 60] / c[pb] - 1 if pb is not None and pb + 60 < n else np.nan,
            "fret20": c[i + 20] / entry - 1 if i + 20 < n else np.nan,
            "fret60": c[i + 60] / entry - 1 if i + 60 < n else np.nan,
        })

    # 基线：同一只股票上随机挑 6 天买入，用同样的止盈止损规则，用来判断信号是否比"随便买"强
    rng = np.random.default_rng(int(code))
    pick = rng.choice(np.arange(250, n - HOLD - 1), size=6, replace=False) if n - HOLD - 1 - 250 >= 6 else []
    base = [(d[i], _barrier(c, h, l, i)[0], c[i + HOLD] / c[i] - 1) for i in pick]

    cand = None
    T = n - 1
    live = [e for e in events if T - e["i"] <= FRESH and not e["resolved"]]
    if live:
        e = live[-1]
        if (c[T] >= e["pivot"] * 0.97 and c[T] > ma60[T] and ma20[T] > ma60[T]
                and (e["i"] == T or c[T] / ma20[T] - 1 <= MAX_EXT)):   # 当天刚突破的不按"拉开过远"剔除
            dist = c[T] / e["pivot"] - 1
            status = "今日突破" if e["i"] == T else ("回踩突破位" if abs(dist) <= 0.03 else "突破后上行")
            w = slice(max(0, n - 250), n)
            cand = {
                "code": code, "date": d[T], "sig_date": e["date"], "days": T - e["i"], "status": status,
                "close": c[T], "pivot": e["pivot"], "dist": dist, "gain": c[T] / e["entry"] - 1,
                "ext20": c[T] / ma20[T] - 1, "stop": e["entry"] * (1 - STOP), "target": e["entry"] * (1 + TARGET),
                "day_pct": c[T] / c[T - 1] - 1,
                "chart": json.dumps({"c": np.round(c[w], 3).tolist(),
                                     "m": [None if np.isnan(x) else round(float(x), 3) for x in ma60[w]],
                                     "p": round(float(e["pivot"]), 3), "s": int(e["i"] - (n - 250)) if n >= 250 else e["i"]}),
            }
    return events, cand, above60, base


def scan(log=print):
    con = connect()
    kl = pd.read_sql("SELECT * FROM k ORDER BY code, date", con)
    stock = pd.read_sql("SELECT * FROM stock", con).set_index("code")
    idx = kl[kl.code == "IDX"].set_index("date")["close"]
    latest = idx.index[-1]
    events, cands, base = [], [], []
    above, total = pd.Series(dtype=float), pd.Series(dtype=float)
    for code, k in kl[kl.code != "IDX"].groupby("code", sort=False):
        if code not in stock.index:
            continue
        ev, cand, a60, b = _scan(code, k)
        events += ev
        base += b
        above = above.add(a60, fill_value=0)
        total = total.add(pd.Series(1.0, index=a60.index), fill_value=0)
        if cand and cand["date"] == latest:   # 停牌股的最后一根K不是今天，不算
            cands.append(cand)
    ev = pd.DataFrame(events)
    mk = pd.DataFrame({"breadth": above / total, "idx_gap60": idx / idx.rolling(60).mean() - 1,
                       "idx_ret20": idx / idx.shift(20) - 1})
    mk["n_ev"] = ev.groupby("date").size().reindex(mk.index).fillna(0)
    mk.index.name = "date"
    ev = ev.merge(mk.reset_index(), on="date", how="left")
    ev["log_cap"] = np.log10(ev.code.map(stock.float_cap).astype(float).clip(lower=1))
    ev = ev.sort_values(["date", "code"]).reset_index(drop=True)
    ev.to_sql("event", con, if_exists="replace", index=False)
    mk.reset_index().to_sql("market", con, if_exists="replace", index=False)
    pd.DataFrame(cands).to_sql("cand", con, if_exists="replace", index=False)
    pd.DataFrame(base, columns=["date", "y", "fret60"]).to_sql("baseline", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    lab = ev[ev.y.notna()]
    log(f"历史突破信号 {len(ev)} 个（{ev.date.min()} ~ {ev.date.max()}），已出结果 {len(lab)} 个，"
        f"成功率 {lab.y.mean():.1%}；当前候选 {len(cands)} 只")


def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def train(log=print):
    """滚动前推：每 20 个交易日重训，训练集与预测期之间留出 60 天，保证标签不偷看未来。"""
    con = connect()
    ev = pd.read_sql("SELECT * FROM event", con)
    dates = [r[0] for r in con.execute("SELECT date FROM market ORDER BY date")]
    pos = {d: i for i, d in enumerate(dates)}
    ev["di"] = ev.date.map(pos)
    ev["p"] = np.nan
    first = ev.di.min() + MIN_DAYS
    for start in range(first, len(dates), STEP):
        tr = ev[(ev.di <= start - HOLD - 1) & ev.y.notna()]
        te = ev[(ev.di >= start) & (ev.di < start + STEP)]
        if len(te) and len(tr) >= 500 and tr.y.nunique() == 2:
            m = _clf().fit(tr[FEATURES].astype(float), tr.y.astype(int))
            ev.loc[te.index, "p"] = m.predict_proba(te[FEATURES].astype(float))[:, 1]
    ev.drop(columns=["di"]).to_sql("event", con, if_exists="replace", index=False)
    con.commit()
    con.close()

    t = ev[ev.p.notna() & ev.y.notna()]
    out = {"oos_from": dates[first] if first < len(dates) else None, "metrics": None}
    if len(t) >= 300 and t.y.nunique() == 2:
        q = pd.qcut(t.p, 5, labels=False, duplicates="drop")
        top = t.p >= t.p.quantile(0.9)
        out["metrics"] = {
            "n": int(len(t)), "base": float(t.y.mean()), "auc": float(roc_auc_score(t.y, t.p)),
            "top10_actual": float(t.y[top].mean()),
            "buckets": [{"pred": float(t.p[q == i].mean()), "actual": float(t.y[q == i].mean()),
                         "n": int((q == i).sum()), "fret60": float(t.fret60[q == i].mean())} for i in sorted(q.unique())],
        }
        m = out["metrics"]
        log(f"[趋势] 样本外 n={m['n']} 基准成功率={m['base']:.1%} AUC={m['auc']:.3f} 前10%命中={m['top10_actual']:.1%}")
        log("    五分位 预测→实际: " + "  ".join(f"{b['pred']:.0%}→{b['actual']:.0%}" for b in m["buckets"]))
    (DATA / "trend_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------- 复盘页

def _chart(js):
    d = json.loads(js)
    c = d["c"]
    vals = c + [x for x in d["m"] if x is not None] + [d["p"]]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    X = lambda i: i * 180 / (len(c) - 1)
    Y = lambda v: 42 - (v - lo) / rng * 40
    line = lambda a: " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(a) if v is not None)
    s = max(0, min(len(c) - 1, d["s"]))
    return (f'<svg viewBox="0 0 180 44" width="180" height="44" style="display:block">'
            f'<line x1="0" x2="180" y1="{Y(d["p"]):.1f}" y2="{Y(d["p"]):.1f}" stroke="var(--up)" stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>'
            f'<polyline points="{line(d["m"])}" fill="none" stroke="var(--mut)" stroke-width="1" opacity=".7"/>'
            f'<polyline points="{line(c)}" fill="none" stroke="var(--acc)" stroke-width="1.4"/>'
            f'<circle cx="{X(s):.1f}" cy="{Y(c[s]):.1f}" r="2.6" fill="var(--up)"/></svg>')


def report():
    from html import escape as e

    from .report import CSS, JS, _col, _pct, _spark
    con = connect()
    ev = pd.read_sql("SELECT * FROM event", con)
    mk = pd.read_sql("SELECT * FROM market ORDER BY date", con)
    cand = pd.read_sql("SELECT * FROM cand", con)
    bl = pd.read_sql("SELECT * FROM baseline", con)
    stock = pd.read_sql("SELECT * FROM stock", con).set_index("code")
    con.close()
    met = json.loads((DATA / "trend_metrics.json").read_text())
    date = mk.date.iloc[-1]
    if len(cand):
        cand = cand.merge(ev[["code", "date", "base_days", "dd500", "vr", "is250", "range120"]]
                          .rename(columns={"date": "sig_date"}), on=["code", "sig_date"], how="left")
        cand["name"] = cand.code.map(stock.name)
        cand["industry"] = cand.code.map(stock.industry).fillna("")
        cand["cap"] = cand.code.map(stock.float_cap)
        cand = cand.sort_values(["days", "code"])
    lab = ev[ev.y.notna()]

    # ---- 概览
    cur = mk.iloc[-1]
    h = mk.tail(120)
    recent = lab[lab.date >= mk.date.iloc[-180]] if len(mk) > 180 else lab

    def tile(label, val, sub, series=None):
        return (f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small>'
                f'{_spark(series) if series is not None else ""}</div>')

    def rank(col):
        return _pct((mk[col] < cur[col]).mean(), 0)

    tiles = "".join([
        tile("今日新突破", f"{int(cur.n_ev)} 只", f"高于 2024 年以来 {rank('n_ev')} 的交易日", h.n_ev),
        tile("趋势候选", f"{len(cand)} 只", f"近 {FRESH} 个交易日突破且仍在趋势中"),
        tile("全市场站上 60 日线", _pct(cur.breadth, 0), f"高于 {rank('breadth')} 的交易日", h.breadth),
        tile("上证偏离 60 日线", _pct(cur.idx_gap60, 1, True), f"高于 {rank('idx_gap60')} 的交易日", h.idx_gap60),
        tile("近期信号成功率", _pct(recent.y.mean(), 0), f"近 {len(recent)} 个已出结果的信号；全部历史 {_pct(lab.y.mean(), 0)}，随机买入 {_pct(bl.y.mean(), 0)}"),
    ])

    # ---- 候选表
    def table(df):
        if not len(df):
            return '<div class="note">没有符合条件的股票。</div>'
        rows = ""
        for _, r in df.iterrows():
            tag = '<span class="tag">一年新高</span>' if r.is250 == 1 else ""
            rows += (
                f'<tr><td class="l" data-v="{e(r["name"])}">{e(r["name"])}<span class="code">{r.code}</span>{tag}</td>'
                f'<td class="l" data-v="{e(r.industry)}">{e(r.industry)}</td><td class="l" data-v="{e(r.status)}">{e(r.status)}</td>'
                f'<td data-v="{r.days}">{r.sig_date[4:6]}-{r.sig_date[6:]}<span class="code">{r.days}天前</span></td>'
                f'<td data-v="{r.close}">{r.close:.2f}</td><td data-v="{r.dist}" class="{_col(r.dist)}">{_pct(r.dist, 1, True)}</td>'
                f'<td data-v="{r.gain}" class="{_col(r.gain)}">{_pct(r.gain, 1, True)}</td>'
                f'<td data-v="{r.ext20}">{_pct(r.ext20, 1, True)}</td>'
                f'<td data-v="{r.base_days}">{int(r.base_days)}{"+" if r.base_days >= 500 else ""}</td>'
                f'<td data-v="{r.dd500}">{_pct(r.dd500, 0, True)}</td><td data-v="{r.vr}">{r.vr:.1f}倍</td>'
                f'<td data-v="{r.cap}">{r.cap / 1e8:.0f}亿</td>'
                f'<td data-v="{r.stop}">{r.stop:.2f} / {r.target:.2f}</td><td>{_chart(r.chart)}</td></tr>')
        return ('<div class="card scroll"><table><thead><tr><th class="l s">股票</th><th class="l s">行业</th><th class="l s">状态</th>'
                '<th class="s">突破日</th><th class="s">现价</th><th class="s">距突破位</th><th class="s">突破后涨幅</th>'
                '<th class="s">偏离20日线</th><th class="s">底部天数</th><th class="s">距两年高点</th><th class="s">突破日放量</th>'
                '<th class="s">流通市值</th><th class="s">-10% / +25% 价位</th><th class="l">近一年走势</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>')

    new = cand[cand.status == "今日突破"] if len(cand) else cand
    pull = cand[cand.status == "回踩突破位"] if len(cand) else cand
    run = cand[cand.status == "突破后上行"] if len(cand) else cand
    inds = ""
    if len(cand):
        vc = cand[cand.industry != ""].industry.value_counts().head(12)
        inds = "".join(f'<span class="chip">{e(k)}<em>{v}</em></span>' for k, v in vc.items())

    # ---- 历史统计
    def grp(key, title, order=None):
        g = lab.groupby(key, observed=True).agg(n=("code", "size"), y=("y", "mean"), f20=("fret20", "mean"),
                                                f60=("fret60", "mean"), mx=("fmax60", "median"))
        if order:
            g = g.reindex([x for x in order if x in g.index])
        rows = "".join(f'<tr><td class="l">{e(str(k))}</td><td>{int(r.n)}</td><td><b>{_pct(r.y)}</b></td>'
                       f'<td class="{_col(r.f20)}">{_pct(r.f20, 1, True)}</td><td class="{_col(r.f60)}">{_pct(r.f60, 1, True)}</td>'
                       f'<td>{_pct(r.mx, 0, True)}</td></tr>' for k, r in g.iterrows())
        return (f'<div class="card scroll"><h3>{title}</h3><table><thead><tr><th class="l"></th><th>信号数</th><th>成功率</th>'
                f'<th>20日后平均</th><th>60日后平均</th><th>60日内最高涨幅中位</th></tr></thead><tbody>{rows}</tbody></table></div>')

    lab = lab.copy()
    lab["底部天数"] = pd.cut(lab.base_days, [-1, 180, 250, 375, 499, 9999],
                         labels=["120–180 天", "180–250 天", "250–375 天", "375–500 天", "500 天以上（两年新高）"])
    lab["放量"] = pd.cut(lab.vr, [0, 1.5, 2, 3, 5, 999], labels=["1.2–1.5 倍", "1.5–2 倍", "2–3 倍", "3–5 倍", "5 倍以上"])
    lab["位置"] = pd.cut(lab.dd500, [-9, -0.5, -0.35, -0.2, -0.05, 9],
                       labels=["较两年高点跌 50% 以上", "跌 35–50%", "跌 20–35%", "跌 5–20%", "接近或创两年新高"])
    lab["市场"] = pd.qcut(lab.breadth, 3, labels=["市场弱（站上60日线的股票少）", "市场中性", "市场强"])
    lab["突破日"] = pd.cut(lab.day_pct, [-1, 0.03, 0.06, 0.095, 1], labels=["涨 3% 以内", "涨 3–6%", "涨 6–9.5%", "涨停或接近涨停"])
    lab["年份"] = lab.date.str[:4] + " 年"
    stats = "".join([grp("底部天数", "按底部长度（多久没到过这个价）"), grp("位置", "按距两年高点的位置"),
                     grp("放量", "按突破日放量倍数"), grp("突破日", "按突破日涨幅"),
                     grp("市场", "按当时的市场强弱"), grp("年份", "按年份")])

    # ---- 市场状态：同样的买法，在不同市场宽度下的成功率（突破信号 vs 随机买入）
    bins = [0, .2, .35, .5, .65, .8, 1.01]
    names = ["20% 以下", "20–35%", "35–50%", "50–65%", "65–80%", "80% 以上"]
    br = mk.set_index("date").breadth
    blm = bl[bl.y.notna()].assign(b=lambda x: pd.cut(x.date.map(br), bins, labels=names))
    evm = lab.assign(b=pd.cut(lab.breadth, bins, labels=names))
    now_b = pd.cut([cur.breadth], bins, labels=names)[0]
    rows = ""
    for nm in names:
        a, b_ = evm[evm.b == nm], blm[blm.b == nm]
        mark = ' style="background:var(--bar2)"' if nm == now_b else ""
        rows += (f'<tr{mark}><td class="l">{nm}{"（今天）" if nm == now_b else ""}</td><td>{len(a)}</td><td><b>{_pct(a.y.mean())}</b></td>'
                 f'<td class="{_col(a.fret60.mean())}">{_pct(a.fret60.mean(), 1, True)}</td><td>{len(b_)}</td><td><b>{_pct(b_.y.mean())}</b></td>'
                 f'<td class="{_col(b_.fret60.mean())}">{_pct(b_.fret60.mean(), 1, True)}</td></tr>')
    mkt = ('<div class="card scroll" style="max-width:820px"><table><thead><tr><th class="l">全市场站上 60 日线的比例</th>'
           '<th>突破信号数</th><th>成功率</th><th>60日后平均</th><th>随机买入样本</th><th>成功率</th><th>60日后平均</th></tr></thead>'
           f'<tbody>{rows}</tbody></table></div>')

    # ---- 这个形态到底有没有用：和随机买入比
    pb = ev[ev.pb_y.notna()]
    bl = bl[bl.date >= lab.date.min()]

    def cmp_row(name, n, y, f60):
        return (f'<tr><td class="l">{name}</td><td>{n}</td><td><b>{_pct(y.mean())}</b></td>'
                f'<td class="{_col(f60.mean())}">{_pct(f60.mean(), 1, True)}</td><td class="{_col(f60.median())}">{_pct(f60.median(), 1, True)}</td></tr>')

    auc = (met.get("metrics") or {}).get("auc")
    model = ('<div class="card scroll" style="max-width:720px"><table><thead><tr><th class="l">买入方式</th><th>样本</th><th>成功率</th>'
             '<th>60日后平均</th><th>60日后中位</th></tr></thead><tbody>'
             + cmp_row("突破当天收盘买入", len(lab), lab.y, lab.fret60)
             + cmp_row("等回踩到突破位 ±3% 再买", len(pb), pb.pb_y, pb.pb_fret60)
             + cmp_row("同期随机挑股票、随机挑日子买入", len(bl), bl.y, bl.fret60)
             + '</tbody></table></div>'
             + (f'<div class="note">另外试过用模型给每个信号打成功概率，样本外 AUC 只有 {auc:.3f}（0.5 等于瞎猜），所以页面上不显示个股概率。</div>' if auc else ""))

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>趋势复盘 {date}</title><style>{CSS}</style></head><body><main>
<h1>趋势突破复盘 · {date[:4]}-{date[4:6]}-{date[6:]}</h1>
<div class="sub">扫描 {len(stock)} 只沪深 A 股（不含 ST 与北交所） · 历史信号 {len(ev)} 个（{ev.date.min()} 起）</div>
<div class="tiles">{tiles}</div>
<div class="note" style="margin-top:14px"><b>信号</b>：收盘首次站上此前 120 日最高价；突破前 120 日涨幅不超过 30%、振幅不超过 70%；
收盘 &gt; 20 日线 &gt; 60 日线且 60 日线向上、120 日线走平或向上；当日放量 1.2 倍以上；日均成交额 5000 万以上。<br>
<b>成功</b>：突破后 60 个交易日内先涨到 +25%；先跌到 -10% 或到期都没碰到算失败。<br>
走势图：蓝线收盘价，灰线 60 日线，红虚线突破位，红点突破日。</div>

<h2>候选的行业分布</h2><div class="card"><div class="chips">{inds or "无"}</div></div>

<h2>今日新突破（{len(new)}）</h2>{table(new)}

<h2>回踩突破位（{len(pull)}）</h2>
<div class="note">突破后回落到突破位 ±3% 以内，仍守在 60 日线上方。</div>{table(pull)}

<h2>突破后上行中（{len(run)}）</h2>
<div class="note">已离开突破位，但还没到 +25%，离 20 日线不超过 {MAX_EXT:.0%}。</div>{table(run)}

<h2>历史统计（{lab.date.min()} ~ {lab.date.max()}，{len(lab)} 个已出结果的信号）</h2><div class="grid2">{stats}</div>

<h2>市场状态与成功率</h2>
<div class="note">买入时全市场有多少股票站在 60 日线上方。历史上市场越弱（比例越低）时随机买入，之后 60 天的结果反而越好，
这在 2025 年 8 月前后两段时间里方向一致，但只经历过两种行情，不能当成定律。突破信号没有这个特点：弱市里的突破明显不如随机买入。</div>{mkt}

<h2>这个形态的历史表现</h2>
<div class="note">三种买法都用同一套规则（+25% 止盈、-10% 止损、最多拿 60 个交易日）。如果突破信号不比随机买入强，
这张候选表就只是帮你把符合形态的股票找出来，本身不代表胜率更高。</div>{model}
<div class="note" style="margin-top:24px">数据来源：东方财富日K（前复权）。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"trend_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path
