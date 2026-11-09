"""把涨停池记录加工成"特征 + 标签"宽表（表名 feat，一行 = 某天的一只涨停股）。

标签
  y_lb     次日是否继续涨停（出现在下一交易日涨停池）
  y_swing  波段：之后 20 个交易日内最高价较涨停收盘价涨 >= 20%，
           且在达到之前没有跌破"起涨点"（涨停日最低价与前收盘价的较低者）
特征全部只用到涨停当日收盘为止的信息。
"""
import numpy as np
import pandas as pd

from .db import connect

SWING_DAYS = 20
SWING_GAIN = 0.20

FEATURES = [
    # 个股封板质量
    "lb", "days_n", "boards_n", "first_min", "last_min", "open_num",
    "seal_ratio", "seal_vs_amt", "turnover_rate", "log_cap", "type_code", "is20",
    "suc_rate", "is_again_limit",
    # 题材
    "blk_num", "blk_days", "tag_cnt", "n_tags",
    # 位置/量价
    "pos60", "ret5_prev", "ret20_prev", "ma20_gap", "ma60_gap", "vol_ratio",
    # 市场情绪
    "n_zt", "n_dt", "zb_rate", "max_lb", "n_lb2", "idx_pct", "prem_prev", "promo", "promo_lb",
]

FEATURE_CN = {
    "lb": "连板数", "days_n": "N天", "boards_n": "N板", "first_min": "首封时间", "last_min": "末封时间",
    "open_num": "开板次数", "seal_ratio": "封单/流通市值", "seal_vs_amt": "封单/成交额",
    "turnover_rate": "换手率", "log_cap": "流通市值", "type_code": "涨停类型", "is20": "20cm",
    "suc_rate": "历史封板率", "is_again_limit": "回封", "blk_num": "所属题材涨停数",
    "blk_days": "题材持续天数", "tag_cnt": "同原因涨停数", "n_tags": "原因标签数",
    "pos60": "距60日高点", "ret5_prev": "涨停前5日涨幅", "ret20_prev": "涨停前20日涨幅",
    "ma20_gap": "偏离20日线", "ma60_gap": "偏离60日线", "vol_ratio": "量比(对前5日)",
    "n_zt": "全市场涨停数", "n_dt": "跌停数", "zb_rate": "炸板率", "max_lb": "最高板",
    "n_lb2": "连板家数", "idx_pct": "上证涨跌", "prem_prev": "昨日涨停溢价",
    "promo": "昨日涨停晋级率", "promo_lb": "昨日连板晋级率",
}


def _kline_features(k):
    """逐股计算位置特征和前向收益。k: 单只股票按日期升序。"""
    c, h, l, o, v = (k[x].to_numpy(float) for x in ("close", "high", "low", "open", "volume"))
    n = len(c)
    s = pd.Series
    pc = s(c).shift(1)
    out = pd.DataFrame({"date": k["date"].to_numpy()})
    out["pct"] = c / pc - 1
    out["prev_close"] = pc
    out["pos60"] = c / s(h).rolling(60, min_periods=20).max() - 1
    out["ret5_prev"] = pc / s(c).shift(6) - 1
    out["ret20_prev"] = pc / s(c).shift(21) - 1
    out["ma20_gap"] = c / s(c).rolling(20, min_periods=10).mean() - 1
    out["ma60_gap"] = c / s(c).rolling(60, min_periods=30).mean() - 1
    out["vol_ratio"] = v / s(v).shift(1).rolling(5, min_periods=3).mean()
    out["amount"] = k["amount"].to_numpy(float)
    out["k_low"], out["k_close"] = l, c
    # 次日表现（相对涨停收盘价）
    for name, arr in (("n_open", o), ("n_high", h), ("n_low", l), ("n_close", c)):
        out[name] = s(arr).shift(-1) / c - 1
    # 前向波段
    fmax = {5: np.full(n, np.nan), 10: np.full(n, np.nan), 20: np.full(n, np.nan)}
    fret = {5: np.full(n, np.nan), 10: np.full(n, np.nan), 20: np.full(n, np.nan)}
    swing = np.full(n, np.nan)
    pcv = pc.to_numpy()
    for i in range(n):
        for w in fmax:
            if i + w < n:
                fmax[w][i] = h[i + 1:i + 1 + w].max() / c[i] - 1
                fret[w][i] = c[i + w] / c[i] - 1
        floor = np.nanmin([l[i], pcv[i]])
        res = np.nan
        for j in range(i + 1, min(n, i + 1 + SWING_DAYS)):
            if l[j] < floor:      # 先破位算失败（同一天既破位又达标也按失败，偏保守）
                res = 0.0
                break
            if h[j] >= c[i] * (1 + SWING_GAIN):
                res = 1.0
                break
        if np.isnan(res) and i + SWING_DAYS < n:
            res = 0.0             # 窗口走完都没达标
        swing[i] = res
    for w in fmax:
        out[f"fmax{w}"], out[f"fret{w}"] = fmax[w], fret[w]
    out["y_swing"] = swing
    return out


def build(log=print):
    con = connect()
    cal = [r[0] for r in con.execute("SELECT date FROM days_done ORDER BY date")]
    nxt = dict(zip(cal, cal[1:]))
    prv = dict(zip(cal[1:], cal))
    pool = pd.read_sql("SELECT * FROM pool", con)
    zt = pool[pool.kind == "zt"].copy()

    # ---- 连板数：优先同花顺天梯，缺失时按"昨天是否在涨停池"自行递推
    ladder = pd.read_sql("SELECT * FROM ladder", con).set_index(["date", "code"])["continue_num"]
    by_day = {d: set(g.code) for d, g in zt.groupby("date")}
    lb_self, last = {}, {}
    for d in cal:
        cur = {c: last.get(c, 0) + 1 for c in by_day.get(d, ())}
        lb_self.update({(d, c): n for c, n in cur.items()})
        last = cur
    key = list(zip(zt.date, zt.code))
    zt["lb"] = [ladder.get(k_, np.nan) for k_ in key]
    zt["lb"] = zt["lb"].fillna(pd.Series([lb_self[k_] for k_ in key], index=zt.index)).astype(int)

    # ---- 标签：次日是否连板
    zt["y_lb"] = [float(c in by_day.get(nxt[d], ())) if d in nxt else np.nan for d, c in key]

    # ---- 个股封板质量
    zt["seal_ratio"] = zt.order_amount / zt.currency_value
    zt["log_cap"] = np.log10(zt.currency_value.clip(lower=1))
    zt["type_code"] = zt.limit_up_type.map({"换手板": 0, "T字板": 1, "一字板": 2})
    zt["is20"] = zt.market_type.isin(["GEM", "STAR"]).astype(int)

    # ---- 题材
    blk = pd.read_sql("SELECT s.date, s.code, b.limit_up_num, b.days FROM block_stock s "
                      "JOIN block b ON b.date=s.date AND b.block_code=s.block_code", con)
    g = blk.groupby(["date", "code"]).agg(blk_num=("limit_up_num", "max"), blk_days=("days", "max"))
    zt = zt.join(g, on=["date", "code"])
    tags = zt.reason_type.fillna("").str.split("+")
    zt["n_tags"] = tags.map(lambda t: len([x for x in t if x]))
    ex = zt[["date"]].assign(tag=tags).explode("tag")
    ex = ex[ex.tag.fillna("") != ""]
    cnt = ex.groupby(["date", "tag"]).size()
    ex["c"] = [cnt[(d, t)] for d, t in zip(ex.date, ex.tag)]
    zt["tag_cnt"] = ex.groupby(level=0)["c"].max().reindex(zt.index).fillna(1)

    # ---- 日K位置特征 + 前向收益
    kl = pd.read_sql("SELECT * FROM kline WHERE code!='IDX' ORDER BY code, date", con)
    parts = []
    for code, kk in kl.groupby("code", sort=False):
        f = _kline_features(kk)
        f["code"] = code
        parts.append(f)
    kf = pd.concat(parts, ignore_index=True)
    zt = zt.merge(kf, on=["date", "code"], how="left")
    zt["seal_vs_amt"] = zt.order_amount / zt.amount

    # ---- 市场情绪（按天）
    idx = pd.read_sql("SELECT date, close FROM kline WHERE code='IDX' ORDER BY date", con)
    idx["idx_pct"] = idx.close.pct_change()
    cntk = pool.groupby(["date", "kind"]).size().unstack(fill_value=0).reindex(columns=["zt", "zb", "dt"], fill_value=0)
    mk = pd.DataFrame({"n_zt": cntk.zt, "n_zb": cntk.zb, "n_dt": cntk.dt})
    mk["zb_rate"] = mk.n_zb / (mk.n_zt + mk.n_zb).clip(lower=1)
    mk["max_lb"] = zt.groupby("date").lb.max()
    mk["n_lb2"] = zt[zt.lb >= 2].groupby("date").size().reindex(mk.index).fillna(0)
    pct = kf.set_index(["date", "code"])["pct"]
    lbmap = zt.set_index(["date", "code"])["lb"]
    prem, promo, promo_lb = {}, {}, {}
    for d in cal:
        p = prv.get(d)
        if not p or not by_day.get(p):
            continue
        y = by_day[p]
        today = by_day.get(d, set())
        vals = [pct.get((d, c), np.nan) for c in y]
        prem[d] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        promo[d] = len(y & today) / len(y)
        ylb = {c for c in y if lbmap[(p, c)] >= 2}
        promo_lb[d] = len(ylb & today) / len(ylb) if ylb else np.nan
    mk["prem_prev"], mk["promo"], mk["promo_lb"] = pd.Series(prem), pd.Series(promo), pd.Series(promo_lb)
    mk = mk.join(idx.set_index("date")["idx_pct"])
    mk.reset_index().rename(columns={"index": "date"}).to_sql("market", con, if_exists="replace", index=False)
    zt = zt.merge(mk.drop(columns=["n_zb"]).reset_index(), on="date", how="left")

    zt = zt.sort_values(["date", "lb", "first_min"], ascending=[True, False, True])
    zt.to_sql("feat", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS feat_date ON feat(date)")
    con.commit()
    con.close()
    log(f"特征表 {len(zt)} 行，{zt.date.nunique()} 天；"
        f"次日连板率 {zt.y_lb.mean():.1%}，波段达成率 {zt.y_swing.mean():.1%}，"
        f"缺日K {zt.k_close.isna().sum()} 行")
    return zt
