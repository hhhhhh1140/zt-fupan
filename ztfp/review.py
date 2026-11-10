"""涨停预测复盘页：模型过去给的连板概率，事后看准不准。reports/zt_review_YYYYMMDD.html

只统计样本外预测（给出概率时模型没见过答案）且次日结果已出的记录。
"""
from html import escape as e

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .db import ROOT, connect
from .report import CSS, JS, _col, _pct, _spark

TOP = 5


def _auc(y, p):
    return roc_auc_score(y, p) if y.nunique() == 2 else np.nan


def make():
    con = connect()
    f = pd.read_sql("SELECT f.*, p.p_lb, p.oos FROM feat f JOIN pred p USING(date, code)", con)
    con.close()
    f = f[(f.oos == 1) & f.p_lb.notna() & f.y_lb.notna()].copy()
    last = f.date.max()
    lim = np.where(f.is20 == 1, 0.2, 0.1)
    f["buyable"] = f.limit_up_type != "一字板"                 # 当天能买进的
    f["open_lim"] = f.n_open >= lim - 0.003                   # 次日开盘就涨停，开盘也买不到
    f["oc"] = (1 + f.n_close) / (1 + f.n_open) - 1            # 次日开盘买、收盘卖
    f["rank"] = f.groupby("date").p_lb.rank(ascending=False, method="first")
    f["rank_b"] = f[f.buyable].groupby("date").p_lb.rank(ascending=False, method="first")

    # ---------- 每天
    def day(g):
        t, tb = g[g["rank"] <= TOP], g[g.rank_b <= TOP]
        best = g.loc[g.p_lb.idxmax()]
        return pd.Series({
            "n": len(g), "base": g.y_lb.mean(), "hit": t.y_lb.sum(), "hit_b": tb.y_lb.sum(), "k_b": len(tb),
            "ret_b": tb.n_close.mean(), "ret_all": g.n_close.mean(), "auc": _auc(g.y_lb, g.p_lb),
            "best": best["name"], "best_p": best.p_lb, "best_y": best.y_lb, "best_ret": best.n_close})

    d = f.groupby("date").apply(day)
    d["rate_b"] = d.hit_b / d.k_b
    d["month"] = d.index.str[:4] + "-" + d.index.str[4:6]

    def tile(label, val, sub, series=None):
        return (f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small>'
                f'{_spark(series) if series is not None else ""}</div>')

    r20 = d.tail(20)
    roll = (d.hit_b.rolling(20).sum() / d.k_b.rolling(20).sum()).dropna()
    tiles = "".join([
        tile("样本外预测", f"{len(f)} 条", f"{len(d)} 个交易日，{d.index[0]} ~ {last}"),
        tile("整体区分力 AUC", f"{_auc(f.y_lb, f.p_lb):.3f}", f"近 20 日 {_auc(f[f.date >= r20.index[0]].y_lb, f[f.date >= r20.index[0]].p_lb):.3f}；0.5 等于瞎猜"),
        tile(f"每天概率前 {TOP} 的晋级率", _pct(d.hit.sum() / (TOP * len(d)), 0), f"全部涨停股的晋级率 {_pct(f.y_lb.mean(), 0)}"),
        tile(f"剔除一字板后的前 {TOP}", _pct(d.hit_b.sum() / d.k_b.sum(), 0), f"近 20 日 {_pct(r20.hit_b.sum() / r20.k_b.sum(), 0)}（走势为滚动 20 日）", roll),
        tile(f"前 {TOP}（非一字）次日平均涨幅", _pct(d.ret_b.mean(), 1, True), f"全部涨停股 {_pct(d.ret_all.mean(), 1, True)}，按涨停价到次日收盘"),
    ])

    # ---------- 最近 30 天逐日
    rows = ""
    for dt, r in d.tail(30).iloc[::-1].iterrows():
        rows += (f'<tr><td class="l">{dt[4:6]}-{dt[6:]}</td><td>{int(r.n)}</td><td>{_pct(r.base, 0)}</td>'
                 f'<td>{int(r.hit)} / {TOP}</td><td><b>{int(r.hit_b)} / {int(r.k_b)}</b></td>'
                 f'<td class="{_col(r.ret_b)}">{_pct(r.ret_b, 1, True)}</td><td class="{_col(r.ret_all)}">{_pct(r.ret_all, 1, True)}</td>'
                 f'<td class="l">{e(r.best)}<span class="code">{_pct(r.best_p, 0)}</span></td>'
                 f'<td class="{"up" if r.best_y == 1 else _col(r.best_ret)}">{"晋级" if r.best_y == 1 else _pct(r.best_ret, 1, True)}</td></tr>')
    daily = ('<div class="card scroll"><table><thead><tr><th class="l">预测日</th><th>涨停数</th><th>整体晋级率</th>'
             f'<th>概率前 {TOP} 晋级</th><th>非一字前 {TOP} 晋级</th><th>非一字前 {TOP} 次日涨幅</th><th>全部涨停股次日涨幅</th>'
             f'<th class="l">概率最高的一只</th><th>它的次日结果</th></tr></thead><tbody>{rows}</tbody></table></div>')

    # ---------- 按月
    g = d.groupby("month")
    mm = pd.DataFrame({"days": g.size(), "base": f.groupby(f.date.str[:4] + "-" + f.date.str[4:6]).y_lb.mean(),
                       "top": g.hit.sum() / (TOP * g.size()), "top_b": g.hit_b.sum() / g.k_b.sum(),
                       "ret_b": g.ret_b.mean(), "ret_all": g.ret_all.mean(),
                       "auc": f.groupby(f.date.str[:4] + "-" + f.date.str[4:6]).apply(lambda x: _auc(x.y_lb, x.p_lb))})
    rows = "".join(
        f'<tr><td class="l">{k}</td><td>{int(r.days)}</td><td>{_pct(r.base, 0)}</td><td>{_pct(r.top, 0)}</td><td><b>{_pct(r.top_b, 0)}</b></td>'
        f'<td class="{_col(r.top_b - r.base)}">{(r.top_b - r.base) * 100:+.0f} 点</td>'
        f'<td class="{_col(r.ret_b)}">{_pct(r.ret_b, 1, True)}</td><td class="{_col(r.ret_all)}">{_pct(r.ret_all, 1, True)}</td><td>{r.auc:.3f}</td></tr>'
        for k, r in mm.iloc[::-1].iterrows())
    monthly = ('<div class="card scroll"><table><thead><tr><th class="l">月份</th><th>交易日</th><th>整体晋级率</th>'
               f'<th>概率前 {TOP}</th><th>非一字前 {TOP}</th><th>比整体高</th><th>非一字前 {TOP} 次日涨幅</th><th>全部涨停股次日涨幅</th><th>AUC</th>'
               f'</tr></thead><tbody>{rows}</tbody></table></div>')

    # ---------- 按概率分档（固定阈值）
    def bucket(df, title):
        b = pd.cut(df.p_lb, [0, .10, .15, .25, .40, .60, 1.01], right=False,
                   labels=["10% 以下", "10–15%", "15–25%", "25–40%", "40–60%", "60% 以上"])
        rows = ""
        for k, x in df.groupby(b, observed=True):
            can = x[~x.open_lim]
            rows += (f'<tr><td class="l">{k}</td><td>{len(x)}</td><td>{_pct(x.p_lb.mean(), 0)}</td><td><b>{_pct(x.y_lb.mean(), 0)}</b></td>'
                     f'<td class="{_col(x.n_open.mean())}">{_pct(x.n_open.mean(), 1, True)}</td>'
                     f'<td class="{_col(x.n_close.mean())}">{_pct(x.n_close.mean(), 1, True)}</td>'
                     f'<td>{_pct(x.open_lim.mean(), 0)}</td><td class="{_col(can.oc.mean())}">{_pct(can.oc.mean(), 1, True)}</td></tr>')
        return (f'<div class="card scroll"><h3>{title}</h3><table><thead><tr><th class="l">模型给的概率</th><th>样本</th><th>平均预测</th>'
                '<th>实际晋级率</th><th>次日开盘</th><th>次日收盘</th><th>次日开盘即涨停</th><th>开盘买收盘卖</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    buckets = bucket(f, "全部涨停股") + bucket(f[f.buyable], "剔除一字板（当天能买进的）")

    # ---------- 按类别看偏差
    def bias(key, title, order=None):
        gg = f.groupby(key, observed=True).agg(n=("code", "size"), p=("p_lb", "mean"), y=("y_lb", "mean"))
        if order:
            gg = gg.reindex([o for o in order if o in gg.index])
        rows = "".join(f'<tr><td class="l">{e(str(k))}</td><td>{int(r.n)}</td><td>{_pct(r.p, 0)}</td><td><b>{_pct(r.y, 0)}</b></td>'
                       f'<td class="{_col(r.y - r.p)}">{(r.y - r.p) * 100:+.1f} 点</td></tr>' for k, r in gg.iterrows())
        return (f'<div class="card scroll"><h3>{title}</h3><table><thead><tr><th class="l"></th><th>样本</th><th>平均预测</th>'
                f'<th>实际晋级率</th><th>实际 - 预测</th></tr></thead><tbody>{rows}</tbody></table></div>')

    f["板数"] = f.lb.clip(upper=5).map({1: "首板", 2: "2 板", 3: "3 板", 4: "4 板", 5: "5 板及以上"})
    f["板块"] = f.is20.map({0: "主板 10cm", 1: "创业板/科创板 20cm"})
    f["情绪"] = pd.qcut(f.zb_rate, 3, labels=["炸板率低", "炸板率中", "炸板率高"])
    biases = "".join([bias("板数", "按连板数", ["首板", "2 板", "3 板", "4 板", "5 板及以上"]),
                      bias("limit_up_type", "按涨停类型", ["换手板", "T字板", "一字板"]),
                      bias("板块", "按涨跌幅限制"), bias("情绪", "按当日炸板率")])

    # ---------- 近 10 天看走眼的
    rec = f[f.date >= d.index[-10]]

    def miss(df, title):
        rows = "".join(
            f'<tr><td class="l">{r.date[4:6]}-{r.date[6:]}</td><td class="l">{e(r["name"])}<span class="code">{r.code}</span></td>'
            f'<td>{r.lb}</td><td class="l">{e(r.limit_up_type or "")}</td><td><b>{_pct(r.p_lb, 0)}</b></td>'
            f'<td class="{"up" if r.y_lb == 1 else _col(r.n_close)}">{"晋级" if r.y_lb == 1 else _pct(r.n_close, 1, True)}</td>'
            f'<td class="r">{e(r.reason_type or "")}</td></tr>' for _, r in df.iterrows())
        return (f'<div class="card scroll"><h3>{title}</h3><table><thead><tr><th class="l">预测日</th><th class="l">股票</th><th>连板</th>'
                f'<th class="l">类型</th><th>当时概率</th><th>次日结果</th><th class="l">涨停原因</th></tr></thead><tbody>{rows}</tbody></table></div>')

    misses = (miss(rec[rec.y_lb == 0].nlargest(8, "p_lb"), "概率高却没晋级")
              + miss(rec[rec.y_lb == 1].nsmallest(8, "p_lb"), "概率低却晋级了"))

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>涨停预测复盘 {last}</title><style>{CSS}</style></head><body><main>
<h1>涨停预测复盘 · 截至 {last[:4]}-{last[4:6]}-{last[6:]}</h1>
<div class="sub">模型过去给出的次日连板概率，事后对不对。只统计给出概率时模型没见过答案、且次日结果已出的预测。</div>
<div class="tiles">{tiles}</div>

<h2>最近 30 个交易日</h2>
<div class="note">"非一字前 {TOP}"是剔除当天一字板（买不进）之后概率最高的 {TOP} 只。次日涨幅按涨停价到次日收盘计算，晋级的记为涨停幅度。</div>
{daily}

<h2>按月</h2>
<div class="note">"比整体高"是非一字前 {TOP} 的晋级率减去当月全部涨停股的晋级率，持续为正才说明排序有用。</div>{monthly}

<h2>概率说多少，实际是多少</h2>
<div class="note">"开盘买收盘卖"只统计次日开盘没有直接涨停的股票，是次日开盘价买入、当天收盘价卖出的平均收益，未扣手续费和滑点。</div>
<div class="grid2">{buckets}</div>

<h2>模型在哪类股票上偏高或偏低</h2>
<div class="note">"实际 - 预测"为正表示模型低估了这类股票，为负表示高估。</div><div class="grid2">{biases}</div>

<h2>近 10 个交易日看走眼的</h2><div class="grid2">{misses}</div>
<div class="note" style="margin-top:24px">数据来源：同花顺涨停数据中心、东方财富日K（前复权）。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"zt_review_{last}.html"
    path.write_text(html, encoding="utf-8")
    return path
