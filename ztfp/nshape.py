"""N 型回调：2 板以上的连板股断板后回调几天，再度启动。

每天列出两类股票：
    回调中      连板段在 2~12 个交易日前结束，之后还没有再涨停过
    今天再涨停  回调之后今天第一次重新涨停（N 型的右边那一笔开始了）
并统计历史上这类股票之后的走势，方便判断值不值得做。

需要全市场日K（趋势库 data/trend.db），没有的话这一页会被跳过。

    python -m ztfp nshape     # 生成 reports/nshape_YYYYMMDD.html
"""
from html import escape as e

import numpy as np
import pandas as pd

from . import hot
from .db import ROOT, connect
from .report import CSS, JS, _col, _pct

MIN_LB = 2         # 第一波至少几连板
TRACK = 15         # 断板后跟踪多少个交易日
WATCH = (2, 12)    # 断板后第几天到第几天算"回调中"


def panel():
    """每个连板段断板后 1..TRACK 天的状态和之后的结果。"""
    from .trend_data import connect as tc
    con = connect()
    f = pd.read_sql("SELECT date, code, name, lb, reason_type, y_lb, n_open, n_close, is20 FROM feat", con)
    con.close()
    f = f.sort_values(["code", "date"]).reset_index(drop=True)
    f["sid"] = ((f.code != f.code.shift()) | (f.lb != f.lb.shift() + 1)).cumsum()
    st = f.groupby("sid").agg(code=("code", "first"), name=("name", "last"), start=("date", "first"), end=("date", "last"),
                              lb0=("lb", "first"), final=("lb", "max"), reason=("reason_type", "last")).reset_index()
    st = st[(st.lb0 == 1) & (st.final >= MIN_LB)]
    c2 = tc()
    k = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM k WHERE date>=?", c2, params=(min(st.start.min(), "20250801"),))
    c2.close()
    k = k[k.code.isin(set(st.code))].sort_values(["code", "date"])
    zt = set(zip(f.date, f.code))
    K = {c: g.reset_index(drop=True) for c, g in k.groupby("code")}
    rows = []
    for s in st.itertuples():
        g = K.get(s.code)
        if g is None:
            continue
        dts = g.date.to_numpy()
        ix = {d: i for i, d in enumerate(dts)}
        if s.start not in ix or s.end not in ix or ix[s.start] == 0:
            continue
        a, en = ix[s.start], ix[s.end]
        c, o, h, l, v = (g[x].to_numpy() for x in ("close", "open", "high", "low", "volume"))
        p0, p1 = c[a - 1], c[en]                      # 起涨前收盘、最后一个板的收盘
        vmax = v[a:en + 2].max()
        first = None
        for d in range(1, TRACK + 1):
            i = en + d
            if i >= len(c):
                break
            is_zt = (dts[i], s.code) in zt
            if is_zt and first is None:
                first = d
            n = len(c)
            rows.append(dict(
                sid=s.sid, code=s.code, name=s.name, final=s.final, start=s.start, end=s.end, reason=s.reason, date=dts[i], d=d,
                zt=is_zt, seen_zt=first is not None and first < d, close=c[i], pct=c[i] / c[i - 1] - 1,
                retr=(p1 - c[i]) / (p1 - p0), dd=c[i] / p1 - 1, v3=v[max(en + 1, i - 2):i + 1].mean() / vmax,
                ma10=c[i] / c[max(0, i - 9):i + 1].mean() - 1,
                c5=c[i + 5] / c[i] - 1 if i + 5 < n else np.nan, c10=c[i + 10] / c[i] - 1 if i + 10 < n else np.nan,
                o5=c[i + 6] / o[i + 1] - 1 if i + 6 < n else np.nan,
                hi10=h[i + 1:i + 11].max() / c[i] - 1 if i + 10 < n else np.nan,
                zt5=float(any((dts[j], s.code) in zt for j in range(i + 1, i + 6))) if i + 5 < n else np.nan))
    p = pd.DataFrame(rows)
    p["fb"] = p.final.clip(upper=5).map(lambda x: f"{x} 板" if x < 5 else "5 板以上")
    p["rb"] = pd.cut(p.retr, [-99, 0.2, 0.4, 0.6, 0.8, 1.0, 99], labels=["不到 20%", "20–40%", "40–60%", "60–80%", "80–100%", "跌破起涨点"])
    p["vb"] = pd.cut(p.v3, [0, .25, .4, .6, 99], labels=["25% 以下", "25–40%", "40–60%", "60% 以上"])
    return p, f


def _stat(x, by):
    return x.groupby(by, observed=True).agg(n=("code", "size"), zt5=("zt5", "mean"), c5=("c5", "mean"), c10=("c10", "mean"),
                                            win=("c10", lambda s: (s > 0).mean()), hi=("hi10", "mean"), o5=("o5", "mean"))


def report(log=print):
    p, f = panel()
    date = f.date.max()
    cur_tok = {}
    try:
        cur, _, _ = hot.current(date)
        cur_tok = {(c["kind"], c["tok"]): (c["label"], int(c["age"])) for c in cur}
    except Exception:
        pass

    def theme(name, reason):
        t = hot._tokens(name, reason)
        return next((f"{lab} · 第 {age} 天" for tk, (lab, age) in cur_tok.items() if tk in t), "")

    fmt = lambda d: f"{d[4:6]}-{d[6:]}"
    hist = p[(p.d >= 2) & (p.d <= 10) & ~p.zt & ~p.seen_zt & p.c10.notna()]      # 历史上"回调中"的每一天
    by_r, by_v = _stat(hist, "rb"), _stat(hist, "vb")

    def like(r):      # 同类历史：同样回吐幅度的股票，之后 5 日内再涨停的比例和 5 日平均涨幅
        s = by_r.loc[r.rb] if r.rb in by_r.index else None
        return (f'<td>{_pct(s.zt5, 0)}</td><td class="{_col(s.c5)}">{_pct(s.c5, 1, True)}</td>' if s is not None else "<td>-</td><td>-</td>")

    def rows(df):
        out = ""
        for r in df.itertuples():
            th = theme(r.name, r.reason)
            tag = f'<span class="tag">{e(th)}</span>' if th else ""
            out += (f'<tr><td class="l">{e(r.name)}<span class="code">{r.code}</span>{tag}</td><td>{r.final} 板<br><small>{fmt(r.start)} ~ {fmt(r.end)}</small></td>'
                    f'<td>第 {r.d} 天</td><td class="{_col(r.pct)}">{_pct(r.pct, 1, True)}</td><td>{r.close:.2f}</td>'
                    f'<td class="{_col(r.dd)}">{_pct(r.dd, 0, True)}</td><td>{_pct(r.retr, 0)}</td><td>{_pct(r.v3, 0)}</td>'
                    f'<td>{"线上" if r.ma10 > 0 else "线下"}</td>{like(r)}<td class="r">{e(r.reason or "")}</td></tr>')
        return out

    t = p[p.date == date]
    again = t[t.zt & ~t.seen_zt & (t.d >= 3)].sort_values(["final", "d"], ascending=[False, True])
    pull = t[~t.zt & ~t.seen_zt & t.d.between(*WATCH)].sort_values(["final", "v3"], ascending=[False, True])
    head = ('<th class="l">股票</th><th>第一波</th><th>断板后</th><th>今日涨跌</th><th>今收</th><th>距第一波高点</th><th>已回吐</th>'
            '<th>近3日量 / 第一波最大量</th><th>10日线</th><th>同类5日内再涨停</th><th>同类后5日</th><th class="l">第一波的涨停原因</th>')
    tb = lambda r: f'<div class="card scroll"><table><thead><tr>{head}</tr></thead><tbody>{r}</tbody></table></div>'

    # 历史统计
    full = p[p.groupby("sid").d.transform("max") >= TRACK]
    s = full.groupby("sid").apply(lambda x: pd.Series(dict(
        fb=x.fb.iloc[0], rezt=float(x.zt[(x.d >= 3) & (x.d <= 12)].any()), newhigh=float((x.dd[x.d >= 3] > 0).any()), mindd=x.dd.min())))
    g1 = s.groupby("fb").agg(n=("rezt", "size"), rezt=("rezt", "mean"), nh=("newhigh", "mean"), dd=("mindd", "mean"))
    r1 = "".join(f'<tr><td class="l">{k}</td><td>{int(r.n)}</td><td><b>{_pct(r.rezt, 0)}</b></td><td>{_pct(r.nh, 0)}</td><td class="dn">{_pct(r.dd, 0, True)}</td></tr>'
                 for k, r in g1.iterrows())

    def stat_rows(g):
        return "".join(
            f'<tr><td class="l">{k}</td><td>{int(r.n)}</td><td>{_pct(r.zt5, 0)}</td><td class="{_col(r.c5)}">{_pct(r.c5, 1, True)}</td>'
            f'<td class="{_col(r.c10)}">{_pct(r.c10, 1, True)}</td><td>{_pct(r.win, 0)}</td><td class="up">{_pct(r.hi, 0, True)}</td>'
            f'<td class="{_col(r.o5)}">{_pct(r.o5, 1, True)}</td></tr>' for k, r in g.iterrows())

    sh = ('<th>样本</th><th>5日内再涨停</th><th>后5日</th><th>后10日</th><th>10日胜率</th><th>10日内最高</th><th>次日开盘买持有5日</th>')
    allh = _stat(hist.assign(a="全部回调中的股票"), "a")
    by_f = _stat(hist, "fb")
    # N 型确认日 vs 普通首板
    cf = p[(p.d >= 3) & (p.d <= 12) & p.zt & ~p.seen_zt].drop_duplicates(["date", "code"])[["date", "code", "o5"]]
    f1 = f[(f.lb == 1) & f.y_lb.notna()].merge(cf, on=["date", "code"], how="left", indicator=True)
    f1["g"] = np.where(f1._merge == "both", "回调后再涨停", "普通首板")
    g3 = f1.groupby("g").agg(n=("code", "size"), y=("y_lb", "mean"), no=("n_open", "mean"), nc=("n_close", "mean"))
    r3 = "".join(f'<tr><td class="l">{k}</td><td>{int(r.n)}</td><td><b>{_pct(r.y, 0)}</b></td><td class="{_col(r.no)}">{_pct(r.no, 1, True)}</td>'
                 f'<td class="{_col(r.nc)}">{_pct(r.nc, 1, True)}</td></tr>' for k, r in g3.iterrows())
    o5 = cf.o5.mean()

    def tile(label, val, sub):
        return f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small></div>'

    tiles = "".join([
        tile("回调中", f"{len(pull)} 只", f"断板后第 {WATCH[0]}–{WATCH[1]} 天，还没再涨停"),
        tile("今天回调后再涨停", f"{len(again)} 只", "N 型右边那一笔的第一个板"),
        tile("历史上断板后 3–12 天内再涨停", _pct(s.rezt.mean(), 0), f"{len(s)} 个 2 板以上的连板段"),
        tile("回调中买入、持有 5 日", _pct(allh.c5.iloc[0], 1, True), f"10 日 {_pct(allh.c10.iloc[0], 1, True)}，胜率 {_pct(allh.win.iloc[0], 0)}"),
    ])
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>N 型回调 {date}</title><style>{CSS}</style></head><body><main>
<h1>N 型回调 · {date[:4]}-{date[4:6]}-{date[6:]}</h1>
<div class="sub">连板 {MIN_LB} 板以上的股票断板后回调几天、再度启动的形态。这里列出正在回调的和今天重新涨停的，并给出历史上同类股票之后的走势。这是形态筛选结果，不是买卖建议。</div>
<div class="tiles">{tiles}</div>

<h2>今天回调后再涨停（{len(again)} 只）</h2>
<div class="note">断板至少 2 天之后今天第一次重新涨停。"已回吐"是今天收盘相对第一波涨幅吐回了多少，负数表示已经超过第一波的高点。</div>
{tb(rows(again)) if len(again) else '<div class="note">今天没有。</div>'}

<h2>回调中（{len(pull)} 只）</h2>
<div class="note">按第一波板数从高到低、量能从小到大排序。"同类"两列是历史上回吐幅度相近的股票：之后 5 日内再涨停的比例，和之后 5 日的平均涨幅（收盘买入口径）。
名字后面的标记表示它属于正在炒作的情绪主线。</div>
{tb(rows(pull)) if len(pull) else '<div class="note">今天没有。</div>'}

<h2>断板之后会发生什么</h2>
<div class="note">2 板以上的连板段断板后 {TRACK} 个交易日内的情况。再涨停的比例很高，N 型是常见走势；但期间的平均最大回撤也有 15–19%。</div>
<div class="card scroll" style="max-width:760px"><table><thead><tr><th class="l">第一波</th><th>段数</th><th>第 3–12 天再涨停</th><th>第 3 天后收盘创新高</th>
<th>期间最大回撤（相对高点）</th></tr></thead><tbody>{r1}</tbody></table></div>

<h2>回调中买入的历史结果</h2>
<div class="note">样本是历史上每只"回调中"的股票的每一天（断板后第 2–10 天、还没再涨停）。除了最后一列，都是当天收盘买入的口径。
平均收益在各种分法下都是负的或接近零：再涨停的那部分涨得多，但没再涨停的更多、跌得也不少。</div>
<div class="grid2">
<div class="card scroll"><table><thead><tr><th class="l">已回吐第一波涨幅</th>{sh}</tr></thead><tbody>{stat_rows(allh)}{stat_rows(by_r)}</tbody></table></div>
<div class="card scroll"><table><thead><tr><th class="l">近3日量 / 第一波最大量</th>{sh}</tr></thead><tbody>{stat_rows(by_v)}</tbody></table></div>
</div>
<div class="card scroll" style="max-width:900px;margin-top:14px"><table><thead><tr><th class="l">第一波板数</th>{sh}</tr></thead><tbody>{stat_rows(by_f)}</tbody></table></div>

<h2>回调后再涨停的那个板，次日表现</h2>
<div class="note">和普通首板比，晋级率差不多，次日收盘反而更低。次日开盘买入并持有 5 日，历史平均 {_pct(o5, 1, True)}。</div>
<div class="card scroll" style="max-width:640px"><table><thead><tr><th class="l">类型</th><th>样本</th><th>次日晋级率</th><th>次日开盘</th><th>次日收盘</th></tr></thead><tbody>{r3}</tbody></table></div>
<div class="note" style="margin-top:24px">数据来源：同花顺涨停数据中心、东方财富日K（前复权）。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    path = ROOT / "reports" / f"nshape_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path
