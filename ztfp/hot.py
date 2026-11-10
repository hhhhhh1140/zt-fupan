"""情绪主线检测：最近几天的涨停股里，哪个"字"、哪个"词"或哪个题材突然扎堆，已经炒了多少天。

三类线索：
    字    股票名字里的字，按读音归并（羊/阳/洋 算同一个），也吸收同花顺标的"X字辈"
    词    股票名字里的两字词（东方、新华、传媒）
    题材  同花顺涨停原因里的标签（业绩、国资这类非情绪标签不算）

判定"热"：最近 W 个交易日里，带这个字/词/题材的涨停数远高于平时应有的数量
（名字类和全市场股票名字里的占比比，题材类和它自己前 120 天的占比比），并且里面已经走出 3 板以上的股票。
连续热的日子连成一轮"炒作"，起点取这一轮最早那只连板股的首板日。

    python -m ztfp hot        # 重新检测并生成 reports/hot_YYYYMMDD.html
"""
import re
from collections import Counter
from html import escape as e

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .db import ROOT, connect
from .report import CSS, JS, _col, _pct, _spark

try:
    from pypinyin import Style, lazy_pinyin

    def _py(ch):
        return lazy_pinyin(ch, style=Style.TONE3)[0]
except ImportError:      # 没装 pypinyin 时不做谐音归并，每个字单独算
    def _py(ch):
        return ch

W = 5              # 观察窗口（交易日）
BASE = 120         # 题材标签的对照期
GAP = 5            # 中间断这么多天以内算同一轮
MIN_LIFT = 3.0     # 至少是平时的几倍
MAX_P = 1e-4
LONG = 12          # 历史上只有约 3% 的炒作能超过这么多个交易日
LATE = 22          # 约一个月
STAGES = [(5, "起势"), (8, "发酵"), (LONG, "后段"), (LATE, "少见的长度"), (10 ** 6, "超过一个月")]

_STRIP = re.compile(r"ST|股份|集团|科技|控股|实业|发展")
_NOISE = re.compile(r"季报|年报|中报|业绩|预增|扭亏|减亏|增长|回购|增持|次新|国企|央企|国资|摘帽|复牌|小盘|分红|送转|字辈")
_ZI = re.compile(r"[“\"](.+?)[”\"]字辈")


def _clean(name):
    return re.sub(r"[^一-鿿]", "", _STRIP.sub("", name or ""))


def _name_tokens(name):
    s = _clean(name)
    return {("字", _py(c)) for c in s} | {("词", s[i:i + 2]) for i in range(len(s) - 1)}


def _tokens(name, reason):
    t = _name_tokens(name)
    for g in str(reason or "").split("+"):
        g = g.strip()
        m = _ZI.match(g)
        if m:      # 同花顺标了"羊字辈"的，即使名字里是别的写法也算
            t |= {("字", _py(c)) for c in m.group(1)} if len(m.group(1)) == 1 else {("词", m.group(1))}
        elif g and "已澄清" not in g and not _NOISE.search(g):
            t.add(("题材", g))
    return t


def _universe(names):
    """全市场股票名字，用来算一个字/词"平时"该出现多少次。没有趋势库时退回用涨停过的股票。"""
    try:
        from .trend_data import connect as tc
        con = tc()
        u = pd.read_sql("SELECT name FROM stock", con).name.tolist()
        con.close()
        if len(u) > 1000:
            return u
    except Exception:
        pass
    return list(names)


def _stage(age):
    return next(s for lim, s in STAGES if age <= lim)


def scan(log=print):
    con = connect()
    f = pd.read_sql("SELECT date, code, name, lb, reason_type, y_lb, n_close FROM feat", con)
    cal = sorted(f.date.unique())
    ci = {d: i for i, d in enumerate(cal)}
    f["di"] = f.date.map(ci)
    f["tk"] = [_tokens(a, b) for a, b in zip(f.name, f.reason_type)]
    uni = _universe(f.name.unique())
    ushare = Counter(t for nm in uni for t in _name_tokens(nm))
    ex = f[["date", "di", "code", "name", "lb", "reason_type", "tk"]].explode("tk").dropna(subset=["tk"])
    n_day = f.groupby("di").size().reindex(range(len(cal)), fill_value=0).to_numpy()
    cum = np.r_[0, n_day.cumsum()]

    rows = []
    for i in range(W - 1, len(cal)):
        w = ex[(ex.di > i - W) & (ex.di <= i)]
        N = cum[i + 1] - cum[i + 1 - W]
        lo = max(i + 1 - W - BASE, 0)
        NB = cum[i + 1 - W] - cum[lo]
        bc = ex[(ex.di <= i - W) & (ex.di >= lo)].groupby("tk").size()
        g = w.groupby("tk").agg(n=("code", "size"), stocks=("code", "nunique"), nlb=("lb", lambda s: int((s >= 2).sum())),
                                maxlb=("lb", "max"), codes=("code", lambda s: frozenset(s)))
        g = g[(g.stocks >= 3) & (g.n >= 5) & (g.maxlb >= 3) & (g.nlb >= 2)]
        day = []
        for tk, r in g.iterrows():
            if tk[0] == "题材":
                if NB < 300:
                    continue
                share = (bc.get(tk, 0) + 2) / (NB + 400)
            else:
                share = (ushare.get(tk, 0) + 1) / len(uni)
            exp = share * N
            p = poisson.sf(r.n - 1, exp)
            if r.n / exp >= MIN_LIFT and p < MAX_P:
                day.append(dict(date=cal[i], di=i, kind=tk[0], tok=tk[1], n=int(r.n), stocks=int(r.stocks), nlb=int(r.nlb),
                                maxlb=int(r.maxlb), lift=r.n / exp, heat=-np.log10(max(p, 1e-300)), codes=r.codes))
        # 去重："东方"和"方"、"传媒"和"媒"说的是同一批股票，只留热度最高的那个
        day.sort(key=lambda x: -x["heat"])
        keep = []
        for x in day:
            if not any(len(x["codes"] & k["codes"]) / len(x["codes"] | k["codes"]) >= 0.5 for k in keep):
                keep.append(x)
        rows += keep
    h = pd.DataFrame(rows)
    if h.empty:
        log("没有检测到情绪主线")
        return h

    # 连成一轮一轮的炒作
    h = h.sort_values(["kind", "tok", "di"]).reset_index(drop=True)
    h["ep"] = ((h.kind != h.kind.shift()) | (h.tok != h.tok.shift()) | (h.di - h.di.shift() > GAP)).cumsum()
    key = ex.set_index("tk")
    eps = []
    for ep, g in h.groupby("ep"):
        tk = (g.kind.iloc[0], g.tok.iloc[0])
        m = key.loc[[tk]]
        i0, i1 = g.di.iloc[0], g.di.iloc[-1]
        first = m[(m.di > i0 - W) & (m.di <= i0) & (m.lb >= 2)]
        start = int(max((first.di - first.lb + 1).min(), 0))            # 最早那只连板股的首板日
        mm = m[(m.di >= start) & (m.di <= i1)]
        top = mm.sort_values(["lb", "di"], ascending=[False, True]).iloc[0]
        if tk[0] == "字":      # 读音 -> 实际出现的字
            py = tk[1]
            cnt = Counter(c for nm in mm.drop_duplicates("code").name for c in _clean(nm) if _py(c) == py)
            zi = Counter(c for r in mm.reason_type.dropna() for g in r.split("+") for m in [_ZI.match(g.strip())] if m
                         for c in m.group(1) if _py(c) == py)      # 同花顺点名的那个字排最前（羊）
            chars = [c for c, _ in zi.most_common(1)] + [c for c, _ in cnt.most_common(3)]
            label = "/".join(list(dict.fromkeys(chars))[:3]) or py
        else:
            label = tk[1]
        pk = g.loc[g.heat.idxmax()]
        eps.append(dict(ep=ep, kind=tk[0], tok=tk[1], label=label, start=cal[start], start_i=start, first_hot=g.date.iloc[0],
                        last_hot=g.date.iloc[-1], last_i=int(i1), peak_date=pk.date, peak_heat=pk.heat, max_lb=int(top.lb),
                        leader=top["name"], n_stock=int(mm.code.nunique()), hot_days=len(g)))
    eps = pd.DataFrame(eps)
    h = h.merge(eps[["ep", "label", "start", "start_i"]], on="ep")
    h["age"] = h.di - h.start_i + 1
    h["codes"] = h.codes.map(lambda s: ",".join(sorted(s)))
    h.drop(columns=["start_i"]).to_sql("hot", con, if_exists="replace", index=False)
    eps.to_sql("hot_ep", con, if_exists="replace", index=False)
    con.close()
    log(f"情绪主线：{len(eps)} 轮，{h.date.nunique()} 天有检测结果")
    return h


def _load():
    con = connect()
    h = pd.read_sql("SELECT * FROM hot", con)
    eps = pd.read_sql("SELECT * FROM hot_ep", con)
    f = pd.read_sql("SELECT date, code, name, lb, reason_type, y_lb, n_close FROM feat", con)
    con.close()
    return h, eps, f


def current(date=None):
    """某天还在进行的主线：最近 GAP 个交易日内触发过。返回 (列表, 交易日历)。"""
    h, eps, f = _load()
    cal = sorted(f.date.unique())
    date = date or cal[-1]
    i = cal.index(date)
    eps = eps[(eps.first_hot <= date)].copy()
    hh = h[h.date <= date]
    last = hh.groupby("ep").agg(last_hot=("date", "max"), last_i=("di", "max"))
    eps = eps.drop(columns=["last_hot", "last_i"]).merge(last, on="ep")
    cur = eps[i - eps.last_i <= GAP].copy()
    cur["age"] = i - cur.start_i + 1
    cur["idle"] = i - cur.last_i
    today = f[f.date == date]
    out = []
    for _, r in cur.iterrows():
        g = hh[hh.ep == r.ep].sort_values("di")
        tk = (r.kind, r.tok)
        mem = today[[tk in _tokens(a, b) for a, b in zip(today.name, today.reason_type)]].sort_values("lb", ascending=False)
        pk = g.loc[g.heat.idxmax()]
        now = g.iloc[-1] if r.idle == 0 else None
        out.append(dict(r, hist=g.heat.tolist(), mem=mem, peak_date=pk.date, peak_heat=pk.heat,
                        heat=now.heat if now is not None else np.nan, lift=now.lift if now is not None else np.nan,
                        n=int(g.iloc[-1].n), nlb=int(g.iloc[-1].nlb), w_maxlb=int(g.iloc[-1].maxlb)))
    out.sort(key=lambda x: (x["idle"] > 0, -(x["heat"] if x["heat"] == x["heat"] else 0)))
    # 说的是同一批股票的（AI漫剧 / 文 / 影）并到最热的那条下面，只在它旁边提一句
    for c in out:
        c["set"] = set(hh[hh.ep == c["ep"]].sort_values("di").codes.iloc[-1].split(","))
        c["rel"] = []
    main = []
    for c in out:
        m = next((k for k in main if len(c["set"] & k["set"]) / min(len(c["set"]), len(k["set"])) >= 0.5), None)
        if m is not None:
            m["rel"].append(c["label"])
        else:
            main.append(c)
    return main, cal, date


def today_tags(date=None):
    """{股票代码: "羊/阳/洋 · 第 30 天"}，给明日关注名单打标记用。一只股票属于多条主线时取最热的。"""
    try:
        cur, _, _ = current(date)
    except Exception:
        return {}
    tags = {}
    for c in cur:
        for code in c["mem"].code:
            tags.setdefault(code, (c["label"], int(c["age"])))
    return tags


def _backtest(h, eps, f, cal):
    """历史上：一轮炒作持续多久；炒作到第几天时，里面的涨停股次日表现如何。"""
    ci = {d: i for i, d in enumerate(cal)}
    done = eps[eps.last_i < len(cal) - 1 - GAP].copy()
    done["dur"] = done.last_i - done.start_i + 1
    # 每个"股票-交易日"归到当天最热的那条主线
    x = h.assign(code=h.codes.str.split(",")).explode("code").sort_values("heat", ascending=False).drop_duplicates(["date", "code"])
    z = f.merge(x[["date", "code", "age", "kind"]], on=["date", "code"], how="left")
    z = z[z.y_lb.notna() & (z.date >= h.date.min())]
    z["b"] = pd.cut(z.age, [0] + [lim for lim, _ in STAGES], labels=[s for _, s in STAGES]).astype(object).fillna("不在任何主线里")
    return done, z


def _sector_html():
    """板块强度两张表：又涨又有资金流入的、涨得多但资金流出的。没有板块数据时返回空串。"""
    try:
        from . import sector
        top, outflow, sd = sector.top()
    except Exception:
        return ""
    fmt = lambda d: f"{d[4:6]}-{d[6:]}"
    yi = lambda v: f"{v / 1e8:+.1f} 亿"

    def lead(m):
        return "、".join(f'{e(n)}<span class="code">{"%d板" % lb if lb else "近5日%d次" % z}{"" if r != r else " %+.0f%%" % (r * 100)}</span>'
                        for n, z, lb, r in zip(m.name, m.zt, m.lb, m.r5)) or "近 5 日没有涨停股"

    r1 = "".join(
        f'<tr><td>{o["rk"]}</td><td class="l"><b>{e(o["name"])}</b><span class="code">{o["kind"]}</span></td><td class="{_col(o["r"])}">{_pct(o["r"], 1, True)}</td>'
        f'<td class="{_col(o["f"])}">{yi(o["f"])}</td><td>{_pct(o["fr"], 1)}</td><td><b>{o["days"]}</b> 天<br><small>{fmt(o["since"])} 起</small></td>'
        f'<td>{o["n_zt"]} / {o["n_stock"]}</td><td class="l">{lead(o["lead"])}</td></tr>' for o in top)
    r2 = "".join(
        f'<tr><td class="l"><b>{e(o["name"])}</b><span class="code">{o["kind"]}</span></td><td class="{_col(o["r"])}">{_pct(o["r"], 1, True)}</td>'
        f'<td class="{_col(o["f"])}">{yi(o["f"])}</td><td>{o["n_zt"]}</td><td class="l">{lead(o["lead"])}</td></tr>' for o in outflow)
    return f"""
<h2>资金和涨幅最强的板块（近 5 日）</h2>
<div class="note">东方财富的行业和概念板块，按"近 5 日涨幅、近 5 日主力净流入、净流入占成交额"三项的综合排名，必须既涨又有资金流入。
成分股高度重合的板块只留排名靠前的一个。"上榜"指连续排在前 30 名里的天数。龙头按近 5 日涨停次数、今天的板数、5 日涨幅排序。数据截至 {fmt(sd)}。</div>
<div class="card scroll"><table><thead><tr><th>排名</th><th class="l">板块</th><th>5日涨幅</th><th>5日主力净流入</th><th>占成交额</th><th>已上榜</th>
<th>近5日涨停股 / 成分股</th><th class="l">板块内龙头</th></tr></thead><tbody>{r1}</tbody></table></div>

<h2>涨得多但主力资金在流出的板块</h2>
<div class="note">情绪炒作的板块经常出现在这里：股价靠小单和游资推上去，大单统计口径下反而是净流出。影视、短剧这类最近就是这样。</div>
<div class="card scroll" style="max-width:980px"><table><thead><tr><th class="l">板块</th><th>5日涨幅</th><th>5日主力净流入</th><th>近5日涨停股</th>
<th class="l">板块内龙头</th></tr></thead><tbody>{r2}</tbody></table></div>
"""


def report(log=print):
    cur, cal, date = current()
    h, eps, f = _load()
    done, z = _backtest(h, eps, f, cal)
    ci = {d: i for i, d in enumerate(cal)}
    fmt = lambda d: f"{d[4:6]}-{d[6:]}"
    days_cal = lambda a, b: (pd.Timestamp(b) - pd.Timestamp(a)).days + 1

    def stage_tag(c):
        s = _stage(c["age"])
        warn = c["age"] > LONG
        cool = c["idle"] > 0 or (c["heat"] == c["heat"] and c["heat"] < 0.5 * c["peak_heat"] and ci[date] - ci[c["peak_date"]] >= 3)
        red = ' style="color:var(--up);border-color:var(--up)"' if warn else ""
        t = f'<span class="tag"{red}>{s}</span>'
        if cool:
            t += '<span class="tag">热度回落</span>'
        return t

    rows = ""
    for c in cur:
        mem = "、".join(f'{e(n)}<span class="code">{lb}板</span>' for n, lb in zip(c["mem"].name.head(8), c["mem"].lb.head(8)))
        more = f" 等 {len(c['mem'])} 只" if len(c["mem"]) > 8 else ""
        rel = f'<br><small>相关：{e("、".join(c["rel"][:5]))}</small>' if c["rel"] else ""
        state = "今天仍在触发" if c["idle"] == 0 else f"已 {c['idle']} 天未触发"
        rows += (
            f'<tr><td class="l"><b>{e(c["label"])}</b><span class="code">{c["kind"]}</span>{rel}</td>'
            f'<td><b>{c["age"]}</b> 个交易日<br><small>{days_cal(c["start"], date)} 天</small></td><td class="l">{stage_tag(c)}</td>'
            f'<td>{fmt(c["start"])}</td><td>{fmt(c["first_hot"])}</td>'
            f'<td>{c["n"]} / {c["nlb"]}</td><td>{c["w_maxlb"]} 板</td>'
            f'<td>{"%.1f 倍" % c["lift"] if c["lift"] == c["lift"] else "-"}</td>'
            f'<td class="s">{_spark(c["hist"])}</td><td>{fmt(c["peak_date"])}</td>'
            f'<td class="l">{e(c["leader"])} {c["max_lb"]} 板</td><td class="l">{mem or "今天没有涨停"}{more}<br><small>{state}</small></td></tr>')
    head = ('<th class="l">主线</th><th>已炒作</th><th class="l">阶段</th><th>起点</th><th>首次检出</th><th>近5日涨停/连板</th>'
            '<th>近5日最高</th><th>相对平时</th><th>热度走势</th><th>热度峰值日</th><th class="l">本轮最高</th><th class="l">今天涨停的成员</th>')
    t_cur = (f'<div class="card scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
             if rows else '<div class="note">今天没有检测到正在进行的情绪主线。</div>')

    # 历史：持续时间
    dur = done.dur
    name_ep = done[done.kind != "题材"]
    bins = [(1, 5), (6, 8), (9, LONG), (LONG + 1, LATE), (LATE + 1, 10 ** 6)]
    drow = "".join(
        f'<tr><td class="l">{s}（{"%d–%d" % (a, b) if b < 10 ** 5 else "%d 以上" % a} 个交易日）</td><td>{int(((dur >= a) & (dur <= b)).sum())}</td>'
        f'<td>{_pct(((dur >= a) & (dur <= b)).mean(), 0)}</td><td>{int(((name_ep.dur >= a) & (name_ep.dur <= b)).sum())}</td></tr>'
        for (a, b), (_, s) in zip(bins, STAGES))
    # 历史：炒作到第几天时，里面的涨停股次日怎么样
    order = [s for _, s in STAGES] + ["不在任何主线里"]
    g = z.groupby("b").agg(n=("code", "size"), y=("y_lb", "mean"), nc=("n_close", "mean"))
    g2 = z[z.lb >= 2].groupby("b").agg(n2=("code", "size"), y2=("y_lb", "mean"))
    g = g.join(g2).reindex(order).dropna(subset=["n"])
    arow = "".join(
        f'<tr><td class="l">{k}</td><td>{int(r.n)}</td><td><b>{_pct(r.y, 0)}</b></td><td class="{_col(r.nc)}">{_pct(r.nc, 1, True)}</td>'
        f'<td>{int(r.n2) if r.n2 == r.n2 else 0}</td><td>{_pct(r.y2, 0)}</td></tr>' for k, r in g.iterrows())
    # 过去的主线
    past = eps.sort_values("start", ascending=False)
    past = past[(past.last_i - past.start_i + 1 >= 6) | (past.max_lb >= 5)].head(40)
    prow = "".join(
        f'<tr><td class="l"><b>{e(r.label)}</b><span class="code">{r.kind}</span></td><td>{r.start[:4]}-{fmt(r.start)}</td><td>{fmt(r.first_hot)}</td>'
        f'<td>{fmt(r.last_hot)}{"（进行中）" if r.last_i >= len(cal) - 1 - GAP else ""}</td><td>{r.last_i - r.start_i + 1}</td>'
        f'<td>{ci[r.first_hot] - r.start_i}</td><td>{fmt(r.peak_date)}</td><td class="l">{e(r.leader)} {r.max_lb} 板</td><td>{r.n_stock}</td></tr>'
        for r in past.itertuples())

    def tile(label, val, sub):
        return f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small></div>'

    live = [c for c in cur if c["idle"] == 0]
    late = [c for c in cur if c["age"] > LONG]
    tiles = "".join([
        tile("正在进行的主线", f"{len(cur)} 条", f"其中今天仍在触发 {len(live)} 条"),
        tile(f"已超过 {LONG} 个交易日", f"{len(late)} 条", "、".join(f'{c["label"]}（{c["age"]} 天）' for c in late[:3]) or "无"),
        tile("历史上一轮炒作的中位长度", f"{dur.median():.0f} 个交易日", f"{len(dur)} 轮已结束的炒作"),
        tile(f"撑过 {LONG} 个交易日的比例", _pct((dur > LONG).mean(), 0), f"撑过 {LATE} 个交易日（约一个月）的 {int((dur > LATE).sum())} 轮"),
    ])
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>情绪主线 {date}</title><style>{CSS}</style></head><body><main>
<h1>情绪主线 · {date[:4]}-{date[4:6]}-{date[6:]}</h1>
<div class="sub">最近 {W} 个交易日的涨停股里，哪个字、哪个词或哪个题材的数量远多于平时，并且已经走出 3 板以上的股票。同音字算同一个（羊/阳/洋）。</div>
<div class="tiles">{tiles}</div>

<h2>正在进行的主线</h2>
<div class="note">"已炒作"从这一轮最早那只连板股的首板日算起。"相对平时"是近 {W} 日带这个字/词/题材的涨停数，除以按平时占比应有的数量。
超过 {LONG} 个交易日标红（历史上很少有炒作能撑这么久）；热度不到峰值一半、或已有几天没再触发的，标"热度回落"。</div>
{t_cur}

{_sector_html()}
<h2>炒作到第几天时，里面的涨停股次日表现</h2>
<div class="note">把历史上每只涨停股按"它所在主线当时已炒作多少天"分组。用来看主线越到后面是不是越难做。</div>
<div class="card scroll" style="max-width:820px"><table><thead><tr><th class="l">所处阶段</th><th>涨停样本</th><th>次日晋级率</th><th>次日收盘</th>
<th>其中连板股</th><th>连板股次日晋级率</th></tr></thead><tbody>{arow}</tbody></table></div>

<h2>一轮炒作能持续多久</h2>
<div class="note">只统计已经结束的炒作（最后一次触发之后又过了 {GAP} 个交易日以上）。长度从起点算到最后一次触发。</div>
<div class="card scroll" style="max-width:820px"><table><thead><tr><th class="l">长度</th><th>轮数</th><th>占比</th><th>其中名字类（字/词）</th></tr></thead><tbody>{drow}</tbody></table></div>

<h2>过去的主线</h2>
<div class="note">只列持续 6 个交易日以上或走出过 5 板的。"检出用时"是从起点到系统第一次检测到之间隔了几个交易日。</div>
<div class="card scroll"><table><thead><tr><th class="l">主线</th><th>起点</th><th>首次检出</th><th>最后触发</th><th>持续交易日</th><th>检出用时</th>
<th>热度峰值日</th><th class="l">最高</th><th>涉及股票</th></tr></thead><tbody>{prow}</tbody></table></div>
<div class="note" style="margin-top:24px">数据来源：同花顺涨停数据中心。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    path = ROOT / "reports" / f"hot_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path
