"""集合竞价数据：9:15~9:25 每分钟的虚拟撮合价、9:25 的竞价成交量、9:30 开盘后第一分钟的成交。

数据来自通达信行情服务器的历史逐笔（pytdx），一年前的也能取到。
只抓两类"股票-交易日"：前一天涨停的，和前一天没涨停但当天高开 MIN_GAP 以上的。

    python -m ztfp auction            # 9:25 之后、开盘前跑：抓今天的竞价，生成竞价复盘页
    python -m ztfp auction 20261110   # 重看某一天的竞价复盘（带当天的实际结果）
    python -m ztfp auction fetch      # 补抓历史竞价数据（可中断，下次接着抓）
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .db import connect

MIN_GAP = 0.02
HOSTS = ["115.238.56.198", "115.238.90.165", "119.97.185.59", "124.70.199.56", "124.71.187.72", "60.191.117.167",
         "124.71.187.122", "218.75.126.9", "60.12.136.250", "180.153.18.170"]
LIVE = ["sztdx.gtjas.com", "shtdx.gtjas.com"]      # 只有这几台给当天的实时数据，其余的只有历史
MINS = [f"09:{m}" for m in range(15, 25)]
COLS = ["date", "code", "pc", "open", "vol"] + [f"m{m[3:]}" for m in MINS] + ["pmax", "pmin", "n_snap", "p930", "v1", "a1", "h1", "l1", "c1"]
SCHEMA = f"CREATE TABLE IF NOT EXISTS auc ({', '.join(c + (' TEXT' if c in ('date', 'code') else ' REAL') for c in COLS)}, PRIMARY KEY (date, code))"

_local = threading.local()
_seq = iter(range(10 ** 9))


def _api(live=False):
    from pytdx.hq import TdxHq_API
    key = "live" if live else "api"
    hosts = LIVE if live else HOSTS
    api = getattr(_local, key, None)
    if api is None:
        for _ in range(len(hosts) * 2):
            api = TdxHq_API(raise_exception=False)
            try:
                if api.connect(hosts[next(_seq) % len(hosts)], 7709, time_out=5):
                    setattr(_local, key, api)
                    return api
            except Exception:
                pass
        raise RuntimeError("通达信服务器都连不上")
    return api


def _page(mk, code, start, date):
    for _ in range(3):
        try:
            if date is None:      # 当天的逐笔
                d = _api(True).get_transaction_data(mk, code, start, 2000)
            else:
                d = _api().get_history_transaction_data(mk, code, start, 2000, int(date))
            return d or []
        except Exception:
            pass
        _local.api = _local.live = None      # 连接坏了，换一台重连
        time.sleep(0.3)
    return []


def _early(mk, code, date):
    """当天最早的那部分逐笔（竞价段 + 开盘第一分钟）。逐笔是从收盘往前分页的，所以要找到最后一页。"""
    pages, start = [], 2000
    d = _page(mk, code, start, date)
    if not d:
        return _page(mk, code, 0, date)
    while len(d) == 2000:
        pages.append(d)
        start += 2000
        d = _page(mk, code, start, date)
    rows = list(d)
    if not rows or rows[-1]["time"] <= "09:31":      # 最早一页太短，第一分钟跨到了后一页
        rows += list(pages[-1]) if pages else list(_page(mk, code, start - 2000, date))
    return rows


def one(date, code, pc, today=False):
    """一只股票一天的竞价摘要。today=True 时取当天的实时逐笔。"""
    rows = _early(1 if code[0] == "6" else 0, code, None if today else date)
    pre = [r for r in rows if r["time"] < "09:25" or (r["time"] == "09:25" and r["vol"] == 0)]
    match = [r for r in rows if r["time"] == "09:25" and r["vol"] > 0]
    m1 = [r for r in rows if r["time"] == "09:30"]
    if not match:
        return None
    op, vol = match[0]["price"], sum(r["vol"] for r in match)
    path, last = [], np.nan
    for m in MINS:      # 每分钟最后一次虚拟撮合价，没有就沿用上一分钟的
        ps = [r["price"] for r in pre if r["time"] == m and r["price"] > 0]
        last = ps[-1] if ps else last
        path.append(last)
    pp = [r["price"] for r in pre if r["price"] > 0]
    v1 = sum(r["vol"] for r in m1)
    return (date, code, pc, op, vol, *path, max(pp) if pp else np.nan, min(pp) if pp else np.nan, len(pp),
            m1[0]["price"] if m1 else np.nan, v1, sum(r["vol"] * r["price"] for r in m1) * 100,
            max(r["price"] for r in m1) if m1 else np.nan, min(r["price"] for r in m1) if m1 else np.nan, m1[-1]["price"] if m1 else np.nan)


def targets():
    """要抓的"股票-交易日"：前一天涨停的 + 前一天没涨停但高开的。需要趋势库里的全市场日K。"""
    from .trend_data import connect as tc
    con = connect()
    f = pd.read_sql("SELECT date, code FROM feat", con)
    con.close()
    c2 = tc()
    k = pd.read_sql("SELECT code, date, open, close FROM k WHERE code!='IDX' AND date>=?", c2, params=(f.date.min(),))
    c2.close()
    k = k.sort_values(["code", "date"])
    k["pc"] = k.groupby("code").close.shift(1)
    cal = sorted(f.date.unique())
    nxt = dict(zip(cal[:-1], cal[1:]))
    yz = set(zip(f.date.map(nxt), f.code))
    k = k.dropna(subset=["pc"])
    k = k[k.date > cal[0]]
    k["yzt"] = [(d, c) in yz for d, c in zip(k.date, k.code)]
    return k[k.yzt | (k.open / k.pc - 1 >= MIN_GAP)][["date", "code", "pc", "yzt"]]


def fetch(log=print, workers=10):
    con = connect()
    con.execute(SCHEMA)
    done = set(con.execute("SELECT date, code FROM auc").fetchall())
    t = targets()
    todo = [(d, c, p) for d, c, p in zip(t.date, t.code, t.pc) if (d, c) not in done]
    log(f"竞价数据：共 {len(t)} 条，已有 {len(done)}，待抓 {len(todo)}")
    t0, n, bad = time.time(), 0, 0
    sql = f"INSERT OR REPLACE INTO auc VALUES ({','.join('?' * len(COLS))})"

    def safe(a):
        try:
            return one(*a) or (a[0], a[1], a[2]) + (None,) * (len(COLS) - 3)      # 没取到也记一行，免得反复重试
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(safe, todo):
            n += 1
            if r is None:
                bad += 1
            else:
                con.execute(sql, tuple(None if isinstance(x, float) and x != x else x for x in r))
            if n % 2000 == 0:
                con.commit()
                log(f"  {n}/{len(todo)}  失败 {bad}  {n / (time.time() - t0):.0f} 条/秒")
    con.commit()
    con.close()
    log(f"竞价数据抓取完成：{n} 条，失败 {bad}，用时 {(time.time() - t0) / 60:.1f} 分钟")




# ---------------------------------------------------------------- 特征、模型、页面
FEATS = ["gap", "gapn", "late", "e19", "drop20", "fake", "vr", "turn", "lim", "n_snap", "pret5", "ylb", "yz", "p_lb", "sealr", "yzt"]
STEP, MIN_DAYS, TOPN = 10, 80, 20


def load():
    """竞价数据 + 9:25 时能知道的背景（昨天是否涨停、昨日量、流通市值）+ 事后的结果。"""
    from .trend_data import connect as tc
    con = connect()
    a = pd.read_sql("SELECT * FROM auc WHERE open IS NOT NULL", con)
    f = pd.read_sql("SELECT f.date, f.code, f.name, f.lb, f.limit_up_type, f.order_amount, f.first_min, f.currency_value, f.reason_type, p.p_lb "
                    "FROM feat f LEFT JOIN pred p USING(date, code)", con)
    con.close()
    c2 = tc()
    k = pd.read_sql("SELECT code, date, open, high, close, volume FROM k WHERE code!='IDX' AND date>=?", c2, params=(str(int(a.date.min()) - 100),))
    stock = pd.read_sql("SELECT code, name, float_cap FROM stock", c2).set_index("code")
    cal = [r[0] for r in c2.execute("SELECT date FROM k WHERE code='IDX' ORDER BY date")]
    c2.close()
    k = k.sort_values(["code", "date"])
    g = k.groupby("code")
    k["n_open"], k["n_close"] = g.open.shift(-1), g.close.shift(-1)
    # 昨日量、近 5 日涨幅按"上一根K线"取，这样今天早上（今天的日K还没有）也能算
    last = k.groupby("code").tail(6)
    prevk = pd.DataFrame({"pvol_l": last.groupby("code").volume.last(), "c_l": last.groupby("code").close.last(),
                          "c6": last.groupby("code").close.first(), "d_l": last.groupby("code").date.last()})
    k["pvol"] = g.volume.shift(1)
    k["pret5"] = g.close.shift(1) / g.close.shift(6) - 1
    a = a.merge(k[["code", "date", "high", "close", "pvol", "pret5", "n_open", "n_close"]], on=["code", "date"], how="left")
    live = a.close.isna()                       # 今天的日K还没入库：用最近一根K线当"昨天"
    a.loc[live, "pvol"] = a.loc[live, "code"].map(prevk.pvol_l)
    a.loc[live, "pret5"] = a.loc[live, "code"].map(prevk.c_l / prevk.c6 - 1)
    fd = sorted(f.date.unique())
    allcal = sorted(set(cal) | set(a.date))
    prev = dict(zip(allcal[1:], allcal[:-1]))
    a["pd"] = a.date.map(prev)
    y = f.rename(columns={"date": "pd", "lb": "ylb", "limit_up_type": "ytype", "name": "yname", "reason_type": "yreason"})
    a = a.merge(y[["pd", "code", "ylb", "ytype", "order_amount", "currency_value", "p_lb", "yname", "yreason"]], on=["pd", "code"], how="left")
    a = a.merge(f[["date", "code", "lb", "first_min"]], on=["date", "code"], how="left")
    a["name"] = a.yname.fillna(a.code.map(stock.name)).fillna(a.code)
    a["known"] = a.date <= fd[-1]               # 这一天的涨停结果是否已经有了
    a["yzt"] = a.ylb.notna().astype(float)
    a["zt"] = a.lb.notna()
    a["lim"] = np.where(a.code.str[:2].isin(["30", "68"]), 0.2, 0.1)
    a["up"] = (a.pc * (1 + a.lim)).round(2)
    a["gap"] = a.open / a.pc - 1
    a["gapn"] = a.gap / a.lim
    a["ref"] = a.m20.fillna(a.m21).fillna(a.m22)            # 9:20 撤单截止后的第一个虚拟价
    a["late"] = a.open / a.ref - 1                          # 不可撤单阶段价格的变化
    a["e19"] = a.m19 / a.pc - 1                             # 可撤单阶段最后的虚拟涨幅
    a["drop20"] = a.ref / a.m19 - 1                         # 9:20 那一下撤单带来的变化
    a["fake"] = ((a.pmax >= a.up - 0.011) & (a.open < a.up - 0.011)).astype(float)   # 竞价中顶过涨停，最后没开在涨停
    a["vr"] = a.vol / a.pvol
    a["amt"] = a.vol * 100 * a.open
    a["cap"] = a.currency_value.fillna(a.code.map(stock.float_cap))
    a["turn"] = a.amt / a.cap
    a["sealr"] = a.amt / a.order_amount
    a["ylb"] = a.ylb.fillna(0)
    a["yz"] = (a.ytype == "一字板").astype(float)
    a["buy"] = a.open < a.up - 0.011
    a["px"] = np.where(a.p930 > 0, a.p930, a.open)          # 9:30 第一笔成交价
    a["r_c"], a["r_no"], a["r_nc"] = a.close / a.px - 1, a.n_open / a.px - 1, a.n_close / a.px - 1
    a["fast5"] = a.zt & (a.first_min <= 5)
    a[FEATS] = a[FEATS].astype(float)
    return a


def train(a, log=print):
    """滚动训练"当天是否涨停"，给每一行一个样本外概率 p_zt；最新的日子用之前全部数据训练。"""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    a["p_zt"] = np.nan
    days = sorted(a.date.unique())
    lab = a.buy & a.known
    for i in range(MIN_DAYS, len(days), STEP):
        tr = a[lab & (a.date < days[i])]
        te = a[a.date.isin(days[i:i + STEP])]
        m = HGB(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(tr[FEATS], tr.zt)
        a.loc[te.index, "p_zt"] = m.predict_proba(te[FEATS])[:, 1]
    return a


def _flag(r):
    if not r.buy:
        return "一字开盘"
    if r.p_zt >= 0.4:
        return "强抢筹"
    if r.p_zt >= 0.2:
        return "抢筹"
    if r.yzt and (r.gap <= -0.02 or (r.fake and r.open / r.pmax - 1 <= -0.05)):
        return "流出"
    return ""


def report(date=None, log=print):
    from html import escape as e
    from . import hot
    from .db import ROOT
    from .report import CSS, JS, _col, _hm, _pct, _prob
    a = train(load(), log)
    date = date or a.date.max()
    d = a[a.date == date].copy()
    if d.empty:
        raise RuntimeError(f"{date} 没有竞价数据")
    known = bool(d.known.iloc[0])
    try:
        cur, _, _ = hot.current(min(date, sorted(a[a.known].date.unique())[-1]))
        toks = {(c["kind"], c["tok"]): c["label"] for c in cur}
    except Exception:
        toks = {}
    d["flag"] = [_flag(r) for r in d.itertuples()]
    fmt = lambda x: f"{x[4:6]}-{x[6:]}"
    yi = lambda v: "-" if v != v else (f"{v / 1e8:.2f} 亿" if v >= 1e8 else f"{v / 1e4:.0f} 万")
    FL = {"强抢筹": "color:var(--up);border-color:var(--up)", "抢筹": "color:var(--up)", "流出": "color:var(--dn);border-color:var(--dn)", "一字开盘": ""}

    def rows(df, ranked=False):
        out = ""
        for i, r in enumerate(df.itertuples(), 1):
            th = next((lab for tk, lab in toks.items() if tk in hot._tokens(r.name, r.yreason if isinstance(r.yreason, str) else "")), "")
            tags = (f'<span class="tag">昨 {int(r.ylb)} 板</span>' if r.ylb else "") + (f'<span class="tag">{e(th)}</span>' if th else "") + (
                f'<span class="tag" style="{FL[r.flag]}">{r.flag}</span>' if r.flag else "")
            res = ""
            if known:
                res = (f'<td>{"<b class=up>涨停</b> " + _hm(r.first_min) if r.zt else "未涨停"}</td>'
                       f'<td class="{_col(r.r_c)}">{_pct(r.r_c, 1, True)}</td><td class="{_col(r.r_no)}">{_pct(r.r_no, 1, True)}</td>')
            out += ((f'<tr><td>{i}</td>' if ranked else "<tr>")
                    + f'<td class="l">{e(str(r.name))}<span class="code">{r.code}</span>{tags}</td>'
                    f'<td class="{_col(r.e19)}">{_pct(r.e19, 1, True)}</td><td class="{_col(r.gap)}"><b>{_pct(r.gap, 1, True)}</b></td>'
                    f'<td class="{_col(r.late)}">{_pct(r.late, 1, True)}</td><td>{yi(r.amt)}</td><td>{_pct(r.turn, 2)}</td><td>{_pct(r.vr, 1)}</td>'
                    f'<td>{_pct(r.p_lb, 0) if r.p_lb == r.p_lb else "-"}</td>{_prob(r.p_zt) if r.buy else "<td>买不进</td>"}{res}</tr>')
        return out

    head = ('<th class="l">股票</th><th>9:20 前虚拟涨幅</th><th>开盘涨幅</th><th>9:20 后变化</th><th>竞价金额</th><th>竞价换手</th>'
            '<th>竞价量 / 昨日量</th><th>昨日连板概率</th><th>当日涨停概率</th>' + ('<th>结果</th><th>买入到收盘</th><th>买入到次日开盘</th>' if known else ""))
    tb = lambda r, rk=False: f'<div class="card scroll"><table><thead><tr>{"<th></th>" if rk else ""}{head}</tr></thead><tbody>{r}</tbody></table></div>'
    top = d[d.buy].sort_values("p_zt", ascending=False).head(TOPN)
    yz = d[d.yzt == 1].sort_values("gap", ascending=False)
    out = yz[yz.flag == "流出"]

    # 历史统计（全部样本外）
    h = a[a.buy & a.known & a.p_zt.notna() & a.r_nc.notna()]
    h = h.assign(rk=h.groupby("date").p_zt.rank(ascending=False, method="first"))

    def line(name, x):
        return (f'<tr><td class="l">{name}</td><td>{len(x)}</td><td><b>{_pct(x.zt.mean(), 0)}</b></td><td>{_pct(x.fast5.mean(), 0)}</td>'
                f'<td>{_pct(x.gap.mean(), 1, True)}</td><td class="{_col(x.r_c.mean())}">{_pct(x.r_c.mean(), 1, True)}</td>'
                f'<td class="{_col(x.r_no.mean())}">{_pct(x.r_no.mean(), 1, True)}</td><td>{_pct((x.r_no > 0).mean(), 0)}</td></tr>')

    hrow = "".join([line("当日涨停概率前 3", h[h.rk <= 3]), line("当日涨停概率前 5", h[h.rk <= 5]), line("当日涨停概率前 10", h[h.rk <= 10]),
                    line("昨日涨停股里，昨日连板概率前 5", h[h.yzt == 1].assign(r2=lambda x: x.groupby("date").p_lb.rank(ascending=False, method="first")).query("r2 <= 5")),
                    line("全部买得进的（基准）", h)])
    pb = pd.cut(h.p_zt, [0, .05, .1, .2, .3, .5, 1], labels=["5% 以下", "5–10%", "10–20%", "20–30%", "30–50%", "50% 以上"])
    crow = "".join(line(k, x) for k, x in h.groupby(pb, observed=True))
    gb = pd.cut(h.gap, [-1, -.03, 0, .02, .04, .06, .08, 1], labels=["低开 3% 以上", "低开 0–3%", "高开 0–2%", "高开 2–4%", "高开 4–6%", "高开 6–8%", "高开 8% 以上"])
    g1 = "".join(line(k, x) for k, x in h[h.yzt == 1].groupby(gb, observed=True))
    g0 = "".join(line(k, x) for k, x in h[h.yzt == 0].groupby(gb, observed=True))
    sh = '<th>样本</th><th>当天涨停</th><th>5 分钟内封板</th><th>平均开盘涨幅</th><th>买入到收盘</th><th>买入到次日开盘</th><th>次日开盘胜率</th>'
    # 最近几天的复盘
    recent = ""
    for dt in sorted(h.date.unique())[-5:][::-1]:
        x = h[(h.date == dt) & (h.rk <= 5)].sort_values("rk")
        cells = "、".join(f'{e(str(r.name))}<span class="code">{_pct(r.gap, 0, True)} → {"涨停 " + _hm(r.first_min) if r.zt else _pct(r.close / r.pc - 1, 0, True)}</span>' for r in x.itertuples())
        recent += f'<tr><td class="l">{fmt(dt)}</td><td>{int(x.zt.sum())} / {len(x)}</td><td>{int(x.fast5.sum())}</td><td class="l" style="white-space:normal">{cells}</td></tr>'

    def tile(label, val, sub):
        return f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small></div>'

    t10 = top.head(10)
    tiles = "".join([
        tile("昨日涨停股", f"{len(yz)} 只", f"一字开盘 {int((~yz.buy).sum())} 只，高开 {int((yz.gap > 0).sum())} 只，低开 {int((yz.gap < 0).sum())} 只"),
        tile("昨天没涨停、今天高开 2% 以上", f"{int((d.yzt == 0).sum())} 只", f"其中判为抢筹 {int(((d.yzt == 0) & d.flag.isin(['抢筹', '强抢筹'])).sum())} 只"),
        tile("竞价最强的 10 只" + ("，当天涨停" if known else ""), f"{int(t10.zt.sum())} 只" if known else f"{_pct(t10.p_zt.mean(), 0)}",
             f"其中 5 分钟内封板 {int(t10.fast5.sum())} 只" if known else "平均当日涨停概率"),
        tile("历史上前 3 的当天涨停率", _pct(h[h.rk <= 3].zt.mean(), 0), f"5 分钟内封板 {_pct(h[h.rk <= 3].fast5.mean(), 0)}，{h.date.nunique()} 个交易日"),
        tile("历史上前 3，9:30 买到次日开盘", _pct(h[h.rk <= 3].r_no.mean(), 1, True), "按 9:30 第一笔成交价买入，未扣手续费"),
    ])
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>竞价复盘 {date}</title><style>{CSS}</style></head><body><main>
<h1>竞价复盘 · {date[:4]}-{date[4:6]}-{date[6:]} 9:25</h1>
<div class="sub">集合竞价结束时的情况：昨日涨停的股票，加上昨天没涨停但今天高开 {MIN_GAP:.0%} 以上的股票。{"这一天已经收盘，表里附了实际结果。" if known else "盘前生成，结果列要等收盘后才有。"}这是模型的排序结果，不是买卖建议。</div>
<div class="tiles">{tiles}</div>

<h2>竞价最强的 {TOPN} 只（开盘买得进的）</h2>
<div class="note">按"当日涨停概率"排序，这个概率只用 9:25 能看到的信息算：开盘涨幅、竞价金额占流通市值的比例、竞价量和昨日全天量的比、9:20 前后的虚拟价格等。
"9:20 前虚拟涨幅"是可撤单阶段最后的价格，可能是虚的；"9:20 后变化"是不可撤单的五分钟里价格的变化。概率 40% 以上标"强抢筹"，20% 以上标"抢筹"。</div>
{tb(rows(top, True), True)}

<h2>昨日涨停股里竞价走弱的（{len(out)} 只）</h2>
<div class="note">低开 2% 以上，或者竞价中一度顶到涨停、最后比最高虚拟价低 5% 以上开出来。</div>
{tb(rows(out)) if len(out) else '<div class="note">没有。</div>'}

<h2>全部昨日涨停股的竞价（{len(yz)} 只）</h2>
{tb(rows(yz))}

<h2>最近 5 个交易日：竞价最强的 5 只后来怎么样</h2>
<div class="note">每只后面是"开盘涨幅 → 结果"，涨停的写首封时间，没涨停的写当天收盘涨幅。</div>
<div class="card scroll"><table><thead><tr><th class="l">日期</th><th>涨停 / 只数</th><th>5 分钟内封板</th><th class="l">明细</th></tr></thead><tbody>{recent}</tbody></table></div>

<h2>竞价强弱和当天结果（历史，样本外）</h2>
<div class="note">竞价强的股票当天涨停的比例很高，而且多数在开盘 5 分钟内封板。但 9:30 买入的平均收益是负的：竞价越强开得越高，没封住的那一半回落也越多。</div>
<div class="card scroll" style="max-width:980px"><table><thead><tr><th class="l">每天选</th>{sh}</tr></thead><tbody>{hrow}</tbody></table></div>
<div class="grid2" style="margin-top:14px">
<div class="card scroll"><table><thead><tr><th class="l">模型给的当日涨停概率</th>{sh}</tr></thead><tbody>{crow}</tbody></table></div>
<div class="card scroll"><table><thead><tr><th class="l">昨日涨停股，按开盘涨幅</th>{sh}</tr></thead><tbody>{g1}</tbody></table></div>
</div>
<div class="card scroll" style="max-width:980px;margin-top:14px"><table><thead><tr><th class="l">昨天没涨停的，按开盘涨幅</th>{sh}</tr></thead><tbody>{g0}</tbody></table></div>
<div class="note" style="margin-top:24px">数据来源：通达信行情（逐笔）、同花顺涨停数据中心、东方财富日K。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    path = ROOT / "reports" / f"auction_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path


def live(log=print):
    """9:25 之后跑：全市场取开盘价 -> 挑出昨日涨停的和高开的 -> 抓它们今天的竞价逐笔 -> 生成页面。"""
    from .fetch import now_cn
    from .trend_data import connect as tc
    now = now_cn()
    today = now.strftime("%Y%m%d")
    if now.strftime("%H%M") < "0925":
        log("还没到 9:25，竞价没结束，现在看到的价格可能是虚的。")
    con = connect()
    con.execute(SCHEMA)
    last = con.execute("SELECT MAX(date) FROM feat").fetchone()[0]
    yz = {r[0] for r in con.execute("SELECT code FROM feat WHERE date=?", (last,))}
    if last >= today:
        raise RuntimeError(f"{today} 已经收盘入库了，用 python -m ztfp auction {today} 看这一天的竞价复盘")
    c2 = tc()
    codes = [r[0] for r in c2.execute("SELECT code FROM stock")]
    c2.close()
    quotes = []
    for i in range(0, len(codes), 80):      # 一次最多 80 只
        quotes += _quotes(codes[i:i + 80])
    if not quotes:
        raise RuntimeError("取不到实时行情（非交易时段，或行情服务器连不上）")
    cand = [(today, q["code"], q["last_close"]) for q in quotes
            if q["open"] > 0 and q["last_close"] > 0 and (q["code"] in yz or q["open"] / q["last_close"] - 1 >= MIN_GAP)]
    log(f"{today} 行情 {len(quotes)} 只，昨日（{last}）涨停 {len(yz)} 只，候选 {len(cand)} 只，开始抓竞价逐笔")
    sql = f"INSERT OR REPLACE INTO auc VALUES ({','.join('?' * len(COLS))})"
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(lambda a: _try_one(*a), cand):
            if r:
                con.execute(sql, tuple(None if isinstance(x, float) and x != x else x for x in r))
    con.commit()
    con.close()
    return report(today, log)


def _quotes(batch):
    """一批里只要有一只服务器不认的代码，整批都返回空，所以取不到就对半拆开再取。"""
    q = None
    for _ in range(2):
        try:
            q = _api(True).get_security_quotes([(1 if c[0] == "6" else 0, c) for c in batch])
        except Exception:
            q = None
            _local.live = None
        if q:
            return q
    if len(batch) == 1:
        return []
    h = len(batch) // 2
    return _quotes(batch[:h]) + _quotes(batch[h:])


def _try_one(date, code, pc):
    try:
        return one(date, code, pc, today=True)
    except Exception:
        return None
