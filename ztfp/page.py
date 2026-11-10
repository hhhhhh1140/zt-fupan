"""合并页：把"明日关注 / 涨停复盘 / 预测复盘 / 趋势突破"做成一个带标签页的 HTML。

    reports/fupan_YYYYMMDD.html   当天的合并页
    reports/index.html            永远是最新一天

"明日关注"是按固定规则从今天的涨停池里筛出来的名单：非一字板、按模型的次日连板概率从高到低取前 N 只，
并附上历史上同一概率档的实际结果。它是模型的排序结果，不是买卖建议。
"""
import re
import shutil
from html import escape as e

import numpy as np
import pandas as pd

from . import hot, nshape, report, review, sector, trend
from .db import ROOT, connect
from .report import CSS, JS, _col, _hm, _pct, _prob

TOP_N = 10
WRAP = ' style="white-space:normal;min-width:300px"'     # 成员列允许换行，不然右边的列会被挤出屏幕
BINS = [0, .10, .15, .25, .40, .60, 1.01]
BIN_NAMES = ["10% 以下", "10–15%", "15–25%", "25–40%", "40–60%", "60% 以上"]

TAB_CSS = """
.tabs{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.tabs div{max-width:1240px;margin:0 auto;padding:0 16px;display:flex;gap:4px;overflow-x:auto}
.tabs button{font:inherit;font-size:14.5px;color:var(--mut);background:none;border:0;border-bottom:2px solid transparent;padding:12px 14px;cursor:pointer;white-space:nowrap}
.tabs button:hover{color:var(--ink)}.tabs button.on{color:var(--ink);font-weight:600;border-bottom-color:var(--up)}
section[data-p]{display:none}section[data-p].on{display:block}
.rank{display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:50%;background:var(--bar);color:var(--up);font-weight:600;font-size:12.5px}
"""
TAB_JS = """
const show=k=>{document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===k));
document.querySelectorAll('section[data-p]').forEach(s=>s.classList.toggle('on',s.dataset.p===k));};
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{show(b.dataset.t);history.replaceState(null,'','#'+b.dataset.t);window.scrollTo(0,0)});
const h=location.hash.slice(1);show(document.querySelector('section[data-p="'+h+'"]')?h:'watch');
"""


def _main(path):
    """取出一个单页的 <main> 内容。"""
    return re.search(r"<main>(.*)</main>", path.read_text(encoding="utf-8"), re.S).group(1)


def _r5(date):
    """全市场股票近 5 日涨幅 {代码: 涨幅}，来自趋势库；没有就返回空。"""
    try:
        from .trend_data import connect as tc
        con = tc()
        ds = [r[0] for r in con.execute("SELECT DISTINCT date FROM k WHERE code='IDX' AND date<=? ORDER BY date DESC LIMIT 6", (date,))]
        k = pd.read_sql("SELECT code, date, close FROM k WHERE date IN (?, ?)", con, params=(ds[-1], ds[0]))
        con.close()
        p = k.pivot(index="code", columns="date", values="close")
        return (p[ds[0]] / p[ds[-1]] - 1).dropna().to_dict()
    except Exception:
        return {}


def _themes(d, date):
    """明日关注的板块和主线：每条带上今天涨停的成员（含连板概率）和没涨停但近 5 日最强的成员。
    返回 (表格 html, {代码: [所属板块/主线名]})。"""
    prob = d.set_index("code").p_lb.to_dict()
    lbs = d.set_index("code").lb.to_dict()
    names = d.set_index("code")["name"].to_dict()
    r5 = _r5(date)
    belong = {}
    yi = lambda v: f"{v / 1e8:+.1f} 亿"

    def cells(codes, allnames):
        """codes: 这个板块/主线的全部成员代码 -> (今天涨停的, 没涨停但近 5 日最强的)"""
        zt = sorted([c for c in codes if c in prob], key=lambda c: -(prob[c] if prob[c] == prob[c] else 0))
        a = "、".join(f'{e(names[c])}<span class="code">{lbs[c]}板 {_pct(prob[c], 0)}</span>' for c in zt[:8]) or "今天没有涨停"
        if len(zt) > 8:
            a += f" 等 {len(zt)} 只"
        rest = sorted([c for c in codes if c not in prob and c in r5], key=lambda c: -r5[c])[:5]
        b = "、".join(f'{e(allnames.get(c, c))}<span class="code">{r5[c] * 100:+.0f}%</span>' for c in rest) or "-"
        return a, b

    rows = ""
    try:
        cur, _, _ = hot.current(date)
    except Exception:
        cur = []
    for c in cur:
        if c["idle"] > 0:
            continue
        allm = hot.name_members(c["kind"], c["tok"])
        codes = set(allm) | set(c["mem"].code)
        for code in c["mem"].code:
            belong.setdefault(code, []).append(c["label"])
        a, b = cells(codes, allm)
        late = c["age"] > hot.LONG
        rows += (f'<tr><td class="l"><b>{e(c["label"])}</b><span class="code">主线·{c["kind"]}</span></td>'
                 f'<td>第 <b>{c["age"]}</b> 天{"<br><small style=color:var(--up)>少见的长度</small>" if late else ""}</td>'
                 f'<td>-</td><td>-</td><td class="l"{WRAP}>{a}</td><td class="l"{WRAP}>{b if c["kind"] != "题材" else "-"}</td></tr>')
    try:
        top, outflow, _ = sector.top(date)
    except Exception:
        top, outflow = [], []
    for o, flag in [(o, "") for o in top[:10]] + [(o, "资金流出") for o in outflow if o["n_zt"] > 0][:4]:
        m = o["mem"]
        allm = dict(zip(m.stock, m.name))
        for code in m.stock:
            if code in prob:
                belong.setdefault(code, []).append(o["name"])
        a, b = cells(set(m.stock), allm)
        days = f'上榜 <b>{o["days"]}</b> 天' if not flag else f'<small>{flag}</small>'
        rows += (f'<tr><td class="l"><b>{e(o["name"])}</b><span class="code">{o["kind"]}板块</span></td><td>{days}</td>'
                 f'<td class="{_col(o["r"])}">{_pct(o["r"], 1, True)}</td><td class="{_col(o["f"])}">{yi(o["f"])}</td>'
                 f'<td class="l"{WRAP}>{a}</td><td class="l"{WRAP}>{b}</td></tr>')
    if not rows:
        return "", belong
    html = f"""<h2>明日关注的主线和板块</h2>
<div class="note">前面几行是正在炒作的情绪主线，后面是近 5 日涨幅和主力资金流入综合最强的板块，最后几行是涨得多但主力资金流出的板块。
"今天涨停的成员"后面是板数和模型给的明日连板概率；"没涨停但近 5 日最强"后面是近 5 日涨幅。</div>
<div class="card scroll"><table><thead><tr><th class="l">主线 / 板块</th><th>持续</th><th>5日涨幅</th><th>5日主力净流入</th>
<th class="l">今天涨停的成员</th><th class="l">没涨停但近 5 日最强的成员</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="note">历史上（最近 123 个交易日）：同时在主线和强势板块里的涨停股，次日晋级率 28%，都不在的是 16%。
但没涨停的成分股并没有稳定的后续涨幅：强势板块成分股之后 5 日比全市场多 0.3%，10 日反而少 0.3%，板块内涨幅最靠前的几只更差。</div>
"""
    return html, belong


def _watch():
    con = connect()
    f = pd.read_sql("SELECT f.*, p.p_lb, p.p_swing, p.oos FROM feat f JOIN pred p USING(date, code)", con)
    con.close()
    date = f.date.max()
    d = f[f.date == date].copy()
    d["open_num"] = d.open_num.fillna(0)

    # 历史上同一概率档（非一字板）的实际结果
    ha = f[(f.oos == 1) & f.p_lb.notna() & f.y_lb.notna()].copy()
    ha["open_lim"] = ha.n_open >= np.where(ha.is20 == 1, 0.2, 0.1) - 0.003
    ha["oc"] = (1 + ha.n_close) / (1 + ha.n_open) - 1
    ha["b"] = pd.cut(ha.p_lb, BINS, right=False, labels=BIN_NAMES)

    def by_bucket(x):
        g = x.groupby("b", observed=True).agg(n=("code", "size"), y=("y_lb", "mean"), no=("n_open", "mean"),
                                              nc=("n_close", "mean"), ol=("open_lim", "mean"))
        g["oc"] = x[~x.open_lim].groupby("b", observed=True).oc.mean()
        return g

    h = ha[ha.limit_up_type != "一字板"].copy()
    hb, hb_yz = by_bucket(h), by_bucket(ha[ha.limit_up_type == "一字板"])   # 一字板和自己的同类比
    d["b"] = pd.cut(d.p_lb, BINS, right=False, labels=BIN_NAMES)

    # 规则本身的历史成绩：每天非一字板里概率最高的 N 只
    h["rk"] = h.groupby("date").p_lb.rank(ascending=False, method="first")
    top = h[h.rk <= TOP_N]
    allz = f[(f.oos == 1) & f.y_lb.notna()]

    theme = hot.today_tags(date)       # {代码: (主线, 已炒作天数)}
    d["b"] = pd.cut(d.p_lb, BINS, right=False, labels=BIN_NAMES)
    themes_html, belong = _themes(d, date)

    def rows(df, ranked, hb):
        out = ""
        for i, (_, r) in enumerate(df.iterrows(), 1):
            tags = ('<span class="tag">20cm</span>' if r.is20 else "") + (
                '<span class="tag" style="color:var(--up);border-color:var(--up)">近异动线</span>' if r.dev10_next >= 0.9 else "")
            if r.code in theme:
                tags += f'<span class="tag">{e(theme[r.code][0])} · 第 {theme[r.code][1]} 天</span>'
            s = hb.loc[r.b] if r.b in hb.index else None
            hist = (f'<td>{_pct(s.y, 0)}</td><td class="{_col(s.no)}">{_pct(s.no, 1, True)}</td><td>{_pct(s.ol, 0)}</td>'
                    if s is not None else "<td>-</td><td>-</td><td>-</td>")
            out += (
                (f'<tr><td><span class="rank">{i}</span></td>' if ranked else "<tr>")
                + f'<td class="l">{e(r["name"])}<span class="code">{r.code}</span>{tags}</td>'
                f'<td>{r.lb} 板</td><td class="l">{e(r.limit_up_type or "")}</td><td>{_hm(r.first_min)}</td><td>{int(r.open_num)}</td>'
                f'<td>{_pct(r.seal_ratio, 2)}</td><td>{r.turnover_rate:.1f}%</td><td>{r.latest:.2f}</td>'
                f'{_prob(r.p_lb)}{hist}<td class="l">{e("、".join(belong.get(r.code, [])[:3])) or "-"}</td>'
                f'<td class="r">{e(r.reason_type or "")}</td></tr>')
        return out

    head = ('<th class="l">股票</th><th>连板</th><th class="l">类型</th><th>首封</th><th>开板</th><th>封单/流通</th><th>换手</th>'
            '<th>今收</th><th>明日连板概率</th><th>同档历史晋级率</th><th>同档次日平均高开</th><th>同档开盘即涨停</th><th class="l">所属主线 / 强势板块</th><th class="l">涨停原因</th>')
    buy = d[d.limit_up_type != "一字板"].sort_values("p_lb", ascending=False)
    yz = d[d.limit_up_type == "一字板"].sort_values("p_lb", ascending=False)
    t1 = f'<div class="card scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows(buy.head(TOP_N), True, hb)}</tbody></table></div>'
    t2 = (f'<div class="card scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows(yz, False, hb_yz)}</tbody></table></div>'
          if len(yz) else '<div class="note">今天没有一字板。</div>')
    brow = "".join(
        f'<tr><td class="l">{k}</td><td>{int(r.n)}</td><td><b>{_pct(r.y, 0)}</b></td><td class="{_col(r.no)}">{_pct(r.no, 1, True)}</td>'
        f'<td class="{_col(r.nc)}">{_pct(r.nc, 1, True)}</td><td>{_pct(r.ol, 0)}</td><td class="{_col(r.oc)}">{_pct(r.oc, 1, True)}</td></tr>'
        for k, r in hb.iterrows())

    def tile(label, val, sub):
        return f'<div class="tile"><small>{label}</small><b>{val}</b><small>{sub}</small></div>'

    tiles = "".join([
        tile("今日涨停", f"{len(d)} 只", f"其中一字板 {len(yz)} 只，连板 {(d.lb >= 2).sum()} 只"),
        tile(f"历史上前 {TOP_N} 的晋级率", _pct(top.y_lb.mean(), 0), f"全部涨停股 {_pct(allz.y_lb.mean(), 0)}，{top.date.nunique()} 个交易日"),
        tile(f"历史上前 {TOP_N} 的次日涨幅", _pct(top.n_close.mean(), 1, True), f"全部涨停股 {_pct(allz.n_close.mean(), 1, True)}，涨停价到次日收盘"),
        tile(f"历史上前 {TOP_N} 的次日高开", _pct(top.n_open.mean(), 1, True), f"其中 {_pct(top.open_lim.mean(), 0)} 开盘直接涨停"),
        tile("次日开盘买、收盘卖", _pct(top[~top.open_lim].oc.mean(), 1, True), "只算开盘没涨停的；未扣手续费"),
    ])
    return date, f"""<main>
<h1>明日关注 · 基于 {date[:4]}-{date[4:6]}-{date[6:]} 收盘</h1>
<div class="sub">从今天的涨停池里按固定规则筛出：非一字板，按模型的次日连板概率从高到低取前 {TOP_N} 只。这是模型的排序结果，不是买卖建议。</div>
<div class="tiles">{tiles}</div>

{themes_html}
<h2>明日连板概率最高的 {TOP_N} 只（非一字板）</h2>
<div class="note">"同档"指历史上模型给出相近概率的非一字板股票，后三列是它们次日的实际情况。名字后面的"羊 · 第 N 天"表示它属于正在炒作的情绪主线，详见"情绪主线"页。全部涨停股的概率见"涨停复盘"页。</div>
{t1}

<h2>一字板（{len(yz)} 只）</h2>
<div class="note">概率通常更高，但今天买不进，明天也多半高开或继续一字，单列出来仅供观察。这里的"同档"是和历史上的一字板比。</div>{t2}

<h2>各概率档的历史结果（非一字板）</h2>
<div class="note">概率越高，次日高开越多、开盘直接涨停的比例越大。"开盘买收盘卖"是次日开盘价买入、当天收盘卖出的平均收益，
只统计开盘没涨停的；它在各档都接近零，说明次日开盘再追基本没有额外收益。</div>
<div class="card scroll" style="max-width:860px"><table><thead><tr><th class="l">模型给的概率</th><th>样本</th><th>实际晋级率</th>
<th>次日开盘</th><th>次日收盘</th><th>开盘即涨停</th><th>开盘买收盘卖</th></tr></thead><tbody>{brow}</tbody></table></div>
<div class="note" style="margin-top:24px">数据来源：同花顺涨停数据中心、东方财富日K（前复权）。以上为历史统计结果，不构成投资建议。</div>
</main>"""


def make(log=print):
    """生成各个单页，再合并成带标签页的一页。"""
    date, watch = _watch()
    tabs = [("watch", "明日关注", watch)]
    try:
        tabs.append(("hot", "情绪主线", f"<main>{_main(hot.report())}</main>"))
    except Exception as ex:
        log(f"情绪主线页跳过：{ex}")
    try:
        tabs.append(("nshape", "N 型回调", f"<main>{_main(nshape.report())}</main>"))
    except Exception as ex:      # 需要趋势库里的全市场日K
        log(f"N 型回调页跳过：{ex}")
    tabs += [("zt", "涨停复盘", f"<main>{_main(report.make(date))}</main>"),
            ("review", "预测复盘", f"<main>{_main(review.make())}</main>")]
    try:
        tabs.append(("trend", "趋势突破", f"<main>{_main(trend.report())}</main>"))
    except Exception as ex:      # 趋势数据还没建时，其余三页照常出
        log(f"趋势突破页跳过：{ex}")
    nav = "".join(f'<button data-t="{k}">{name}</button>' for k, name, _ in tabs)
    body = "".join(f'<section data-p="{k}">{html}</section>' for k, _, html in tabs)
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>每日复盘 {date}</title><style>{CSS}{TAB_CSS}</style></head><body>
<nav class="tabs"><div>{nav}</div></nav>{body}<script>{JS}{TAB_JS}</script></body></html>"""
    out = ROOT / "reports"
    path = out / f"fupan_{date}.html"
    path.write_text(html, encoding="utf-8")
    shutil.copyfile(path, out / "index.html")
    return path
