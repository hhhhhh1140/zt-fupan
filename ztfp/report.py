"""生成单文件 HTML 复盘页：reports/zt_YYYYMMDD.html"""
import json
from html import escape as e

import numpy as np
import pandas as pd

from .db import DATA, ROOT, connect

CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1c1b19;--mut:#77736b;--line:#e4e1da;--up:#c8322b;--dn:#1f8a4c;--bar:#f3d6d3;--bar2:#dbe4f3;--acc:#2b5fc8}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--card:#1f1f1c;--ink:#ecebe6;--mut:#99958c;--line:#33322e;--up:#ef6a61;--dn:#4fbf7f;--bar:#5a2a27;--bar2:#263652;--acc:#7da4f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:24px;margin:0}h2{font-size:16px;margin:36px 0 10px}h3{font-size:13px;margin:0 0 6px;color:var(--mut);font-weight:500}
.sub{color:var(--mut);font-size:13px;margin-top:4px}.note{color:var(--mut);font-size:12.5px;margin:6px 0 10px;max-width:900px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums;line-height:1.25}.tile small{color:var(--mut);font-size:12px}
.tile svg{display:block;width:100%;height:28px;margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{font-weight:500;color:var(--mut);font-size:12px;position:sticky;top:0;background:var(--card)}
th.s{cursor:pointer}th.s:hover{color:var(--ink)}td.l,th.l{text-align:left}td.r{white-space:normal;min-width:200px;max-width:320px;color:var(--mut);font-size:12.5px;text-align:left}
tr:last-child td{border-bottom:0}.code{color:var(--mut);font-size:12px;margin-left:6px}
.p{position:relative;min-width:76px}.p i{position:absolute;left:0;top:5px;bottom:5px;background:var(--bar);border-radius:3px}.p.w i{background:var(--bar2)}.p span{position:relative;padding-left:6px;font-weight:600}
.up{color:var(--up)}.dn{color:var(--dn)}.tag{display:inline-block;font-size:11.5px;padding:0 6px;border-radius:4px;border:1px solid var(--line);color:var(--mut);margin-left:4px}
.rung{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}.rung:last-child{border:0}.rung>b{flex:0 0 46px;font-size:15px;color:var(--up)}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--line);border-radius:6px;padding:3px 8px;font-size:13px}.chip em{font-style:normal;color:var(--up);font-weight:600;margin-left:6px}.chip small{color:var(--mut);margin-left:6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
"""

JS = """
document.querySelectorAll('th.s').forEach(th=>th.onclick=()=>{const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),b=t.tBodies[0];
const d=th.dataset.d=th.dataset.d==='1'?'-1':'1';[...b.rows].sort((x,y)=>{const a=x.cells[i].dataset.v,c=y.cells[i].dataset.v;
return (isNaN(a)||isNaN(c)?String(a).localeCompare(c):a-c)*d}).forEach(r=>b.appendChild(r))});
"""


def _hm(m):
    if m is None or pd.isna(m):
        return "-"
    t = 570 + m if m <= 120 else 780 + (m - 120)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _pct(v, d=1, sign=False):
    if v is None or pd.isna(v):
        return "-"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _col(v):
    return "" if v is None or pd.isna(v) or v == 0 else ("up" if v > 0 else "dn")


def _spark(vals):
    v = [x for x in vals if not pd.isna(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * 200 / (len(v) - 1):.1f},{26 - (x - lo) / rng * 24:.1f}" for i, x in enumerate(v))
    return (f'<svg viewBox="0 0 200 28" preserveAspectRatio="none"><polyline points="{pts}" fill="none" '
            f'stroke="var(--acc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>')


def _prob(v, cls=""):
    if pd.isna(v):
        return '<td data-v="-1">-</td>'
    return f'<td class="p {cls}" data-v="{v:.4f}"><i style="width:{min(v, 1) * 100:.0f}%"></i><span>{v * 100:.0f}%</span></td>'


def _group_table(df, key, title, order=None):
    g = df.groupby(key, observed=True).agg(
        n=("code", "size"), lb=("y_lb", "mean"), no=("n_open", "mean"), nc=("n_close", "mean"),
        sw=("y_swing", "mean"), f20=("fmax20", "median"))
    if order:
        g = g.reindex([o for o in order if o in g.index])
    rows = "".join(
        f'<tr><td class="l">{e(str(k))}</td><td>{int(r.n)}</td><td><b>{_pct(r.lb)}</b></td>'
        f'<td class="{_col(r.no)}">{_pct(r.no, 1, True)}</td><td class="{_col(r.nc)}">{_pct(r.nc, 1, True)}</td>'
        f'<td><b>{_pct(r.sw)}</b></td><td>{_pct(r.f20, 0, True)}</td></tr>' for k, r in g.iterrows())
    return (f'<div class="card scroll"><h3>{title}</h3><table><thead><tr><th class="l"></th><th>样本</th><th>次日连板率</th>'
            f'<th>次日开盘</th><th>次日收盘</th><th>波段达成率</th><th>20日最高涨幅中位</th></tr></thead><tbody>{rows}</tbody></table></div>')


def make(date=None):
    con = connect()
    f = pd.read_sql("SELECT * FROM feat", con).merge(pd.read_sql("SELECT * FROM pred", con), on=["date", "code"], how="left")
    mk = pd.read_sql("SELECT * FROM market ORDER BY date", con)
    dates = sorted(f.date.unique())
    date = date or dates[-1]
    if date not in dates:
        raise SystemExit(f"没有 {date} 的数据，可选范围 {dates[0]} ~ {dates[-1]}")
    prev = dates[dates.index(date) - 1] if dates.index(date) else None
    f["open_num"] = f.open_num.fillna(0)
    d = f[f.date == date].copy()
    hist = f[f.date < date]              # 统计只用当天之前的样本
    blocks = pd.read_sql("SELECT * FROM block WHERE date=? ORDER BY limit_up_num DESC", con, params=(date,))
    bst = pd.read_sql("SELECT * FROM block_stock WHERE date=?", con, params=(date,))
    con.close()
    met = json.loads((DATA / "metrics.json").read_text())

    # 同板数历史基准
    hist = hist.assign(lbb=hist.lb.clip(upper=5))
    base = hist.groupby("lbb").agg(lb=("y_lb", "mean"), sw=("y_swing", "mean"))
    d["base_lb"] = d.lb.clip(upper=5).map(base.lb)
    d["base_sw"] = d.lb.clip(upper=5).map(base.sw)
    done = d.y_lb.notna().any()          # 这一天的次日结果是否已知

    # ---------- 情绪
    m = mk[mk.date <= date]
    cur, h60 = m.iloc[-1], m.tail(60)

    def tile(label, col, fmt, hint=""):
        v = cur[col]
        rank = (m[col] < v).mean() if not pd.isna(v) else np.nan
        return (f'<div class="tile"><small>{label}</small><b>{fmt(v)}</b>'
                f'<small>高于过去一年 {_pct(rank, 0)} 的交易日{hint}</small>{_spark(h60[col])}</div>')

    tiles = "".join([
        tile("涨停家数", "n_zt", lambda v: f"{v:.0f}"),
        tile("炸板率", "zb_rate", _pct),
        tile("跌停家数", "n_dt", lambda v: f"{v:.0f}"),
        tile("最高连板", "max_lb", lambda v: f"{v:.0f} 板"),
        tile("连板家数", "n_lb2", lambda v: f"{v:.0f}"),
        tile("昨日涨停今日平均涨幅", "prem_prev", lambda v: _pct(v, 1, True)),
        tile("昨日涨停晋级率", "promo", _pct),
    ])

    # ---------- 连板天梯
    rungs = ""
    for lb, g in d[d.lb >= 2].groupby("lb", sort=False):
        chips = "".join(
            f'<span class="chip">{e(r["name"])}<small>{e((r.reason_type or "").split("+")[0])}</small><em>{_pct(r.p_lb, 0)}</em></span>'
            for _, r in g.sort_values("p_lb", ascending=False).iterrows())
        rungs += f'<div class="rung"><b>{lb} 板</b><div class="chips">{chips}</div></div>'
    rungs = rungs or '<div class="note">当日没有连板股。</div>'

    # ---------- 个股表
    rows = ""
    for _, r in d.sort_values("p_lb", ascending=False).iterrows():
        res = ""
        if done:
            ok = r.y_lb == 1
            res = (f'<td data-v="{r.n_close if not pd.isna(r.n_close) else -9}" class="{_col(r.n_close)}">'
                   f'{"连板" if ok else _pct(r.n_close, 1, True)}</td>')
        t20 = '<span class="tag">20cm</span>' if r.is20 else ""
        rows += (
            f'<tr><td class="l" data-v="{e(r["name"])}">{e(r["name"])}<span class="code">{r.code}</span>{t20}</td>'
            f'<td data-v="{r.lb}">{r.lb}</td><td class="l" data-v="{r.boards_n or 0}">{e(r.high_days or "")}</td>'
            f'<td class="l" data-v="{r.type_code}">{e(r.limit_up_type or "")}</td>'
            f'<td data-v="{r.first_min}">{_hm(r.first_min)}</td><td data-v="{r.open_num or 0}">{int(r.open_num or 0)}</td>'
            f'<td data-v="{r.seal_ratio}">{_pct(r.seal_ratio, 2)}</td><td data-v="{r.turnover_rate}">{r.turnover_rate:.1f}%</td>'
            f'<td data-v="{r.currency_value}">{r.currency_value / 1e8:.0f}亿</td>'
            f'{_prob(r.p_lb)}<td data-v="{r.base_lb}">{_pct(r.base_lb, 0)}</td>'
            f'{_prob(r.p_swing, "w")}<td data-v="{r.base_sw}">{_pct(r.base_sw, 0)}</td>{res}'
            f'<td class="r">{e(r.reason_type or "")}</td></tr>')
    res_th = '<th class="s">次日实际</th>' if done else ""
    stock_table = (
        '<div class="card scroll"><table><thead><tr><th class="l s">股票</th><th class="s">连板</th><th class="l s">几天几板</th>'
        '<th class="l s">类型</th><th class="s">首封</th><th class="s">开板</th><th class="s">封单/流通</th><th class="s">换手</th>'
        '<th class="s">流通市值</th><th class="s">连板概率</th><th class="s">同板数基准</th><th class="s">波段概率</th>'
        f'<th class="s">同板数基准</th>{res_th}<th class="l">涨停原因</th></tr></thead><tbody>{rows}</tbody></table></div>')

    # ---------- 题材梯队
    info = d.set_index("code")
    themes = ""
    for _, b in blocks.head(8).iterrows():
        codes = [c for c in bst[bst.block_code == b.block_code].code if c in info.index]
        codes.sort(key=lambda c: (-info.lb[c], info.first_min[c]))
        chips = "".join(f'<span class="chip">{e(info["name"][c])}'
                        + (f'<small>{info.lb[c]}板</small>' if info.lb[c] >= 2 else "")
                        + f'<em>{_pct(info.p_lb[c], 0)}</em></span>' for c in codes)
        themes += (f'<div class="rung"><b style="flex-basis:150px;color:var(--ink);font-size:14px">{e(b["name"])}'
                   f'<br><small style="color:var(--mut);font-weight:400">{b.limit_up_num} 只涨停 · 持续 {b.days} 天</small></b>'
                   f'<div class="chips">{chips}</div></div>')

    # ---------- 上一交易日预测回看
    review = ""
    if prev:
        pv = f[(f.date == prev) & f.p_lb.notna()].sort_values("p_lb", ascending=False)
        if len(pv):
            top = pv.head(10)
            trs = "".join(
                f'<tr><td class="l">{e(r["name"])}<span class="code">{r.code}</span></td><td>{r.lb}</td>{_prob(r.p_lb)}'
                f'<td class="{"up" if r.y_lb == 1 else ""}">{"晋级" if r.y_lb == 1 else "未晋级"}</td>'
                f'<td class="{_col(r.n_open)}">{_pct(r.n_open, 1, True)}</td><td class="{_col(r.n_close)}">{_pct(r.n_close, 1, True)}</td></tr>'
                for _, r in top.iterrows())
            review = (
                f'<h2>上一交易日（{prev}）预测回看</h2><div class="note">连板概率最高的 10 只里 {int(top.y_lb.sum())} 只晋级；'
                f'当天全部 {len(pv)} 只涨停股中 {int(pv.y_lb.sum())} 只晋级（{_pct(pv.y_lb.mean(), 0)}）。</div>'
                '<div class="card scroll"><table><thead><tr><th class="l">股票</th><th>连板</th><th>当时给的连板概率</th>'
                f'<th>结果</th><th>今日开盘</th><th>今日收盘</th></tr></thead><tbody>{trs}</tbody></table></div>')

    # ---------- 历史分组统计
    hs = hist.copy()
    hs["板数"] = hs.lbb.map({1: "首板", 2: "2 板", 3: "3 板", 4: "4 板", 5: "5 板及以上"})
    hs["首封时段"] = pd.cut(hs.first_min, [-1, 0.01, 10, 60, 120, 180, 241],
                        labels=["竞价/开盘即封", "9:40 前", "9:40–10:30", "10:30–11:30", "13:00–14:00", "14:00 后"])
    hs["开板"] = hs.open_num.fillna(0).clip(upper=3).map({0: "未开板", 1: "开板 1 次", 2: "开板 2 次", 3: "开板 3 次以上"})
    hs["情绪"] = pd.qcut(hs.zb_rate, 3, labels=["炸板率低（情绪好）", "炸板率中", "炸板率高（情绪差）"])
    stats = "".join([
        _group_table(hs, "板数", "按连板数", ["首板", "2 板", "3 板", "4 板", "5 板及以上"]),
        _group_table(hs, "limit_up_type", "按涨停类型", ["换手板", "T字板", "一字板"]),
        _group_table(hs, "首封时段", "按首次封板时间"),
        _group_table(hs, "开板", "按开板次数", ["未开板", "开板 1 次", "开板 2 次", "开板 3 次以上"]),
        _group_table(hs, "情绪", "按当日炸板率"),
        _group_table(hs.assign(b=hs.is20.map({0: "主板 10cm", 1: "创业板/科创板 20cm"})), "b", "按涨跌幅限制"),
    ])

    # ---------- 模型回测
    def model_card(key, title):
        x = met["metrics"].get(key)
        if not x:
            return ""
        bk = "".join(f'<tr><td class="l">第 {i + 1} 档</td><td>{b["n"]}</td><td>{_pct(b["pred"], 0)}</td><td><b>{_pct(b["actual"], 0)}</b></td></tr>'
                     for i, b in enumerate(x["buckets"]))
        imp = "、".join(i["name"] for i in met["importance"].get(key, [])[:6])
        return (f'<div class="card scroll"><h3>{title}</h3><div class="note">样本外 {x["n"]} 条 · 基准率 {_pct(x["base"])} · '
                f'AUC {x["auc"]:.3f} · 概率最高的 10% 实际命中 {_pct(x["top10_actual"])}</div>'
                '<table><thead><tr><th class="l">按预测概率五等分</th><th>样本</th><th>平均预测</th><th>实际发生</th></tr></thead>'
                f'<tbody>{bk}</tbody></table><div class="note">影响最大的因素：{imp}</div></div>')

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>涨停复盘 {date}</title><style>{CSS}</style></head><body><main>
<h1>涨停复盘 · {date[:4]}-{date[4:6]}-{date[6:]}</h1>
<div class="sub">{len(d)} 只涨停（不含 ST 与北交所） · 历史样本 {len(hist)} 条 / {hist.date.nunique()} 个交易日</div>
<div class="tiles">{tiles}</div>

<h2>连板天梯</h2><div class="note">红色数字是模型给出的次日继续涨停概率。</div><div class="card">{rungs}</div>

<h2>题材梯队</h2><div class="card">{themes}</div>

<h2>全部涨停股</h2>
<div class="note">点表头排序。"同板数基准"是历史上同样连板数的股票的实际比例，用来对照模型概率高了还是低了。
波段 = 之后 20 个交易日内最高价较今日收盘涨 20% 以上，且此前不跌破今日起涨点。一字板次日通常买不进，概率高不等于可操作。</div>
{stock_table}
{review}

<h2>历史统计（{hist.date.min()} ~ {hist.date.max()}）</h2>
<div class="note">次日开盘/收盘是相对涨停价的平均涨跌幅。</div><div class="grid2">{stats}</div>

<h2>模型可信度</h2>
<div class="note">每 5 个交易日用此前的数据重新训练，再预测之后 5 天，下面全部是这种"没见过答案"的预测与实际的对比（自 {met["oos_from"]} 起）。
AUC 0.5 等于瞎猜，0.7 左右算有明显区分力；低于 0.6 的模型，其概率只比同板数基准略有参考价值。</div>
<div class="grid2">{model_card("lb", "次日连板概率")}{model_card("swing", "波段概率")}</div>
<div class="note" style="margin-top:24px">数据来源：同花顺涨停数据中心、东方财富日K（前复权）。以上为历史统计结果，不构成投资建议。</div>
</main><script>{JS}</script></body></html>"""
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"zt_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path
