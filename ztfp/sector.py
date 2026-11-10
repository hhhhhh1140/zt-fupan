"""板块强度：哪些板块近 5 日涨得最多、主力资金流入最多，里面谁是龙头。

数据来自东方财富：行业板块 + 概念板块的日K、每日主力净流入（只有最近约 120 个交易日）、成分股。
指数类、持股类这些不代表题材的概念板块不算。

    python -m ztfp sector     # 更新板块数据并打印今天的强势板块
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .db import connect
from .fetch import _get

H = {"Referer": "https://quote.eastmoney.com/"}
CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FFLOW = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
KBEG = "20250901"
N = 5              # 看近几日
TOP = 15           # 每天取前几个板块
JUNK = re.compile(
    r"昨日|融资融券|股通|MSCI|标普|富时|重仓|QFII|转债|预盈|预亏|预增|预减|破净|次新|百元|低价|证金|汇金|HS300|上证|中证|深成|深证|创业|"
    r"AH股|B股|ST|持股|养老金|国企改革|举牌|股权激励|送转|茅指数|宁组合|专精特新|北交所|央视|中字头|微盘|小盘|大盘|热股|"
    r"高股息|红利|分拆|壳资源|IPO|独角兽|GDR|同花顺|东方财富|机构|基金|社保|险资|回购|增持|减持|解禁|扭亏|亏损|参股|含.股|权重|长江三角|"
    r"成渝|京津冀|粤港|自贸|一带一路|雄安|西部大开发|振兴东北|海南|新疆|板块$|历史新高|题材股|多板|近期新高|连板|首板|打板|龙虎榜")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sec (code TEXT PRIMARY KEY, name TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS sec_k (code TEXT, date TEXT, close REAL, amount REAL, PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS sec_flow (code TEXT, date TEXT, main REAL, PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS sec_stock (code TEXT, stock TEXT, name TEXT, PRIMARY KEY (code, stock));
"""


def _pages(fs, fields, fid="f12"):
    out, pn = [], 1
    while True:
        j = _get(CLIST, dict(pn=pn, pz=100, po=1, np=1, fltt=2, invt=2, fid=fid, fs=fs, fields=fields), H)
        d = (j.get("data") or {}).get("diff") or []
        out += d
        if len(out) >= ((j.get("data") or {}).get("total") or 0) or not d:
            return out
        pn += 1


def _k(code, beg):
    j = _get(KLINE, dict(secid=f"90.{code}", klt=101, fqt=1, beg=beg, end="20500101", fields1="f1,f2,f3", fields2="f51,f53,f57"), H)
    rows = []
    for s in ((j.get("data") or {}).get("klines") or []):
        d, c, a = s.split(",")
        rows.append((code, d.replace("-", ""), float(c), float(a)))
    return rows


def _flow(code, lmt):
    j = _get(FFLOW, dict(lmt=lmt, klt=101, secid=f"90.{code}", fields1="f1,f2,f3,f7", fields2="f51,f52"), H)
    return [(code, s.split(",")[0].replace("-", ""), float(s.split(",")[1])) for s in ((j.get("data") or {}).get("klines") or [])]


def _members(code):
    return [(code, x["f12"], x["f14"]) for x in _pages(f"b:{code}", "f12,f14")]


def update(log=print):
    con = connect()
    con.executescript(SCHEMA)
    secs = [(x["f12"], x["f14"], kind) for t, kind in ((2, "行业"), (3, "概念")) for x in _pages(f"m:90+t:{t}", "f12,f14")]
    secs = [s for s in secs if not (s[2] == "概念" and JUNK.search(s[1]))]
    con.execute("DELETE FROM sec")
    con.executemany("INSERT OR REPLACE INTO sec VALUES (?,?,?)", secs)
    last = dict(con.execute("SELECT code, MAX(date) FROM sec_k GROUP BY code").fetchall())
    has_flow = {r[0] for r in con.execute("SELECT DISTINCT code FROM sec_flow")}

    def one(s):
        code = s[0]
        try:
            return _k(code, last.get(code, KBEG)), _flow(code, 10 if code in has_flow else 0)
        except Exception as ex:
            return ex, code

    t = time.time()
    fail = 0
    with ThreadPoolExecutor(6) as ex:
        for a, b in ex.map(one, secs):
            if isinstance(a, Exception):
                fail += 1
                continue
            con.executemany("INSERT OR REPLACE INTO sec_k VALUES (?,?,?,?)", a)
            con.executemany("INSERT OR REPLACE INTO sec_flow VALUES (?,?,?)", b)
    con.commit()
    log(f"板块 {len(secs)} 个，行情和资金流更新完成，失败 {fail} 个，用时 {time.time() - t:.0f}s")

    # 成分股：只抓上过榜的板块，够用来找龙头
    r = rank(con)
    r["rr"] = r.groupby("date").r.rank(ascending=False)       # 单看涨幅也靠前的（资金流出的情绪板块）
    cand = r[(r.rk <= TOP * 2) | (r.rr <= 20)]
    need = sorted(set(cand.code) - {x[0] for x in con.execute("SELECT DISTINCT code FROM sec_stock")}
                  | set(cand[cand.date == cand.date.max()].code))
    with ThreadPoolExecutor(4) as ex:
        for code, m in zip(need, ex.map(lambda c: _try(_members, c), need)):
            if m:
                con.execute("DELETE FROM sec_stock WHERE code=?", (code,))
                con.executemany("INSERT OR REPLACE INTO sec_stock VALUES (?,?,?)", m)
    con.commit()
    con.close()
    log(f"成分股更新 {len(need)} 个板块")


def _try(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def rank(con=None):
    """每天每个板块：近 N 日涨幅、近 N 日主力净流入、两者的综合排名。只有有资金流数据的日子才有排名。"""
    own = con is None
    con = con or connect()
    k = pd.read_sql("SELECT * FROM sec_k", con)
    fl = pd.read_sql("SELECT * FROM sec_flow", con)
    sec = pd.read_sql("SELECT * FROM sec", con)
    if own:
        con.close()
    k = k[k.code.isin(sec.code)].sort_values(["code", "date"])
    k["r"] = k.groupby("code").close.transform(lambda s: s / s.shift(N) - 1)
    k["amt"] = k.groupby("code").amount.transform(lambda s: s.rolling(N).sum())
    fl = fl.sort_values(["code", "date"])
    fl["f"] = fl.groupby("code").main.transform(lambda s: s.rolling(N).sum())
    d = k.merge(fl[["code", "date", "f", "main"]], on=["code", "date"]).dropna(subset=["r", "f"])
    d = d.merge(sec, on="code")
    d["fr"] = d.f / d.amt                                   # 净流入占成交额，避免大板块天然占优
    g = d.groupby("date")
    d["score"] = (g.r.rank(pct=True) + g.f.rank(pct=True) + g.fr.rank(pct=True)) / 3
    d.loc[(d.r <= 0) | (d.f <= 0), "score"] = 0             # 必须既涨又有资金流入
    d["rk"] = d.groupby("date").score.rank(ascending=False, method="first")
    return d


def top(date=None, n=TOP):
    """某天的强势板块，带上榜天数和板块内龙头。"""
    con = connect()
    d = rank(con)
    cal = sorted(d.date.unique())
    date = date or cal[-1]
    i = cal.index(date)
    mem = pd.read_sql("SELECT * FROM sec_stock", con)
    zt = pd.read_sql("SELECT date, code, name, lb FROM feat WHERE date>=? AND date<=?", con, params=(cal[max(i - N + 1, 0)], date))
    con.close()
    try:      # 全市场日K在趋势库里，有的话顺便算成分股的 5 日涨幅
        from .trend_data import connect as tc
        c2 = tc()
        kk = pd.read_sql("SELECT code, date, close FROM k WHERE date>=? AND date<=?", c2, params=(cal[max(i - N, 0)], date))
        c2.close()
        kk = kk.sort_values(["code", "date"]).groupby("code").close.agg(["first", "last"])
        r5 = (kk["last"] / kk["first"] - 1).to_dict()
    except Exception:
        r5 = {}
    # 上榜天数：往前数，连续在前 2n 名里（中间允许断 1 天）
    hist = d[d.rk <= n * 2].groupby("code").date.apply(set)
    t = d[(d.date == date) & (d.rk <= n) & (d.score > 0)].sort_values("rk")
    # 成分股高度重合的板块只留排名靠前的那个
    sets = mem.groupby("code").stock.apply(set).to_dict()
    keep = []
    for r in t.itertuples():
        s = sets.get(r.code, set())
        dup = next((k for k in keep if s and sets.get(k.code) and len(s & sets[k.code]) / min(len(s), len(sets[k.code])) >= 0.7), None)
        if dup is None:
            keep.append(r)
    out = []
    ztc = zt.groupby("code").agg(n=("date", "size"), lb=("lb", "max"), name=("name", "last"))
    today = zt[zt.date == date].set_index("code").lb
    for r in keep:
        days, j, miss = 0, i, 0
        while j >= 0 and miss <= 1:
            if cal[j] in hist.get(r.code, ()):
                days, miss = i - j + 1, 0
            else:
                miss += 1
            j -= 1
        m = mem[mem.code == r.code]
        m = m.assign(zt=m.stock.map(ztc.n).fillna(0), lb=m.stock.map(today).fillna(0), r5=m.stock.map(r5))
        m = m.sort_values(["zt", "lb", "r5"], ascending=False)
        out.append(dict(code=r.code, name=r.name, kind=r.kind, r=r.r, f=r.f, fr=r.fr, rk=int(r.rk), days=days,
                        since=cal[i - days + 1] if days else date, n_stock=len(m), n_zt=int((m.zt > 0).sum()),
                        n_today=int((m.lb > 0).sum()), lead=m[m.zt > 0].head(5), gain=m.sort_values("r5", ascending=False).head(3)))
    # 涨得多但主力资金在流出的：情绪炒作的板块常在这里（游资在买、大单在卖）
    o = d[(d.date == date) & (d.f < 0) & (d.r > 0)].sort_values("r", ascending=False).head(8)
    outflow = []
    for r in o.itertuples():
        m = mem[mem.code == r.code]
        m = m.assign(zt=m.stock.map(ztc.n).fillna(0), lb=m.stock.map(today).fillna(0), r5=m.stock.map(r5)).sort_values(["zt", "lb", "r5"], ascending=False)
        outflow.append(dict(code=r.code, name=r.name, kind=r.kind, r=r.r, f=r.f, n_zt=int((m.zt > 0).sum()), lead=m[m.zt > 0].head(5)))
    return out, outflow, date
