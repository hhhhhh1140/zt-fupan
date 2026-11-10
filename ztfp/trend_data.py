"""趋势复盘的数据层：全市场股票列表 + 2023 年起的前复权日K，存在 data/trend.db。"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from .db import DATA
from .fetch import S, _get, fetch_kline, now_cn

DB = DATA / "trend.db"
KLINE_BEG = "20230101"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock (
    code TEXT PRIMARY KEY, name TEXT, industry TEXT, float_cap REAL, list_date TEXT
);
CREATE TABLE IF NOT EXISTS k (
    code TEXT, date TEXT, open REAL, close REAL, high REAL, low REAL,
    volume REAL, amount REAL, turnover REAL,
    PRIMARY KEY (code, date)
) WITHOUT ROWID;
"""


def connect():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.executescript(SCHEMA)
    return con


def fetch_universe():
    """沪深 A 股（主板/创业板/科创板），剔除 ST、退市整理。"""
    params = dict(pn=1, pz=100, po=1, np=1, fltt=2, invt=2, fid="f12",
                  fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23", fields="f12,f14,f20,f21,f26,f100")
    rows = []
    while True:
        d = _get("https://push2.eastmoney.com/api/qt/clist/get", params).get("data")
        if not d or not d.get("diff"):
            break
        rows += d["diff"]
        if len(rows) >= d["total"]:
            break
        params["pn"] += 1
    out = []
    for r in rows:
        name = str(r["f14"])
        if "ST" in name or "退" in name:
            continue
        cap = r["f21"] if isinstance(r["f21"], (int, float)) else None
        out.append((r["f12"], name, r.get("f100") if r.get("f100") != "-" else None, cap, str(r.get("f26") or "")))
    return out


def update(log=print):
    con = connect()
    uni = fetch_universe()
    con.execute("DELETE FROM stock")
    con.executemany("INSERT INTO stock VALUES (?,?,?,?,?)", uni)
    con.commit()

    now = now_cn()
    today = now.strftime("%Y%m%d")
    closed = now.hour * 60 + now.minute >= 15 * 60 + 10
    # 以上证指数的最新交易日为准，已经是最新的股票不重拉
    idx = fetch_kline_idx()
    if not closed:
        idx = [r for r in idx if r[0] < today]
    con.execute("DELETE FROM k WHERE code='IDX'")
    con.executemany("INSERT INTO k VALUES ('IDX',?,?,?,?,?,?,?,?)", idx)
    latest = idx[-1][0]
    fresh = {r[0] for r in con.execute("SELECT code FROM k WHERE date=?", (latest,))}
    codes = [u[0] for u in uni if u[0] not in fresh]
    log(f"股票 {len(uni)} 只，待更新日K {len(codes)} 只（最新交易日 {latest}）")

    def one(code):
        try:
            return code, fetch_kline(code, KLINE_BEG)
        except Exception as e:
            return code, e

    fails = []
    with ThreadPoolExecutor(8) as ex:
        for i, (code, rows) in enumerate(ex.map(one, codes), 1):
            if isinstance(rows, Exception) or not rows:
                fails.append(code)
                continue
            if not closed:
                rows = [r for r in rows if r[0] < today]
            # 前复权价会随分红变化，所以整只重写
            con.execute("DELETE FROM k WHERE code=?", (code,))
            con.executemany("INSERT INTO k VALUES (?,?,?,?,?,?,?,?,?)", [(code, *r) for r in rows])
            if i % 500 == 0:
                con.commit()
                log(f"  日K {i}/{len(codes)}")
    con.commit()
    con.close()
    log(f"日K完成，失败 {len(fails)} 只" + (f": {fails[:20]}" if fails else ""))
    return fails


def fetch_kline_idx():
    from .fetch import _kline_em
    return _kline_em("1.000001", KLINE_BEG)
