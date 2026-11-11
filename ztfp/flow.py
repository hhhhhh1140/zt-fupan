"""个股主力资金流（东方财富，只有最近约 120 个交易日，存下来之后一直保留）。

只采涨停过的股票：近 130 个交易日里上过涨停池的。

    python -m ztfp flow
"""
import time
from concurrent.futures import ThreadPoolExecutor

from .db import connect
from .fetch import _get

URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
H = {"Referer": "https://quote.eastmoney.com/"}
WORKERS = 2          # 并发高了会被东方财富限流
SCHEMA = ("CREATE TABLE IF NOT EXISTS stk_flow (code TEXT, date TEXT, main REAL, super REAL, big REAL, "
          "PRIMARY KEY (code, date))")


def _one(a):
    code, lmt = a
    try:
        j = _get(URL, dict(lmt=lmt, klt=101, secid=f"{1 if code[0] == '6' else 0}.{code}", fields1="f1,f2,f3,f7",
                           fields2="f51,f52,f55,f56"), H, tries=2)
        out = []
        for s in ((j.get("data") or {}).get("klines") or []):
            d, m, b, sp = s.split(",")
            out.append((code, d.replace("-", ""), float(m), float(sp), float(b)))
        time.sleep(0.15)
        return out
    except Exception:
        return None


def update(log=print):
    con = connect()
    con.execute(SCHEMA)
    dates = [r[0] for r in con.execute("SELECT DISTINCT date FROM feat ORDER BY date")]
    last = dates[-1]
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM feat WHERE date>=?", (dates[max(len(dates) - 130, 0)],))]
    have = dict(con.execute("SELECT code, MAX(date) FROM stk_flow GROUP BY code").fetchall())
    todo = [(c, 0 if c not in have else 15) for c in codes if have.get(c, "") < last]
    t, fail, n, bad = time.time(), 0, 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for rows in ex.map(_one, todo):
            if rows is None:
                fail += 1
                bad += 1
                if bad >= 20:      # 连续失败多半是被限流了，停下来，下次接着抓
                    log("个股资金流：接口连续失败，可能被限流，先停下，过一会儿再跑")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                continue
            bad = 0
            con.executemany("INSERT OR REPLACE INTO stk_flow VALUES (?,?,?,?,?)", rows)
            n += 1
            if n % 50 == 0:
                con.commit()
    con.commit()
    con.close()
    log(f"个股资金流：更新 {n} 只，失败 {fail} 只，剩 {len(todo) - n - fail} 只没抓，用时 {time.time() - t:.0f}s")
