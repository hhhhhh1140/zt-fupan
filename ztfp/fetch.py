"""数据采集：同花顺涨停池/炸板池/跌停池/连板天梯/题材榜 + 东财(腾讯兜底)前复权日K。

国内接口走系统代理会握手失败，所以 Session 关闭 trust_env，强制直连。
"""
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .db import connect

S = requests.Session()
S.trust_env = False  # 忽略 HTTP(S)_PROXY，直连
S.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"})

THS = "https://data.10jqka.com.cn/dataapi/limit_up/"
THS_FIELDS = "199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004"
THS_FILTER = "HS,GEM2STAR"
CN = dt.timezone(dt.timedelta(hours=8))


class BadDate(Exception):
    """同花顺不再保留该日期的数据。"""


def _get(url, params=None, headers=None, tries=4):
    err = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # 网络抖动/限流：退避重试
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"请求失败 {url} {params}: {err}")


def _ths(path, date, **extra):
    params = {"filter": THS_FILTER, "date": date, **extra}
    j = _get(THS + path, params, {"Referer": "https://data.10jqka.com.cn/datacenterph/limitup/limtupInfo.html"})
    if j.get("status_code") != 0:
        if "date" in str(j.get("status_msg", "")):
            raise BadDate(date)
        raise RuntimeError(f"同花顺 {path} {date}: {j.get('status_msg')}")
    return j["data"]


def _minute(ts):
    """epoch 秒 -> 开盘后第几分钟（9:30=0，午休扣除，集合竞价记 0）。"""
    if not ts:
        return None
    t = dt.datetime.fromtimestamp(int(ts), CN)
    m = t.hour * 60 + t.minute + t.second / 60 - 570
    if m > 120:  # 11:30 之后
        m = max(120, m - 90)
    return round(max(0, m), 2)


def _pool_rows(kind, path, date, order_field):
    rows, page = [], 1
    while True:
        d = _ths(path, date, page=page, limit=200, field=THS_FIELDS, order_field=order_field, order_type=0)
        for x in d["info"]:
            v = x.get("high_days_value") or 0
            rows.append((
                date, kind, x["code"], x["name"], x.get("market_type"),
                x.get("latest"), x.get("change_rate"), x.get("turnover_rate"), x.get("currency_value"),
                _minute(x.get("first_limit_up_time")), _minute(x.get("last_limit_up_time")),
                x.get("open_num"), x.get("order_amount"),
                x.get("limit_up_type"), x.get("limit_up_suc_rate"),
                x.get("high_days"), v & 0xFFFF or None, v >> 16 or None,
                x.get("reason_type"), x.get("is_again_limit"), x.get("is_new"),
            ))
        if page * 200 >= d["page"]["total"] or not d["info"]:
            return rows
        page += 1


def fetch_day(date):
    """采集一个交易日的全部同花顺数据，返回各表的行。"""
    pool = (_pool_rows("zt", "limit_up_pool", date, "330324")
            + _pool_rows("zb", "open_limit_pool", date, "199112")
            + _pool_rows("dt", "lower_limit_pool", date, "199112"))
    ladder = [(date, s["code"], s["continue_num"])
              for lv in _ths("continuous_limit_up", date) for s in lv["code_list"]]
    blocks, bstocks = [], []
    for b in _ths("block_top", date):
        blocks.append((date, b["code"], b["name"], b.get("limit_up_num"),
                       b.get("continuous_plate_num"), b.get("days"), b.get("change")))
        bstocks += [(date, b["code"], s["code"]) for s in b.get("stock_list", [])]
    return pool, ladder, blocks, bstocks


def _secid(code):
    return ("1." if code[0] in "69" else "0.") + code


def _kline_em(secid, beg):
    j = _get("https://push2his.eastmoney.com/api/qt/stock/kline/get", {
        "secid": secid, "fields1": "f1,f2,f3", "fields2": "f51,f52,f53,f54,f55,f56,f57,f61",
        "klt": 101, "fqt": 1, "beg": beg, "end": "20500101"}, tries=2)
    out = []
    for line in (j.get("data") or {}).get("klines") or []:
        d, o, c, h, l, v, a, t = line.split(",")
        out.append((d.replace("-", ""), float(o), float(c), float(h), float(l), float(v), float(a), float(t)))
    return out


def _kline_tx(code, beg):
    sym = ("sh" if code[0] in "69" else "sz") + code
    j = _get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
             {"param": f"{sym},day,{beg[:4]}-{beg[4:6]}-{beg[6:]},,640,qfq"})
    d = (j.get("data") or {}).get(sym) or {}
    out = []
    for r in d.get("qfqday") or d.get("day") or []:
        out.append((r[0].replace("-", ""), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), None, None))
    return out


def fetch_kline(code, beg):
    """前复权日K。东财字段更全(成交额/换手)，失败时用腾讯兜底。"""
    try:
        rows = _kline_em(_secid(code), beg)
        if rows:
            return rows
    except Exception:
        pass
    return _kline_tx(code, beg)


def now_cn():
    return dt.datetime.now(CN)


def update(kline_beg="20250601", log=print):
    """增量更新：补齐缺失交易日的涨停数据，并刷新相关股票的日K。"""
    con = connect()

    idx = _kline_em("1.000001", kline_beg)
    con.executemany("INSERT OR REPLACE INTO kline VALUES ('IDX',?,?,?,?,?,?,?,?)", idx)
    con.executemany("INSERT OR IGNORE INTO calendar VALUES (?)", [(r[0],) for r in idx])
    con.commit()

    now = now_cn()
    today = now.strftime("%Y%m%d")
    closed = now.hour * 60 + now.minute >= 15 * 60 + 10  # 收盘后池子才完整
    done = {r[0] for r in con.execute("SELECT date FROM days_done")}
    cutoff = (now - dt.timedelta(days=372)).strftime("%Y%m%d")  # 同花顺只留约一年
    todo = [r[0] for r in idx if r[0] >= cutoff and r[0] not in done and (r[0] < today or closed)]
    log(f"待采集交易日 {len(todo)} 个")

    def one(d):
        try:
            return d, fetch_day(d)
        except BadDate:
            return d, None

    n = 0
    with ThreadPoolExecutor(4) as ex:
        for d, res in ex.map(one, todo):
            if res is None:
                continue
            pool, ladder, blocks, bstocks = res
            for t in ("pool", "ladder", "block", "block_stock"):
                con.execute(f"DELETE FROM {t} WHERE date=?", (d,))
            con.executemany("INSERT OR REPLACE INTO pool VALUES (%s)" % ",".join("?" * 21), pool)
            con.executemany("INSERT OR REPLACE INTO ladder VALUES (?,?,?)", ladder)
            con.executemany("INSERT OR REPLACE INTO block VALUES (?,?,?,?,?,?,?)", blocks)
            con.executemany("INSERT OR REPLACE INTO block_stock VALUES (?,?,?)", bstocks)
            con.execute("INSERT OR REPLACE INTO days_done VALUES (?)", (d,))
            con.commit()
            n += 1
            if n % 20 == 0:
                log(f"  涨停数据 {n}/{len(todo)} (到 {d})")
    log(f"涨停数据新增 {n} 天")

    # 日K：没拉过的股票全拉；近 25 个交易日出现过的股票重拉(更新前向标签 + 复权变化)
    cal = [r[0] for r in con.execute("SELECT date FROM calendar ORDER BY date")]
    recent = cal[-25] if len(cal) >= 25 else cal[0]
    have = {r[0] for r in con.execute("SELECT DISTINCT code FROM kline")}
    latest = cal[-1]
    fresh = {r[0] for r in con.execute("SELECT code FROM kline WHERE date=?", (latest,))}
    codes = set()
    for code, last in con.execute("SELECT code, MAX(date) FROM pool WHERE kind IN ('zt','zb') GROUP BY code"):
        if code not in have or (last >= recent and code not in fresh):
            codes.add(code)
    codes = sorted(codes)
    log(f"待更新日K {len(codes)} 只")

    def k(code):
        try:
            return code, fetch_kline(code, kline_beg)
        except Exception as e:
            return code, e

    fails = []
    with ThreadPoolExecutor(6) as ex:
        for i, (code, rows) in enumerate(ex.map(k, codes), 1):
            if isinstance(rows, Exception) or not rows:
                fails.append(code)
                continue
            con.execute("DELETE FROM kline WHERE code=?", (code,))
            con.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?,?,?)", [(code, *r) for r in rows])
            if i % 200 == 0:
                con.commit()
                log(f"  日K {i}/{len(codes)}")
    con.commit()
    log(f"日K完成，失败 {len(fails)} 只" + (f": {fails[:20]}" if fails else ""))
    con.close()
    return fails
