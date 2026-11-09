"""SQLite 存储。所有表都以 (date, code) 为核心，日期统一用 YYYYMMDD 字符串。"""
import pathlib
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "ztfp.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar (date TEXT PRIMARY KEY);

-- kind: zt=涨停池  zb=炸板池  dt=跌停池
CREATE TABLE IF NOT EXISTS pool (
    date TEXT, kind TEXT, code TEXT, name TEXT, market_type TEXT,
    latest REAL, change_rate REAL, turnover_rate REAL, currency_value REAL,
    first_min REAL, last_min REAL,          -- 首封/末封：开盘后第几分钟(午休已扣除)
    open_num INTEGER, order_amount REAL,
    limit_up_type TEXT, suc_rate REAL,
    high_days TEXT, days_n INTEGER, boards_n INTEGER,   -- "6天4板" -> 6, 4
    reason_type TEXT, is_again_limit INTEGER, is_new INTEGER,
    PRIMARY KEY (date, kind, code)
);

-- 同花顺连板天梯：真正的连续涨停天数
CREATE TABLE IF NOT EXISTS ladder (
    date TEXT, code TEXT, continue_num INTEGER, PRIMARY KEY (date, code)
);

CREATE TABLE IF NOT EXISTS block (
    date TEXT, block_code TEXT, name TEXT, limit_up_num INTEGER,
    continuous_plate_num INTEGER, days INTEGER, change REAL,
    PRIMARY KEY (date, block_code)
);
CREATE TABLE IF NOT EXISTS block_stock (
    date TEXT, block_code TEXT, code TEXT, PRIMARY KEY (date, block_code, code)
);

-- 前复权日K；指数用 code='IDX'
CREATE TABLE IF NOT EXISTS kline (
    code TEXT, date TEXT, open REAL, close REAL, high REAL, low REAL,
    volume REAL, amount REAL, turnover REAL,
    PRIMARY KEY (code, date)
);

-- 已完整采集的交易日
CREATE TABLE IF NOT EXISTS days_done (date TEXT PRIMARY KEY);
"""


def connect():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.executescript(SCHEMA)
    return con
