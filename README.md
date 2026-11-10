# 涨停复盘工具
# A股打板复盘
每个交易日 15:10 之后，在本目录运行：

```bash
/opt/anaconda3/bin/python3 -m ztfp daily
```

要用 Anaconda 的 Python：系统自带的 `python3` 没有 pandas，会在采集完之后报错。

依次完成：增量采集 → 重建特征/标签 → 滚动回测 → 生成 `reports/zt_日期.html`。

其它命令：`update`（只采集）、`build`、`train`、`report 20261106`（重看任意历史日）。

- 数据库：`data/ztfp.db`（SQLite）。同花顺只保留约一年的涨停数据，库里存下来的会一直保留，所以越跑样本越多。
- 依赖：requests、pandas、numpy、scikit-learn（Anaconda 自带）。
- 采集强制直连，不走系统代理。
- 范围：沪深主板、创业板、科创板；不含 ST 和北交所。
- 标签定义、特征列表在 `ztfp/features.py` 顶部，可改 `SWING_DAYS` / `SWING_GAIN`。
