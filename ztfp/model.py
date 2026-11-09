"""概率模型 + 回测。

做法：滚动前推(walk-forward)。每 5 个交易日重训一次，每一段的预测都由"只见过这段之前数据"的
模型给出，所以 pred 表里的概率（包括最新一天）全部是样本外的，回测指标也只在这些预测上计算。
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from .db import DATA, connect
from .features import FEATURE_CN, FEATURES, SWING_DAYS

MIN_TRAIN_DAYS = 80
STEP = 5
TARGETS = {"lb": "y_lb", "swing": "y_swing"}


def _clf():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.04, max_iter=150, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0)


def _fit(df, y):
    d = df[df[y].notna()]
    return _clf().fit(d[FEATURES].astype(float), d[y].astype(int)) if d[y].nunique() == 2 else None


def _eval(p, y):
    """样本外预测的质量：AUC、Brier、按预测概率五等分后的实际命中率。"""
    m = p.notna() & y.notna()
    p, y = p[m], y[m]
    if len(y) < 200 or y.nunique() < 2:
        return None
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    buckets = [{"pred": float(p[q == i].mean()), "actual": float(y[q == i].mean()), "n": int((q == i).sum())}
               for i in sorted(q.unique())]
    top = p >= p.quantile(0.9)
    return {"n": int(len(y)), "base": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "brier_base": float(((y - y.mean()) ** 2).mean()),
            "buckets": buckets, "top10_actual": float(y[top].mean()), "top10_n": int(top.sum())}


def train(log=print):
    con = connect()
    df = pd.read_sql("SELECT * FROM feat", con)
    dates = sorted(df.date.unique())
    pos = {d: i for i, d in enumerate(dates)}
    df["di"] = df.date.map(pos)
    pred = df[["date", "code"]].copy()
    pred["p_lb"], pred["p_swing"], pred["oos"] = np.nan, np.nan, 0

    for start in range(MIN_TRAIN_DAYS, len(dates), STEP):
        test = df[(df.di >= start) & (df.di < start + STEP)]
        for key, y in TARGETS.items():
            # 波段标签要用到之后 20 天的行情，训练集必须留出这段间隔，否则会偷看测试期
            gap = SWING_DAYS if key == "swing" else 1
            m = _fit(df[df.di <= start - gap - 1], y)
            if m is not None:
                pred.loc[test.index, f"p_{key}"] = m.predict_proba(test[FEATURES].astype(float))[:, 1]
        pred.loc[test.index, "oos"] = 1

    metrics = {k: _eval(pred[f"p_{k}"][pred.oos == 1], df[y][pred.oos == 1]) for k, y in TARGETS.items()}

    imp = {}
    for key, y in TARGETS.items():
        # 特征重要性：在最后 25% 的日期上做置换检验（模型用前 75% 训练）
        cut = dates[int(len(dates) * 0.75)]
        tr, te = df[(df.date < cut)], df[(df.date >= cut) & df[y].notna()]
        m2 = _fit(tr[tr.di <= pos[cut] - (SWING_DAYS if key == "swing" else 1) - 1], y)
        if m2 is not None and te[y].nunique() == 2:
            r = permutation_importance(m2, te[FEATURES].astype(float), te[y].astype(int),
                                       scoring="roc_auc", n_repeats=5, random_state=0)
            top = sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])[:10]
            imp[key] = [{"name": FEATURE_CN[f], "drop": float(v)} for f, v in top]

    pred.to_sql("pred", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    out = {"metrics": metrics, "importance": imp, "oos_from": dates[MIN_TRAIN_DAYS] if len(dates) > MIN_TRAIN_DAYS else None,
           "last": dates[-1], "n_days": len(dates)}
    (DATA / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, m in metrics.items():
        if m:
            log(f"[{k}] 样本外 n={m['n']} 基准率={m['base']:.1%} AUC={m['auc']:.3f} "
                f"Brier={m['brier']:.4f}(基准{m['brier_base']:.4f}) 前10%命中={m['top10_actual']:.1%}")
            log("    五分位 预测→实际: " + "  ".join(f"{b['pred']:.0%}→{b['actual']:.0%}" for b in m["buckets"]))
    return out
