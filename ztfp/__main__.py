"""涨停复盘工具。

    python -m ztfp daily          # 收盘后跑：更新数据 -> 特征 -> 训练 -> 生成合并复盘页 reports/index.html
    python -m ztfp page           # 只重新生成合并页（明日关注 / 涨停复盘 / 预测复盘 / 趋势突破）
    python -m ztfp update         # 只更新数据
    python -m ztfp build          # 只重建特征/标签
    python -m ztfp train          # 只重新训练并输出回测
    python -m ztfp report [日期]  # 生成某天的复盘页，默认最新一天
    python -m ztfp review         # 涨停预测复盘：过去给的连板概率事后准不准
    python -m ztfp trend          # 趋势突破复盘：更新全市场日K -> 扫描 -> 回测 -> 生成页面
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("update", "daily"):
        from .fetch import update
        update()
    if cmd in ("build", "daily"):
        from .features import build
        build()
    if cmd in ("train", "daily"):
        from .model import train
        train()
    if cmd == "report":
        from .report import make
        print("复盘页:", make(arg))
    if cmd == "review":
        from .review import make as review
        print("预测复盘页:", review())
    if cmd in ("trend", "daily"):
        from . import trend, trend_data
        trend_data.update()
        trend.scan()
        trend.train()
        if cmd == "trend":
            print("趋势复盘页:", trend.report())
    if cmd in ("page", "daily"):
        from .page import make as page
        print("合并复盘页:", page())
    if cmd not in ("update", "build", "train", "report", "daily", "trend", "review", "page"):
        print(__doc__)


if __name__ == "__main__":
    main()
