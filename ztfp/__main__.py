"""涨停复盘工具。

    python -m ztfp daily          # 收盘后跑：更新数据 -> 特征 -> 训练 -> 生成复盘页
    python -m ztfp update         # 只更新数据
    python -m ztfp build          # 只重建特征/标签
    python -m ztfp train          # 只重新训练并输出回测
    python -m ztfp report [日期]  # 生成某天的复盘页，默认最新一天
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
    if cmd in ("report", "daily"):
        from .report import make
        print("复盘页:", make(arg))
    if cmd not in ("update", "build", "train", "report", "daily"):
        print(__doc__)


if __name__ == "__main__":
    main()
