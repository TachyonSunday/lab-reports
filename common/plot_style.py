"""
共享绘图样式配置 — 用于所有实验的数据处理脚本。
提供统一的 matplotlib 样式，确保所有图表风格一致，适合学术报告。
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import glob

# ── 字体与数学渲染 ────────────────────────────────────────────
# 尝试使用系统中可用的中文字体
_CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans SC", "Noto Serif CJK SC",
    "SimSun", "Songti SC", "STSong", "AR PL UMing CN",
    "WenQuanYi Micro Hei", "Microsoft YaHei",
    "SimHei", "Heiti SC", "STHeiti",
    "DejaVu Sans",
]

def _detect_font():
    """检测系统中第一个可用的中文字体"""
    import matplotlib.font_manager as fm
    import glob
    # 尝试注册用户目录下的字体文件
    font_dirs = [
        os.path.expanduser("~/.fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in font_dirs:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "**", "*.otf"), recursive=True):
                try:
                    fm.fontManager.addfont(f)
                except Exception:
                    pass
            for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
                try:
                    fm.fontManager.addfont(f)
                except Exception:
                    pass

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CHINESE_FONT_CANDIDATES:
        if name in available:
            return name
    return matplotlib.rcParams["font.sans-serif"][0]


_SERIF_FONT = _detect_font()

# ── 全局样式 ──────────────────────────────────────────────────
def apply_style():
    """应用统一的学术报告绘图样式"""
    plt.rcParams.update({
        # 字体
        "font.family": "sans-serif",
        "font.sans-serif": [_SERIF_FONT, "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,

        # 图表尺寸（适合 A4 报告）
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 150,

        # 线条与标记
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.8,

        # 坐标轴
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # 图例
        "legend.framealpha": 0.8,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,

        # 保存
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.format": "png",
    })


def new_figure(width=6.5, height=4.0, nrows=1, ncols=1):
    """创建应用了样式的 figure 对象"""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    return fig, axes


def save_figure(fig, path, formats=("png",)):
    """保存图表到指定路径"""
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", format=fmt)
    plt.close(fig)


# ── 常用标注工具 ──────────────────────────────────────────────
def annotate_point(ax, x, y, text, offset=(5, 5)):
    """在数据点上标注文字"""
    ax.annotate(
        text, (x, y),
        textcoords="offset points",
        xytext=offset,
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
    )


def set_axis_labels(ax, xlabel, ylabel, title=None):
    """统一设置坐标轴标签"""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


# ── 自动应用样式 ──────────────────────────────────────────────
apply_style()
