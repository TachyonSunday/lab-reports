#!/usr/bin/env python3
"""
实验一：发动机模拟试车实验 — 数据处理与绘图
数据来源: data/实验1数据.xlsx
表中 n1/n2 已是换算转速，F 已是换算推力，不做额外折合处理。
sfc = Wf / F, πc* = P3* / P2*
"""

import pandas as pd, numpy as np, os, math, sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "实验1数据.xlsx")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def load_data():
    raw = pd.read_excel(DATA, sheet_name='Sheet1', header=None)
    data_rows = []
    for i in range(1, len(raw)):
        floats = []
        for x in raw.iloc[i]:
            try: floats.append(float(x))
            except: pass
        while floats and math.isnan(floats[0]):
            floats.pop(0)
        data_rows.append(floats)

    cols = ['n1','n2','P0','T0','F','Wf','ma','P2_star','T2_star',
            'P3_star','T3_star','P5_star','alpha1','alpha2']
    df = pd.DataFrame([{c: r[i] for i,c in enumerate(cols)} for r in data_rows])
    df['state'] = ['慢车','N2=75%','N2=80%','N2=85%','N2=90%',
                   'N2=95%','N2=97%','最大','加力_a=0','加力_a=1',
                   '加力_a=2','加力_a=3','加力_a=4']
    return df


def process(df):
    """计算派生参数（n1/n2/F 已是换算值，不做折合）"""
    df['F_N'] = df['F'] * 9.81          # kg → N
    df['sfc'] = df['Wf'] / df['F']      # kg/(h·kg)
    df['pi_c'] = df['P3_star'] / df['P2_star']
    return df


def plot_throttle(df):
    """节流特性曲线（7张）"""
    thr = df.iloc[:8]  # 前8行为节流状态
    n1 = thr['n1'].values
    n2 = thr['n2'].values

    curves = [
        ('F_N', n1, 'F (N)', '$n_1$'),
        ('alpha1', n1, '$\\alpha_1$ (°)', '$n_1$'),
        ('pi_c', n2, '$\\pi_c^*$', '$n_2$'),
        ('alpha2', n2, '$\\alpha_2$ (°)', '$n_2$'),
        ('F_N', n2, 'F (N)', '$n_2$'),
        ('sfc', n2, 'sfc (h$^{-1}$)', '$n_2$'),
        ('ma', n2, '$m_a$ (kg/s)', '$n_2$'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (col, x, yl, xl) in zip(axes, curves):
        ax.plot(x, thr[col].values, 'o-', markersize=6)
        ax.set_xlabel(xl, fontsize=13); ax.set_ylabel(yl, fontsize=13); ax.grid(True, alpha=0.3)
    # 隐藏第8个空子图
    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp1_throttle_characteristics.png"), dpi=200)
    plt.close()
    print("  ✓ 节流特性曲线")


def plot_afterburner(df):
    """加力特性曲线"""
    af = df.iloc[8:]
    x = range(len(af))
    curves = [
        ('F_N', '加力推力 F (N)'),
        ('sfc', '加力耗油率 sfc (h$^{-1}$)'),
        ('ma', '加力空气流量 $m_a$ (kg/s)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, yl) in zip(axes, curves):
        ax.plot(x, af[col].values, 's-', color='#d62728', markersize=8)
        ax.set_xlabel("油门杆位置", fontsize=13); ax.set_ylabel(yl, fontsize=13); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp1_afterburner_characteristics.png"), dpi=200)
    plt.close()
    print("  ✓ 加力特性曲线")


def generate_table(df):
    """生成三线表"""
    L = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{发动机模拟试车实验数据}",
        r"  \label{tab:exp1_trial}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{cccccccccc}",
        r"    \toprule",
        r"    工作状态 & $n_1$ & $n_2$ & $F$(N) & $W_f$(kg/h) & $m_a$(kg/s) & $\pi_c^*$ & sfc(h$^{-1}$) & $\alpha_1$ & $\alpha_2$ \\",
        r"    \midrule",
    ]
    for _, r in df.iterrows():
        L.append(f"    {r['state']} & {r['n1']:.1f} & {r['n2']:.1f} & {r['F_N']:.0f} & {r['Wf']:.0f} & {r['ma']:.1f} & {r['pi_c']:.2f} & {r['sfc']:.4f} & {r['alpha1']:.0f} & {r['alpha2']:.0f} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path = os.path.join(TABLE_DIR, "table_exp1_trial.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  ✓ 表格: {path}")


def main():
    print("📂 读取实验数据...")
    df = load_data()
    df = process(df)

    print("\n数据摘要:")
    for _, r in df.iterrows():
        print(f"  {r['state']:>8}: n1={r['n1']:.1f} n2={r['n2']:.1f} F={r['F_N']:.0f}N Wf={r['Wf']:.0f} ma={r['ma']:.1f} πc={r['pi_c']:.2f} sfc={r['sfc']:.4f}")

    print("\n📊 绘制图表...")
    plot_throttle(df)
    plot_afterburner(df)

    print("\n📋 生成表格...")
    generate_table(df)

    print("\n✅ 实验一数据处理完成。")


if __name__ == "__main__":
    main()
