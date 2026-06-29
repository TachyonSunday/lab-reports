#!/usr/bin/env python3
"""
实验一：发动机模拟试车实验 — 数据处理与绘图
数据来源: data/实验1数据.xlsx
n1/n2 已是换算转速，F 已是换算推力，不做额外折合处理。
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

# Excel 物理列序
COLS = ['alpha0','n2','n1','P0','T0','F','Wf','ma',
        'P2_star','T2_star','P3_star','T3_star','P5_star','alpha1','alpha2']


def load_data():
    raw = pd.read_excel(DATA, sheet_name='Sheet1', header=None)
    data_rows = []
    for i in range(1, len(raw)):
        floats = []
        for x in raw.iloc[i]:
            try: floats.append(float(x))
            except: pass
        # 去除前导 NaN（空白单元格）
        while floats and math.isnan(floats[0]):
            floats.pop(0)
        data_rows.append(floats)

    df = pd.DataFrame([{c: r[i] for i, c in enumerate(COLS)} for r in data_rows])

    # 状态标签（LaTeX 格式）
    labels = []
    for i in range(len(df)):
        n2 = df.iloc[i]['n2']; a0 = df.iloc[i]['alpha0']
        if i == 0:
            labels.append(f"\\makecell{{慢车状态 \\\\ (N2={n2:.2f}\\%)}}")
        elif i < 7:
            labels.append(f"N2={n2:.2f}\\%")
        elif i == 7:
            labels.append(f"\\makecell{{最大状态 \\\\ (N2={n2:.2f}\\%)}}")
        else:
            labels.append(f"\\makecell{{加力状态 \\\\ ($a_0={a0:.2f}$)}}")
    df['LABEL'] = labels
    return df


def process(df):
    """计算派生参数"""
    df['F_N'] = df['F'] * 9.81
    df['sfc'] = df['Wf'] / df['F']
    df['pi_c'] = df['P3_star'] / df['P2_star']
    return df


def plot_throttle(df):
    """节流特性曲线（7张，前8行）"""
    thr = df.iloc[:8]
    n1 = thr['n1'].values
    n2 = thr['n2'].values

    curves = [
        ('F_N', n1, 'F (N)', '$n_{1\\mathrm{cor}}$'),
        ('alpha1', n1, '$\\alpha_1$ (°)', '$n_{1\\mathrm{cor}}$'),
        ('pi_c', n2, '$\\pi_c^*$', '$n_{2\\mathrm{cor}}$'),
        ('alpha2', n2, '$\\alpha_2$ (°)', '$n_{2\\mathrm{cor}}$'),
        ('F_N', n2, 'F (N)', '$n_{2\\mathrm{cor}}$'),
        ('sfc', n2, 'sfc (h$^{-1}$)', '$n_{2\\mathrm{cor}}$'),
        ('ma', n2, '$m_a$ (kg/s)', '$n_{2\\mathrm{cor}}$'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (col, x, yl, xl) in zip(axes, curves):
        ax.plot(x, thr[col].values, 'o-', markersize=6)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    axes[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp1_throttle_characteristics.png"), dpi=200)
    plt.close()
    print("  ✓ 节流特性曲线")


def plot_afterburner(df):
    """加力特性曲线（后5行）"""
    af = df.iloc[8:]
    x = af['alpha0'].values
    curves = [
        ('F_N', '加力推力 F (N)'),
        ('sfc', '加力耗油率 sfc (h$^{-1}$)'),
        ('ma', '加力空气流量 $m_a$ (kg/s)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, yl) in zip(axes, curves):
        ax.plot(x, af[col].values, 's-', color='#d62728', markersize=8)
        ax.set_xlabel("$a_0$"); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp1_afterburner_characteristics.png"), dpi=200)
    plt.close()
    print("  ✓ 加力特性曲线")


def generate_table(df):
    """生成三线表——原始测量参数，格式与用户表格一致"""
    table_cols = ['n1','P0','T0','F','Wf','ma','P2_star','T2_star',
                  'P3_star','T3_star','P5_star','alpha1','alpha2']
    L = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{发动机模拟试车实验数据}",
        r"  \label{tab:exp1_trial}",
        r"",
        r"\makebox[\textwidth][l]{",
        r"  试车员\underline{\makebox[3cm][c]{\studentName}} ",
        r"  \hfill ",
        r"  记录员\underline{\makebox[3cm][c]{\studentName}} ",
        r"  \hfill ",
        r"  \expDate",
        r"}",
        r"  \vspace{0.1cm}",
        r"",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \makebox[\textwidth][c]{",
        r"    \begin{tabular}{|c|*{13}{c|}}",
        r"      \hline",
        r"      \diagbox[width=3.6cm, height=1.3cm]{工作状态}{\makecell{参数名称}} & "
        r"$n_1$ & $P_0$ & $T_0$ & $F$ & $W_f$ & $m_a$ & $P_2^*$ & $T_2^*$ & "
        r"$P_3^*$ & $T_3^*$ & $P_5^*$ & $\alpha_1$ & $\alpha_2$ \\",
        r"      \hline",
    ]
    for _, r in df.iterrows():
        vals = " & ".join([f"{r[c]:.2f}" for c in table_cols])
        L.append(f"      {r['LABEL']} & {vals} \\\\")
        L.append(r"      \hline")

    L += [
        r"    \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
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
        print(f"  {r['LABEL']:>22}: n1={r['n1']:.1f} n2={r['n2']:.1f} "
              f"F={r['F_N']:.0f}N sfc={r['sfc']:.4f} πc={r['pi_c']:.2f}")

    print("\n📊 绘制图表...")
    plot_throttle(df)
    plot_afterburner(df)

    print("\n📋 生成表格...")
    generate_table(df)

    print("\n✅ 实验一数据处理完成。")


if __name__ == "__main__":
    main()
