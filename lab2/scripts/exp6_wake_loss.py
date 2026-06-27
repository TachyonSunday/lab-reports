#!/usr/bin/env python3
"""
实验六：平面叶栅尾迹损失实验
==========================
速度计算（不可压缩伯努利方程）：
    v = sqrt(2 · (P2* - P_in) / ρ)
    P2*_abs = P_atm + P2*_gauge, P_in_abs = P_atm + P_in_gauge

用法：python exp6_wake_loss.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

R_AIR = 287.058


def load_data():
    csv_path = os.path.join(DATA_DIR, "exp6_wake_loss.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 数据文件不存在: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, comment="#", encoding="utf-8-sig").dropna(how="all")
    # 列名映射
    rename = {}
    for c in df.columns:
        if '频率' in c or c.strip().endswith('Hz'): rename[c] = 'group'
        elif 'X' in c: rename[c] = 'X'
        elif 'P2' in c: rename[c] = 'P2_star'
        elif 'atm' in c.lower(): rename[c] = 'P_atm'
        elif 'in' in c.lower() and 'static' in c.lower(): rename[c] = 'P_in_static'
    df = df.rename(columns=rename)
    if 'group' not in df.columns: print("❌ 缺少分组列"); sys.exit(1)
    if 'P2_star' not in df.columns: print("❌ 缺少 P2_star 列"); sys.exit(1)
    df = df.dropna(subset=["P2_star"])
    # 频率值 → 组标签
    df['group'] = df['group'].apply(lambda x: f"{float(x):.0f} Hz" if 'Hz' not in str(x) and x != '' else str(x))
    df = df.sort_values(["group", "X"])
    if df.empty: print("❌ 无有效数据"); sys.exit(1)
    return df


def process(df):
    """计算速度和速度比"""
    results = {}
    has_P_in = 'P_in_static' in df.columns and not df['P_in_static'].isna().all()
    T_atm = 293.15
    for group, gdf in df.groupby("group"):
        P_atm = gdf["P_atm"].values[0]
        rho = P_atm / (R_AIR * T_atm)
        P2_abs = P_atm + gdf["P2_star"].values

        if has_P_in:
            P_in_abs = P_atm + gdf["P_in_static"].values[0]
            dp = np.maximum(P2_abs - P_in_abs, 0)
            v = np.sqrt(2 * dp / rho)
        else:
            v = np.zeros(len(P2_abs))

        results[group] = {
            'X': gdf["X"].values,
            'P2_star': gdf["P2_star"].values,
            'P2_abs': P2_abs,
            'v': v,
            'v_ratio': v / np.max(v) if np.max(v) > 0 else np.ones(len(v)),
            'P_atm': P_atm,
            'has_v': has_P_in,
        }
        info = f"{len(gdf)} 点"
        if has_P_in: info += f", v_max={np.max(v):.1f} m/s"
        print(f"  {group}: {info}")
    return results


def generate_tables(df):
    """生成三线表"""
    has_P_in = 'P_in_static' in df.columns and not df['P_in_static'].isna().all()
    for group, gdf in df.groupby("group"):
        P_atm = int(gdf["P_atm"].values[0])
        P_in_str = f", $P_{{\\text{{in}}}}={int(gdf['P_in_static'].values[0])}$ Pa" if has_P_in else ""
        lines = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"  \centering")
        short = f"{group} 尾迹扫描数据"
        long = f"{group}，$P_{{\\text{{atm}}}}={P_atm}$ Pa{P_in_str}"
        lines.append(r"  \caption[" + short + "]{" + long + "}")
        lines.append(r"  \label{tab:exp6_" + group.replace(' ', '_') + "}")
        lines.append(r"  \renewcommand{\arraystretch}{1.2}")
        lines.append(r"  \begin{tabular}{cc}")
        lines.append(r"    \toprule")
        lines.append(r"    $X$ (cm) & $P_2^*$ (Pa) \\")
        lines.append(r"    \midrule")
        for _, r in gdf.sort_values("X").iterrows():
            lines.append(f'    {r["X"]:.2f} & {r["P2_star"]:.0f} \\\\')
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        fname = f"table_exp6_{group.replace(' ','_')}.tex"
        with open(os.path.join(TABLE_DIR, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  ✓ 表格: {fname}")


def plot_wake(results):
    """绘制尾迹分布"""
    has_v = any(r.get('has_v', False) for r in results.values())
    if has_v:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
    colors = {}
    for i, label in enumerate(results.keys()):
        colors[label] = plt.cm.tab10(i)
    for label, r in results.items():
        c = colors[label]
        if has_v:
            ax1.plot(r['X'], r['v'], 'o-', color=c, label=label, markersize=4)
            ax2.plot(r['X'], r['v_ratio'], 's-', color=c, label=label, markersize=4)
        else:
            ax1.plot(r['X'], r['P2_abs'], 'o-', color=c, label=label, markersize=4)
    ax1.set_xlabel("$X$ (cm)")
    ax1.set_ylabel("$v$ (m/s)" if has_v else "$P_2^*$ (Pa)")
    ax1.set_title("尾迹总压分布")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    if has_v:
        ax2.set_xlabel("$X$ (cm)"); ax2.set_ylabel("$v/v_{\\max}$"); ax2.set_title("无量纲速度亏损")
        ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(FIG_DIR, "exp6_wake_velocity"), ("png",))
    print("  ✓ 图表: exp6_wake_velocity.png")


def main():
    print("📂 读取实验数据...")
    df = load_data()
    results = process(df)
    generate_tables(df)
    plot_wake(results)
    print("✅ 实验六数据处理完成。")


if __name__ == "__main__":
    main()
