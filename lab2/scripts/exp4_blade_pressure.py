#!/usr/bin/env python3
"""
实验四：平面叶栅叶片表面压力分布测量 — 数据处理与绘图
===================================================
U 型管（水柱，mmH2O）数据 → Cp 分布 + 等熵马赫数分布。

Cp 直接用水柱比（分母分子 ρg 抵消，无需转 Pa）：
    C_p = (h_i - h_static) / (h_total - h_static)

等熵马赫数需要绝对压强（P_abs = P_atm + 9.81 * h(mmH2O)）：
    v_i = sqrt(2 · (P*_in - P_i) / ρ),  Ma_i = v_i / a

用法：
    python exp4_blade_pressure.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style
from dat_reader import read_lab_data, find_data_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

R_AIR = 287.058
GAMMA = 1.4

X_B = np.array([1, 2, 3, 4, 5, 6]) / 7


def calc_mach(P_gauge, P_in_total_gauge, P_atm, T_atm_C):
    """等熵马赫数：参数均为表压 Pa，P_atm 为大气压 Pa，T_atm_C 为摄氏度"""
    T_K = T_atm_C + 273.15
    rho = P_atm / (R_AIR * T_K)
    a = np.sqrt(GAMMA * R_AIR * T_K)
    P_i = P_atm + P_gauge
    P_total = P_atm + P_in_total_gauge
    dp = np.maximum(P_total - P_i, 0)
    v = np.sqrt(2 * dp / rho)
    return v / a


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp4_blade_pressure")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = pd.read_csv(filepath, comment="#", encoding="utf-8-sig").dropna(how="all")
    if df.empty:
        return None
    # 列名映射：CSV 列名 → 内部标准名
    mapping = {
        'P_in_total': 'P_in_total_gauge',
        'P_in_static': 'P_in_static_gauge',
        'P_static': 'P_static_gauge',
        'P_atm': 'P_atm',
        'T_atm': 'T_atm',
        'attack_angle': 'attack_angle',
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return df


def plot_pressure_distribution(df, group_label):
    """绘制三个攻角下的叶片表面静压分布（绝对压强 Pa）"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, alpha in zip(axes, sorted(df["attack_angle"].unique())):
        sub = df[df["attack_angle"] == alpha]
        P_atm = sub["P_atm"].values[0]
        for side, marker, label in [("suction", "o-", "吸力面"), ("pressure", "s-", "压力面")]:
            sd = sub[sub["side"] == side].sort_values("x_b")
            P_abs = P_atm + sd["P_static_gauge"].values
            ax.plot(sd["x_b"], P_abs, marker, label=label, markersize=6)
        ax.set_xlabel("$x/b$")
        ax.set_ylabel("$P$ (Pa)")
        ax.set_title(f"攻角 $i = {alpha}°$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"叶片表面静压分布（{group_label}）", fontsize=14)
    fig.tight_layout()
    fname = f"exp4_pressure_{group_label.replace(' ','_')}"
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, fname), ("png",))
    print(f"  ✓ 静压分布: {fname}.png")


def plot_mach_distribution(df, group_label):
    """绘制等熵马赫数分布"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, alpha in zip(axes, sorted(df["attack_angle"].unique())):
        sub = df[df["attack_angle"] == alpha]
        P_atm = sub["P_atm"].values[0]
        T_atm = sub["T_atm"].values[0]
        for side, marker, label in [("suction", "o-", "吸力面"), ("pressure", "s-", "压力面")]:
            sd = sub[sub["side"] == side].sort_values("x_b")
            Ma = calc_mach(sd["P_static_gauge"].values,
                           sd["P_in_total_gauge"].values[0],
                           P_atm, T_atm)
            ax.plot(sd["x_b"], Ma, marker, label=label, markersize=6)
        ax.set_xlabel("$x/b$")
        ax.set_ylabel("$Ma$")
        ax.set_title(f"攻角 $i = {alpha}°$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"叶片表面等熵马赫数分布（{group_label}）", fontsize=14)
    fig.tight_layout()
    fname = f"exp4_mach_{group_label.replace(' ','_')}"
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, fname), ("png",))
    print(f"  ✓ Ma 分布: {fname}.png")


def main():
    print("📂 读取实验数据...")
    df = load_real_data()
    if df is None:
        print("❌ 未找到数据文件，请将数据填入 data/exp4_blade_pressure.csv 后重试。")
        sys.exit(1)

    groups = df["group"].unique() if "group" in df.columns else ["default"]
    for g in groups:
        print(f"\n  === {g} ===")
        gdf = df[df["group"] == g] if "group" in df.columns else df
        attacks = sorted(gdf["attack_angle"].unique())
        print(f"  攻角: {attacks}")
        plot_pressure_distribution(gdf, str(g))
        plot_mach_distribution(gdf, str(g))
    print("\n✅ 实验四数据处理完成。")


if __name__ == "__main__":
    main()
