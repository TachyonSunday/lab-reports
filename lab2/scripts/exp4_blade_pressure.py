#!/usr/bin/env python3
"""
实验四：平面叶栅叶片表面压力分布测量 — 数据处理与绘图
===================================================
绘制叶片表面静压分布曲线和等熵马赫数分布曲线。

静压系数：
    C_p = (P - P1) / (P1* - P1)

等熵马赫数（不可压缩流动，伯努利方程）：
    v_i = sqrt(2 · (P1* - P_i) / ρ)
    Ma_i = v_i / a
    其中 a = sqrt(γ · R · T)，γ=1.4, R=287.058 J/(kg·K)

用法：
    python exp4_blade_pressure.py
    python exp4_blade_pressure.py --real
"""

import sys
import os
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

# 叶片中弧线开孔位置（x/b）
X_B_POSITIONS = np.array([1, 2, 3, 4, 5, 6]) / 7


def calc_cp(P_static, P_in_total, P_in_static):
    """计算静压系数 C_p。"""
    return (P_static - P_in_static) / (P_in_total - P_in_static)


def calc_mach(P_static, P_in_total, rho, a):
    """计算等熵马赫数（不可压缩）。"""
    dp = np.maximum(P_in_total - P_static, 0)
    v = np.sqrt(2 * dp / rho)
    return v / a


def generate_mock_data():
    """生成模拟的叶片表面压力数据。"""
    np.random.seed(42)
    attack_angles = [-5, 0, 5]
    P_atm = 101325
    T_atm = 293.15
    rho = P_atm / (R_AIR * T_atm)
    a = np.sqrt(GAMMA * R_AIR * T_atm)

    rows = []
    for alpha in attack_angles:
        # 进口条件（总压随攻角略有变化）
        P_in_total = P_atm * (1.0 + 0.05 * np.cos(np.deg2rad(alpha + 35.6)))
        P_in_static = P_atm * 0.98

        for i, xb in enumerate(X_B_POSITIONS):
            # 吸力面：压力低，马赫数高
            # 压力面：压力高，马赫数低
            # 攻角为正时，前缘吸力面加速更强

            # 吸力面静压（低于进口静压）
            suction_peak = 0.25 + 0.05 * alpha / 5  # 峰值位置和强度
            suction_cp = -0.8 - 1.5 * np.exp(-((xb - suction_peak) / 0.2) ** 2)
            P_suction = P_in_static + suction_cp * (P_in_total - P_in_static)
            P_suction += np.random.randn() * 30

            # 压力面静压（接近或高于进口静压）
            pressure_cp = 0.5 - 0.3 * xb
            P_pressure = P_in_static + pressure_cp * (P_in_total - P_in_static)
            P_pressure += np.random.randn() * 30

            for side, P_s in [("suction", P_suction), ("pressure", P_pressure)]:
                rows.append({
                    "attack_angle": alpha,
                    "side": side,
                    "x_b": xb,
                    "P_in_total": P_in_total,
                    "P_static": P_s,
                    "P_atm": P_atm,
                    "T_atm": T_atm,
                })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp4_blade_pressure")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def plot_cp_distribution(df):
    """绘制三个攻角下的静压系数分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, alpha in zip(axes, sorted(df["attack_angle"].unique())):
        sub = df[df["attack_angle"] == alpha]
        for side, marker, label in [("suction", "o-", "吸力面(叶背)"), ("pressure", "s-", "压力面(叶盆)")]:
            side_data = sub[sub["side"] == side].sort_values("x_b")

            # 取第一行获取进口参数
            P_in_total = side_data["P_in_total"].values
            P_static = side_data["P_static"].values
            # 简化：用同一个进口条件
            cp = calc_cp(P_static, P_in_total[0], P_in_total[0] * 0.98)

            ax.plot(side_data["x_b"], cp, marker, label=label, markersize=6)

        ax.set_xlabel("$x/b$")
        ax.set_ylabel("$C_p$")
        ax.set_title(f"攻角 $i = {alpha}°$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 航空惯例：Cp 轴反向

    fig.suptitle("叶片表面静压系数分布", fontsize=14)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp4_cp_distribution"), ("png",))
    print(f"  ✓ Cp 分布曲线已保存")


def plot_mach_distribution(df):
    """绘制等熵马赫数分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, alpha in zip(axes, sorted(df["attack_angle"].unique())):
        sub = df[df["attack_angle"] == alpha]
        # 计算密度和声速
        P_atm = sub["P_atm"].values[0]
        T_atm = sub["T_atm"].values[0]
        rho = P_atm / (R_AIR * T_atm)
        a = np.sqrt(GAMMA * R_AIR * T_atm)

        for side, marker, label in [("suction", "o-", "吸力面"), ("pressure", "s-", "压力面")]:
            side_data = sub[sub["side"] == side].sort_values("x_b")
            P_in_total = side_data["P_in_total"].values
            P_static = side_data["P_static"].values
            Ma = calc_mach(P_static, P_in_total[0], rho, a)

            ax.plot(side_data["x_b"], Ma, marker, label=label, markersize=6)

        ax.set_xlabel("$x/b$")
        ax.set_ylabel("$Ma$")
        ax.set_title(f"攻角 $i = {alpha}°$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("叶片表面等熵马赫数分布", fontsize=14)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp4_mach_distribution"), ("png",))
    print(f"  ✓ 马赫数分布曲线已保存")


def main():
    use_real = "--real" in sys.argv

    if use_real:
        print("📂 读取真实数据...")
        df = load_real_data()
    else:
        df = None

    if df is None:
        if use_real:
            print("[警告] 无法读取真实数据，使用模拟数据。")
        else:
            print("🔧 使用模拟数据...")
        df = generate_mock_data()

    attacks = sorted(df["attack_angle"].unique())
    print(f"   攻角: {attacks}°")
    plot_cp_distribution(df)
    plot_mach_distribution(df)
    print("✅ 实验四（叶片表面压力分布）数据处理完成。")


if __name__ == "__main__":
    main()
