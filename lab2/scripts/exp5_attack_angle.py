#!/usr/bin/env python3
"""
实验五：平面叶栅攻角特性测量 — 数据处理与绘图
============================================
计算气流转折角和叶栅损失系数，绘制攻角特性曲线。

气流转折角：
    Δβ = β2 - β1
    其中 β1 = 进口几何角（由转盘刻度换算），β2 = 探针测得出口气流角

损失系数：
    ω = (P1* - P2*) / (P1* - P1)

几何关系：
    进口气流角 β1 = 25° + turntable_angle
    攻角 i = β1 - 35.6°（35.6°为进口几何角）

用法：
    python exp5_attack_angle.py
    python exp5_attack_angle.py --real
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

# NACA-65 叶栅几何参数
BETA1_GEOM = 35.6  # 进口几何角 (°)
BETA2_GEOM = 85.6  # 出口几何角 (°)


def calc_turning_angle(probe_angles):
    """计算气流转折角 Δβ = β2 - β1。"""
    beta2 = np.mean(probe_angles)
    return beta2


def calc_loss_coefficient(P1_star, P2_star_list, P1):
    """计算总压损失系数 ω = (P1* - P2*) / (P1* - P1)。"""
    P2_star_avg = np.mean(P2_star_list)
    return (P1_star - P2_star_avg) / (P1_star - P1)


def generate_mock_data():
    """生成模拟的攻角特性数据（基于典型 NACA-65 叶栅特性）。"""
    np.random.seed(123)
    attack_angles = np.array([-15, -10, -5, 0, 5, 7])

    # 典型的攻角特性
    # 转折角随攻角增大而增大（近似线性）
    delta_beta = 50.0 + 0.8 * attack_angles + np.random.randn(len(attack_angles)) * 1.5
    # 损失系数在设计攻角（0°）附近最小，偏离设计点增大
    omega = 0.03 + 0.001 * (attack_angles - 0) ** 2 + np.abs(attack_angles) * 0.002
    omega += np.random.randn(len(attack_angles)) * 0.003

    # 进口条件
    P1_star = np.full(len(attack_angles), 102000)
    P1 = np.full(len(attack_angles), 100000)

    rows = []
    for i, alpha in enumerate(attack_angles):
        # 出口 4 个测点
        P2_star_vals = P1_star[i] - omega[i] * (P1_star[i] - P1[i]) + np.random.randn(4) * 5
        # 出口气流角
        beta2_base = BETA1_GEOM + delta_beta[i]
        beta2_vals = beta2_base + np.random.randn(4) * 0.5
        turntable_angle = alpha + BETA1_GEOM - 25

        rows.append({
            "attack_angle": alpha,
            "turntable_angle": turntable_angle,
            "P1_star": P1_star[i],
            "P1_static": P1[i],
            "P2_star_1": P2_star_vals[0],
            "P2_star_2": P2_star_vals[1],
            "P2_star_3": P2_star_vals[2],
            "P2_star_4": P2_star_vals[3],
            "P2_static_1": P2_star_vals[0] * 0.99,
            "P2_static_2": P2_star_vals[1] * 0.99,
            "P2_static_3": P2_star_vals[2] * 0.99,
            "P2_static_4": P2_star_vals[3] * 0.99,
            "probe_angle_1": beta2_vals[0],
            "probe_angle_2": beta2_vals[1],
            "probe_angle_3": beta2_vals[2],
            "probe_angle_4": beta2_vals[3],
        })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp5_attack_angle")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def process_data(df):
    """从原始数据计算转折角和损失系数。"""
    results = []
    for _, row in df.iterrows():
        P2_star_list = [row[f"P2_star_{i}"] for i in range(1, 5)]
        probe_angles = [row[f"probe_angle_{i}"] for i in range(1, 5)]

        beta2 = calc_turning_angle(probe_angles)
        # β1 = 25° + turntable_angle
        beta1 = 25 + row["turntable_angle"]
        delta_beta = beta2 - beta1
        omega = calc_loss_coefficient(row["P1_star"], P2_star_list, row["P1_static"])

        results.append({
            "attack_angle": row["attack_angle"],
            "beta1": beta1,
            "beta2": beta2,
            "delta_beta": delta_beta,
            "omega": omega,
        })

    return pd.DataFrame(results)


def plot_attack_characteristics(df):
    """绘制攻角特性曲线 (Δβ 和 ω)。"""
    results = process_data(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 气流转折角
    ax1.plot(results["attack_angle"], results["delta_beta"], "o-", color="#1f77b4", markersize=8)
    ax1.set_xlabel("攻角 $i$ (°)")
    ax1.set_ylabel("气流转折角 $\\Delta\\beta$ (°)")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("叶栅攻角特性 — 气流转折角")

    # 损失系数
    ax2.plot(results["attack_angle"], results["omega"], "s-", color="#d62728", markersize=8)
    ax2.set_xlabel("攻角 $i$ (°)")
    ax2.set_ylabel("总压损失系数 $\\omega$")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("叶栅攻角特性 — 总压损失系数")

    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp5_attack_angle_characteristics"), ("png",))
    print(f"  ✓ 攻角特性曲线已保存至 {OUTPUT_DIR}")

    # 打印数据表
    print("\n  计算结果:")
    print(results.to_string(index=False))


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

    print(f"   攻角范围: {sorted(df['attack_angle'].unique())}°")
    plot_attack_characteristics(df)
    print("✅ 实验五（攻角特性）数据处理完成。")


if __name__ == "__main__":
    main()
