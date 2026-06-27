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
def load_real_data():
    """从 CSV 加载实验五数据"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp5_attack_angle")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = pd.read_csv(filepath, comment="#", encoding="utf-8-sig").dropna(how="all")
    if df.empty:
        return None
    df = df.dropna(subset=["P2_star"])
    return df if not df.empty else None


def process_data(df):
    """从原始数据计算转折角和损失系数（新格式：单值 P2* 和 βout）。"""
    results = []
    for _, row in df.iterrows():
        beta1 = 25.0 + row["beta_in"]              # β1 = 25° + 转盘刻度
        beta2 = row["beta_out"]                     # β2 = 探针出口气流角
        delta_beta = beta2 - beta1                  # Δβ
        omega = (row["P1_star"] - row["P2_star"]) / (row["P1_star"] - row["P1"])  # ω̄

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
    print("📂 读取实验数据...")
    df = load_real_data()
    if df is None:
        print("❌ 未找到有效数据，请将数据填入 data/exp5_attack_angle.csv 后重试。")
        sys.exit(1)

    print(f"   攻角范围: {sorted(df['attack_angle'].unique())}°")
    plot_attack_characteristics(df)
    print("✅ 实验五（攻角特性）数据处理完成。")


if __name__ == "__main__":
    main()
