#!/usr/bin/env python3
"""
实验七：三孔气动探针标定校准实验 — 数据处理与绘图
=================================================
计算方向标定系数 Kα、总压标定系数 Kt、速度标定系数 Kv，
并绘制标定曲线。

标定系数定义（非对向校准法）：
    Kα = (P1 - P3) / (P2 - P_avg)   方向系数
    Kt = (Pt - P2) / (P2 - P_avg)   总压系数
    Kv = (Pt - Ps) / (P2 - P_avg)   速度系数

    其中 P_avg = (P1 + P3) / 2

用法：
    python exp7_probe_calibration.py
    python exp7_probe_calibration.py --real
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


def calc_calibration_coefficients(P1, P2, P3, Pt, Ps):
    """
    计算三孔探针标定系数。

    参数:
        P1, P2, P3: 三孔探针三个孔的压力
        Pt: 风洞总压
        Ps: 风洞静压

    返回:
        Kα, Kt, Kv
    """
    P_avg = (P1 + P3) / 2.0
    denom = P2 - P_avg

    # 避免除零
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)

    K_alpha = (P1 - P3) / denom
    K_t = (Pt - P2) / denom
    K_v = (Pt - Ps) / denom

    return K_alpha, K_t, K_v


def generate_mock_data():
    """生成模拟的标定风洞数据。"""
    np.random.seed(42)

    # 标定偏航角范围
    alphas = np.arange(-25, 30, 5)

    # 风洞来流条件
    Pt = 105000   # 风洞总压 (Pa)
    Ps = 100000   # 风洞静压 (Pa) — 来流 Ma ≈ 0.27
    P0 = 101325   # 大气压
    T0 = 293.15

    # 模拟三孔探针在各迎角下的响应
    # Kα 与 α 近似线性（在 ±25° 范围内）
    K_alpha_ideal = 0.04 * alphas
    K_t_ideal = 1.0 + 0.0002 * alphas**2
    K_v_ideal = 1.2 + 0.0003 * alphas**2

    rows = []
    for i, alpha in enumerate(alphas):
        # 反推各孔压力
        P_avg = Ps * 0.8
        P2 = P_avg + (Pt - Ps) / K_v_ideal[i]
        P1_minus_P3 = K_alpha_ideal[i] * (P2 - P_avg)
        P1 = P_avg + P1_minus_P3 / 2 + np.random.randn() * 2
        P3 = P_avg - P1_minus_P3 / 2 + np.random.randn() * 2

        rows.append({
            "alpha": alpha,
            "P1": P1,
            "P2": P2 + np.random.randn() * 3,
            "P3": P3,
            "Pt": Pt,
            "Ps": Ps,
            "P0": P0,
            "T0": T0,
        })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp7_probe_calibration")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def plot_calibration_curves(df):
    """绘制 Kα, Kt, Kv 标定曲线。"""
    alphas = df["alpha"].values
    K_alpha, K_t, K_v = calc_calibration_coefficients(
        df["P1"].values, df["P2"].values, df["P3"].values,
        df["Pt"].values, df["Ps"].values,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Kα — 方向系数
    axes[0].plot(alphas, K_alpha, "o-", color="#1f77b4", markersize=8)
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("迎角 $\\alpha$ (°)")
    axes[0].set_ylabel("$K_\\alpha$")
    axes[0].set_title("方向标定系数")
    axes[0].grid(True, alpha=0.3)

    # Kt — 总压系数
    axes[1].plot(alphas, K_t, "s-", color="#d62728", markersize=8)
    axes[1].set_xlabel("迎角 $\\alpha$ (°)")
    axes[1].set_ylabel("$K_t$")
    axes[1].set_title("总压标定系数")
    axes[1].grid(True, alpha=0.3)

    # Kv — 速度系数
    axes[2].plot(alphas, K_v, "^-", color="#2ca02c", markersize=8)
    axes[2].set_xlabel("迎角 $\\alpha$ (°)")
    axes[2].set_ylabel("$K_v$")
    axes[2].set_title("速度标定系数")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("三孔气动探针标定曲线", fontsize=14)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp7_calibration_curves"), ("png",))
    print(f"  ✓ 标定曲线已保存至 {OUTPUT_DIR}")

    # 打印数据表
    print("\n  标定系数计算结果:")
    print(f"  {'α(°)':>6}  {'Kα':>10}  {'Kt':>10}  {'Kv':>10}")
    print(f"  {'-'*42}")
    for a, ka, kt, kv in zip(alphas, K_alpha, K_t, K_v):
        print(f"  {a:>6}  {ka:>10.4f}  {kt:>10.4f}  {kv:>10.4f}")


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

    print(f"   迎角范围: {df['alpha'].min():.0f}° ~ {df['alpha'].max():.0f}°")
    plot_calibration_curves(df)
    print("✅ 实验七（三孔探针标定）数据处理完成。")


if __name__ == "__main__":
    main()
