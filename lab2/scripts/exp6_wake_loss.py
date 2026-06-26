#!/usr/bin/env python3
"""
实验六：平面叶栅尾迹损失实验 — 数据处理与绘图
===========================================
通过三孔探针扫描叶栅出口尾迹，绘制尾迹速度分布曲线。

速度计算（不可压缩伯努利方程）：
    v = sqrt(2 · (P2* - P_static) / ρ)
    总压亏损 → 速度亏损 → 尾迹损失

用法：
    python exp6_wake_loss.py
    python exp6_wake_loss.py --real
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


def calc_velocity(P_total, P_static, rho):
    """从总静压差计算速度。"""
    dp = np.maximum(P_total - P_static, 0)
    return np.sqrt(2 * dp / rho)


def generate_mock_data():
    """生成模拟的尾迹扫描数据。"""
    np.random.seed(42)
    P_atm = 101325
    T_atm = 293.15
    rho = P_atm / (R_AIR * T_atm)

    # 进口条件
    P_in_total = P_atm + 2000  # 进口总压
    P_in_static = P_atm - 500

    # 出口主流速度 ~ 60 m/s
    v_mainstream = np.sqrt(2 * 2000 / rho)

    # 扫描位置 X (cm)，覆盖约 2 个栅距 (32.5 mm × 2 = 6.5 cm)
    n_points = 40
    X = np.linspace(0, 8, n_points)

    # 总压亏损模拟：在叶片尾缘处产生亏损
    # 2 个叶片尾缘分别在 X ≈ 2 cm 和 X ≈ 5.25 cm 处
    deficit = np.ones(n_points)
    for wake_center in [2.0, 5.25]:
        wake_width = 0.6
        deficit -= 0.35 * np.exp(-((X - wake_center) / wake_width) ** 2)
        # 尾迹内附加小尺度湍流
        deficit += 0.02 * np.random.randn(n_points) * np.exp(-((X - wake_center) / wake_width) ** 2)

    P2_star = P_in_total * deficit + np.random.randn(n_points) * 10

    rows = []
    for i, x in enumerate(X):
        rows.append({
            "X_cm": round(x, 2),
            "P2_star": P2_star[i],
            "P_in_total": P_in_total,
            "P_in_static": P_in_static,
        })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp6_wake_loss")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def plot_wake_velocity(df):
    """绘制尾迹速度分布曲线。"""
    P_atm = 101325
    T_atm = 293.15
    rho = P_atm / (R_AIR * T_atm)

    # 计算速度
    P_in_total = df["P_in_total"].values[0]
    v = calc_velocity(df["P2_star"].values, P_in_total * 0.99, rho)
    X = df["X_cm"].values

    # 速度亏损系数
    v_max = np.max(v)
    v_ratio = v / v_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # 速度分布
    ax1.plot(X, v, "o-", color="#1f77b4", markersize=4)
    ax1.set_xlabel("探针位置 $X$ (cm)")
    ax1.set_ylabel("速度 $v$ (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("叶栅出口尾迹速度分布")

    # 标注尾迹位置
    for wake_x in [2.0, 5.25]:
        ax1.axvline(wake_x, color="#d62728", linestyle="--", alpha=0.5)
    ax1.annotate("叶片尾迹", xy=(2.0, v.min()), fontsize=9, color="#d62728",
                  xytext=(0.5, v.min() * 0.95), arrowprops=dict(arrowstyle="->", color="#d62728"))

    # 速度比分布
    ax2.plot(X, v_ratio, "s-", color="#2ca02c", markersize=4)
    ax2.set_xlabel("探针位置 $X$ (cm)")
    ax2.set_ylabel("$v / v_{\\max}$")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("无量纲速度亏损")
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp6_wake_velocity"), ("png",))
    print(f"  ✓ 尾迹速度分布曲线已保存至 {OUTPUT_DIR}")


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

    print(f"   扫描点数: {len(df)}, X范围: {df['X_cm'].min():.1f} ~ {df['X_cm'].max():.1f} cm")
    plot_wake_velocity(df)
    print("✅ 实验六（尾迹损失）数据处理完成。")


if __name__ == "__main__":
    main()
