#!/usr/bin/env python3
"""
实验一：轴流压气机稳态特性试验 — 数据处理与绘图
=============================================
计算空气质量流量，绘制总压升-质量流量特性曲线（3条转速线）。

质量流量计算（不可压缩假设）：
    ρ = P_atm / (R · T_atm)                  # 空气密度
    v = sqrt(2 · (P_in_total - P_in_static) / ρ)   # 进口速度（伯努利）
    m_dot = ρ · A · v                        # 质量流量

总压升：
    ΔP = P_out_total - P_in_total

用法：
    python exp1_compressor_steady.py
    python exp1_compressor_steady.py --real
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

# 物理常数
R_AIR = 287.058  # J/(kg·K) 空气气体常数
D_INLET = 0.502  # m 进口截面直径
A_INLET = np.pi * D_INLET**2 / 4  # m²


def calculate_mass_flow(P_in_total, P_in_static, P_atm, T_atm):
    """
    计算通过压气机的质量流量。
    使用不可压缩伯努利方程从进口动静压差推算速度。
    """
    rho = P_atm / (R_AIR * T_atm)
    dp = np.maximum(P_in_total - P_in_static, 0)
    v = np.sqrt(2 * dp / rho)
    m_dot = rho * A_INLET * v
    return m_dot, rho


def generate_mock_data():
    """生成模拟的压气机稳态特性数据。"""
    np.random.seed(42)
    speeds = [1200, 1500, 1800]
    points_per_speed = 8

    rows = []
    for s in speeds:
        # 节流锥位置从 170 到 440 mm
        positions = np.linspace(170, 440, points_per_speed)
        P_atm = 101325
        T_atm = 293.15
        rho = P_atm / (R_AIR * T_atm)
        # 进口动压随流量增大
        dp_in = 50 + np.linspace(500, 50, points_per_speed) * (s / 1200) ** 2
        P_in_total = np.full(points_per_speed, P_atm - 20 * (s / 1200) ** 2)  # 进口总压略低于大气
        P_in_static = P_in_total - dp_in
        # 出口总压随转速和节流增大
        dp_out = dp_in * (1.0 + 0.8 * (s / 1200) ** 2 + np.linspace(0, 0.3, points_per_speed))
        dp_out += np.random.randn(points_per_speed) * 20
        P_out_total = np.full(points_per_speed, P_atm) + dp_out * 1.2
        P_out_static = P_out_total - dp_out * 0.3

        for i, pos in enumerate(positions):
            rows.append({
                "speed_rpm": s,
                "throttle_pos": pos,
                "P_in_total": P_in_total[i] + np.random.randn() * 5,
                "P_in_static": P_in_static[i] + np.random.randn() * 5,
                "P_out_total": P_out_total[i],
                "P_out_static": P_out_static[i],
                "P_atm": P_atm,
                "T_atm": T_atm,
            })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp1_compressor_steady")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def plot_characteristics(df):
    """绘制总压升-质量流量特性曲线。"""
    speeds = sorted(df["speed_rpm"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(speeds)))

    fig, ax = plot_style.new_figure(8, 5.5)

    for s, c in zip(speeds, colors):
        sub = df[df["speed_rpm"] == s].sort_values("throttle_pos")
        m_dot, _ = calculate_mass_flow(
            sub["P_in_total"].values,
            sub["P_in_static"].values,
            sub["P_atm"].values,
            sub["T_atm"].values,
        )
        dp = sub["P_out_total"].values - sub["P_in_total"].values

        ax.plot(m_dot, dp, "o-", color=c, label=f"{s} rpm", markersize=7)

        # 标注节流方向
        ax.annotate(
            "节流方向 →", (m_dot[-1], dp[-1]),
            textcoords="offset points", xytext=(-60, -15),
            fontsize=8, color=c,
            arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
        )

    ax.set_xlabel("质量流量 $\\dot{m}$ (kg/s)")
    ax.set_ylabel("总压升 $\\Delta P^*$ (Pa)")
    ax.set_title("对转压气机稳态特性曲线")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp1_compressor_steady_characteristics"), ("png",))
    print(f"  ✓ 特性曲线已保存至 {OUTPUT_DIR}")


def print_calculation_example(df):
    """打印一组质量流量计算示例。"""
    s = df["speed_rpm"].iloc[0]
    sub = df[df["speed_rpm"] == s].iloc[:1]
    m_dot, rho = calculate_mass_flow(
        sub["P_in_total"].values, sub["P_in_static"].values,
        sub["P_atm"].values, sub["T_atm"].values,
    )
    print(f"\n  质量流量计算示例 (转速={s} rpm, 第一个工况):")
    rho_val = float(rho[0]) if hasattr(rho, '__len__') else float(rho)
    print(f"    rho = P_atm / (R * T_atm) = {sub['P_atm'].values[0]:.0f} / (287.06 * {sub['T_atm'].values[0]:.1f}) = {rho_val:.4f} kg/m3")
    dp_val = float(sub['P_in_total'].values[0] - sub['P_in_static'].values[0])
    print(f"    DeltaP_in = P*_in - P_in = {sub['P_in_total'].values[0]:.0f} - {sub['P_in_static'].values[0]:.0f} = {dp_val:.0f} Pa")
    print(f"    v = sqrt(2*DeltaP_in / rho) = {np.sqrt(2 * max(dp_val, 0) / rho_val):.2f} m/s")
    print(f"    m_dot = rho * A * v = {rho_val:.4f} * {A_INLET:.4f} * ... = {m_dot[0]:.4f} kg/s")


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

    print(f"   数据点数: {len(df)}, 转速: {sorted(df['speed_rpm'].unique())} rpm")
    print_calculation_example(df)
    plot_characteristics(df)
    print("✅ 实验一（压气机稳态特性）数据处理完成。")


if __name__ == "__main__":
    main()
