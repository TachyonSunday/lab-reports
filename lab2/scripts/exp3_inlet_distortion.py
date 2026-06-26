#!/usr/bin/env python3
"""
实验三：进气畸变条件压气机性能实验 — 数据处理与绘图
=================================================
绘制均匀进气和畸变进气条件下的压升特性对比曲线，
分析畸变对压气机性能的影响。

畸变类型：无畸变（均匀）、30°扇形畸变、75°月牙形畸变、90°月牙形畸变

用法：
    python exp3_inlet_distortion.py
    python exp3_inlet_distortion.py --real
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
D_INLET = 0.502
A_INLET = np.pi * D_INLET**2 / 4


def calc_mass_flow(P_in_total, P_in_static, P_atm, T_atm):
    """质量流量计算（伯努利方程法）。"""
    rho = P_atm / (R_AIR * T_atm)
    dp = np.maximum(P_in_total - P_in_static, 0)
    v = np.sqrt(2 * dp / rho)
    return rho * A_INLET * v


def generate_mock_data():
    """生成模拟的均匀进气和畸变进气数据。"""
    np.random.seed(123)
    conditions = [
        ("uniform", "none"),
        ("distorted", "30deg_sector"),
        ("distorted", "75deg_crescent"),
    ]
    n_points = 8

    rows = []
    for cond, dtype in conditions:
        P_atm = 101325
        T_atm = 293.15
        rho = P_atm / (R_AIR * T_atm)

        dp_in = np.linspace(550, 50, n_points)
        P_in_total = P_atm - 20
        P_in_static = P_in_total - dp_in

        if dtype == "none":
            # 均匀进气 — 性能最好
            dp_factor = 1.0 + np.linspace(0, 0.25, n_points)
        elif "sector" in dtype:
            # 30°扇形畸变 — 性能下降约 5-10%
            dp_factor = 0.92 + np.linspace(0, 0.20, n_points)
        else:
            # 月牙形畸变 — 性能下降约 10-20%
            dp_factor = 0.82 + np.linspace(0, 0.15, n_points)

        P_out_total = P_atm + dp_in * dp_factor * 1.3

        for i in range(n_points):
            rows.append({
                "condition": cond,
                "distorter_type": dtype,
                "speed_rpm": 1200,
                "throttle_pos": int(170 + i * 30),
                "P_in_total": P_in_total + np.random.randn() * 5,
                "P_in_static": P_in_static[i] + np.random.randn() * 3,
                "P_out_total": P_out_total[i] + np.random.randn() * 10,
                "P_out_static": P_out_total[i] - dp_in[i] * 0.3,
                "P_atm": P_atm,
                "T_atm": T_atm,
            })

    return pd.DataFrame(rows)


def load_real_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp3_inlet_distortion")
    if filepath is None:
        return None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath).dropna(how="all")
    return df if not df.empty else None


def plot_distortion_comparison(df):
    """绘制均匀 vs 畸变特性对比曲线。"""
    fig, ax = plot_style.new_figure(9, 6)

    colors = {"uniform": "#1f77b4", "distorted": "#d62728"}
    markers = {
        "none": "o",
        "30deg_sector": "s",
        "75deg_crescent": "^",
        "90deg_crescent": "D",
    }
    labels = {
        "none": "均匀进气",
        "30deg_sector": "30°扇形畸变",
        "75deg_crescent": "75°月牙形畸变",
        "90deg_crescent": "90°月牙形畸变",
    }

    for (cond, dtype), sub in df.groupby(["condition", "distorter_type"]):
        sub = sub.sort_values("throttle_pos")
        m_dot = calc_mass_flow(
            sub["P_in_total"].values,
            sub["P_in_static"].values,
            sub["P_atm"].values,
            sub["T_atm"].values,
        )
        dp = sub["P_out_total"].values - sub["P_in_total"].values

        color = colors.get(cond, "#333333")
        marker = markers.get(dtype, "x")
        label = labels.get(dtype, dtype)
        ax.plot(m_dot, dp, f"{marker}-", color=color, label=label, markersize=7)

    ax.set_xlabel("质量流量 $\\dot{m}$ (kg/s)")
    ax.set_ylabel("总压升 $\\Delta P^*$ (Pa)")
    ax.set_title("进气畸变对压气机特性的影响 (1200 rpm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注失稳边界提前量
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp3_distortion_comparison"), ("png",))
    print(f"  ✓ 畸变对比曲线已保存至 {OUTPUT_DIR}")


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

    conditions = df.groupby(["condition", "distorter_type"]).size()
    print(f"   数据分组: {dict(conditions)}")
    plot_distortion_comparison(df)
    print("✅ 实验三（进气畸变）数据处理完成。")


if __name__ == "__main__":
    main()
