#!/usr/bin/env python3
"""
实验一：发动机模拟试车实验 — 数据处理与绘图
=============================================
绘制发动机节流特性曲线和加力特性曲线。

用法：
    python exp1_trial_run.py          # 使用模拟数据运行
    python exp1_trial_run.py --real   # 读取 ../data/exp1_trial_run.csv 真实数据
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加 common 模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style
from dat_reader import read_lab_data, find_data_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_mock_data():
    """生成模拟的发动机试车数据，用于测试绘图管线。"""
    # 典型涡扇发动机参数范围
    states = [
        "慢车", "N2=75%", "N2=80%", "N2=85%", "N2=90%",
        "N2=95%", "N2=97%", "最大", "加力1", "加力2", "加力3",
    ]
    n_states = len(states)
    # 8 个非加力状态 + 3 个加力状态
    n_throttle = 8  # 前 8 个为节流状态

    np.random.seed(42)
    # 高压转子转速 (%)
    n2_pct = np.array([45, 75, 80, 85, 90, 95, 97, 100, 100, 100, 100])
    # 低压转子转速 (rpm) — 近似正比
    n1 = n2_pct * 45  # 最大约 4500 rpm
    n2 = n2_pct * 150  # 最大约 15000 rpm

    # 折合转速 (简化：用百分比近似)
    n1_cor = n2_pct.copy()
    n2_cor = n2_pct.copy()

    # 推力 (N) — 随转速指数增长
    F_base = 80000
    F = F_base * (n2_pct / 100) ** 3.5 + np.random.randn(n_states) * 500
    F[:n_throttle] = F[:n_throttle]  # 节流推力

    # 加力推力
    F_af = np.zeros(n_states)
    F_af[n_throttle:] = F_base * np.array([1.3, 1.5, 1.6]) + np.random.randn(3) * 1000

    # 燃油流量 Wf (kg/s)
    Wf = 0.05 + 2.5 * (n2_pct / 100) ** 3 + np.random.randn(n_states) * 0.02
    Wf_af = Wf.copy()
    Wf_af[n_throttle:] = Wf[n_throttle:] * np.array([1.8, 2.2, 2.5])

    # sfc = Wf / F (kg/(N·s))
    sfc = Wf / F
    sfc_af = np.zeros(n_throttle)
    sfc_af = np.append(sfc_af, Wf_af[n_throttle:] / F_af[n_throttle:])

    # 空气流量 ma (kg/s)
    ma = 20 + 120 * (n2_pct / 100) ** 2 + np.random.randn(n_states) * 2

    # 压气机总压比 π*c = P*3 / P*2 (P*2 ≈ 风扇出口)
    pi_c = 1.0 + 25 * (n2_pct / 100) ** 3 + np.random.randn(n_states) * 0.3
    pi_c = np.clip(pi_c, 1.0, None)

    # 导流叶片角度 α1, α2 (°)
    alpha1 = 30 - 25 * (n1_cor / 100) ** 1.5 + np.random.randn(n_states) * 0.5
    alpha2 = 35 - 30 * (n2_cor / 100) ** 1.5 + np.random.randn(n_states) * 0.5

    # 涡轮后总温 T*4 (K)
    T4_star = 600 + 900 * (n2_pct / 100) ** 2.5 + np.random.randn(n_states) * 10

    # 喷口面积 At (m²) — 加力时开大
    At = np.full(n_states, 0.35)
    At[n_throttle:] = np.array([0.42, 0.50, 0.55])

    # 油门杆位置 (°) — 加力状态
    throttle_pos = np.arange(len(states))

    df = pd.DataFrame({
        "工作状态": states,
        "n1": n1, "n2": n2,
        "n1_cor": n1_cor, "n2_cor": n2_cor,
        "n2_pct": n2_pct,
        "F": F, "F_af": F_af,
        "Wf": Wf, "Wf_af": Wf_af,
        "sfc": sfc, "sfc_af": sfc_af,
        "ma": ma,
        "pi_c": pi_c,
        "alpha1": alpha1, "alpha2": alpha2,
        "T4_star": T4_star,
        "At": At,
        "throttle_pos": throttle_pos,
    })
    return df, n_throttle


def load_real_data():
    """从 .dat 或 .csv 文件加载真实实验数据，返回原始 DataFrame。"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    filepath = find_data_file(data_dir, "exp1_trial_run")
    if filepath is None:
        print(f"[警告] 未在 {data_dir} 中找到 exp1_trial_run 的数据文件(.dat/.csv)，将使用模拟数据。")
        return None, None
    print(f"  📂 读取: {filepath}")
    df = read_lab_data(filepath)
    df = df.dropna(how="all")
    if df.empty:
        print("[警告] 数据文件为空，将使用模拟数据。")
        return None, None
    # 判断加力状态起始行
    first_col = df.columns[0]
    n_throttle = sum(~df[first_col].astype(str).str.contains("加力", na=False))
    return df, n_throttle


def normalize_columns(df):
    """将 CSV 中的列名映射到统一的内部列名。

    CSV 列名示例: "n1(rpm)" → dat_reader 处理 → "n1_rpm"
    内部列名: "n1"
    """
    mapping = {}
    for col in df.columns:
        # 去掉后缀的单位部分
        base = col
        for suffix in ["_rpm", "_Pa", "_K", "_N", "_kg_s", "_m2",
                       "_star_Pa", "_star_K", "_deg"]:
            if base.endswith(suffix) and base != suffix:
                base = base[:-len(suffix)]
                break
        mapping[col] = base

    # 手动修正已知列名
    known = {
        # 工作状态列
        "P0": "P0", "T0": "T0",
        # 转速
        "n1": "n1", "n2": "n2",
        # 推力
        "F": "F",
        # 燃油
        "Wf": "Wf",
        # 空气流量
        "ma": "ma",
        # 截面参数
        "P2": "P2_star", "T2": "T2_star",
        "P3": "P3_star", "T3": "T3_star",
        "P5": "P5_star", "T5": "T5_star",
        # 导流叶片
        "alpha1": "alpha1", "alpha2": "alpha2",
        # 喷口面积
        "At": "At",
    }
    for raw, std in known.items():
        for col in df.columns:
            if col.startswith(raw) and len(col) <= len(raw) + 8:
                mapping[col] = std

    # 应用映射
    new_cols = [mapping.get(c, c) for c in df.columns]
    df = df.copy()
    df.columns = new_cols

    # 同名列取第一个（处理可能的重复映射）
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def process_raw_data(df):
    """从原始测量数据计算所有派生参数。

    输入（CSV原始数据）:
        工作状态, n1, n2, P0, T0, F, Wf, ma,
        P2_star, T2_star, P3_star, T3_star, P5_star, T5_star,
        alpha1, alpha2, At

    输出（增加派生列）:
        n1_cor, n2_cor, sfc, pi_c, T4_star,
        F_af, Wf_af, sfc_af, throttle_pos
    """
    df = normalize_columns(df)

    # 折合转速：n_cor = n * sqrt(288.15 / T0)
    T0 = df["T0"].values
    df["n1_cor"] = df["n1"].values * np.sqrt(288.15 / T0)
    df["n2_cor"] = df["n2"].values * np.sqrt(288.15 / T0)

    # 耗油率：sfc = Wf / F (kg/(N·s))
    df["sfc"] = df["Wf"].values / df["F"].values

    # 压气机总压比：πc* = P3* / P2*
    df["pi_c"] = df["P3_star"].values / df["P2_star"].values

    # 涡轮后总温（用 P5*, T5* 作为 T4* 的近似）
    df["T4_star"] = df["T5_star"].values

    # 加力参数（与节流参数相同行，加力行有额外数据时覆盖）
    df["F_af"] = df["F"].values
    df["Wf_af"] = df["Wf"].values
    df["sfc_af"] = df["sfc"].values

    # 油门杆位置：简单序号
    df["throttle_pos"] = range(len(df))

    print("\n  派生参数计算完成:")
    print(f"    n_cor = n × √(288.15/T0)")
    print(f"    sfc   = Wf / F")
    print(f"    πc*   = P3* / P2*")
    print(f"    T4*   = T5* (涡轮后总温)")
    return df


def plot_throttle_characteristics(df, n_throttle):
    """绘制节流特性曲线（8 张子图）。"""
    throttle_df = df.iloc[:n_throttle]
    n1_cor = throttle_df["n1_cor"].values
    n2_cor = throttle_df["n2_cor"].values

    curves = [
        ("F", n1_cor, "推力 $F$ (N)", "$n_{1\\mathrm{cor}}$"),
        ("alpha1", n1_cor, "$\\alpha_1$ (°)", "$n_{1\\mathrm{cor}}$"),
        ("pi_c", n2_cor, "$\\pi_c^*$", "$n_{2\\mathrm{cor}}$"),
        ("alpha2", n2_cor, "$\\alpha_2$ (°)", "$n_{2\\mathrm{cor}}$"),
        ("F", n2_cor, "推力 $F$ (N)", "$n_{2\\mathrm{cor}}$"),
        ("sfc", n2_cor, "$\\mathrm{sfc}$ (kg/(N·s))", "$n_{2\\mathrm{cor}}$"),
        ("ma", n2_cor, "$m_a$ (kg/s)", "$n_{2\\mathrm{cor}}$"),
        ("T4_star", n2_cor, "$T_4^*$ (K)", "$n_{2\\mathrm{cor}}$"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()

    for ax, (col, x, ylabel, xlabel) in zip(axes, curves):
        y = throttle_df[col].values
        ax.plot(x, y, "o-", color="#1f77b4", markersize=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("发动机节流特性曲线", fontsize=14, y=1.01)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp1_throttle_characteristics"), ("png",))
    print(f"  ✓ 节流特性曲线已保存至 {OUTPUT_DIR}")


def plot_afterburner_characteristics(df, n_throttle):
    """绘制加力特性曲线（3 张子图）。"""
    af_df = df.iloc[n_throttle:]
    if len(af_df) == 0:
        print("  ⚠ 无加力状态数据，跳过加力特性曲线。")
        return

    pos = af_df["throttle_pos"].values
    curves = [
        ("F_af", "加力推力 $F_{af}$ (N)", "油门杆位置"),
        ("sfc_af", "加力耗油率 $\\mathrm{sfc}_{af}$ (kg/(N·s))", "油门杆位置"),
        ("At", "喷口面积 $A_e$ (m²)", "油门杆位置"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (col, ylabel, xlabel) in zip(axes, curves):
        y = af_df[col].values
        ax.plot(pos, y, "s-", color="#d62728", markersize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("发动机加力特性曲线", fontsize=14, y=1.02)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp1_afterburner_characteristics"), ("png",))
    print(f"  ✓ 加力特性曲线已保存至 {OUTPUT_DIR}")


def print_data_table(df):
    """打印数据表格摘要。"""
    print("\n" + "=" * 80)
    print("发动机试车数据摘要（前8行为节流状态，后3行为加力状态）")
    print("=" * 80)
    cols = ["工作状态", "n1_cor", "n2_cor", "F", "sfc", "ma", "pi_c", "T4_star"]
    print(df[cols].to_string(index=False))
    print("=" * 80)


def main():
    use_real = "--real" in sys.argv

    if use_real:
        print("📂 读取真实实验数据...")
        df_raw, n_throttle = load_real_data()
        if df_raw is None or n_throttle is None:
            print("回退到模拟数据。")
            df, n_throttle = generate_mock_data()
        else:
            df = process_raw_data(df_raw)
    else:
        print("🔧 使用模拟数据演示...")
        df, n_throttle = generate_mock_data()

    print_data_table(df)
    print("\n📊 绘制特性曲线...")
    plot_throttle_characteristics(df, n_throttle)
    plot_afterburner_characteristics(df, n_throttle)
    print("\n✅ 实验一数据处理完成。")
    print("   真实数据：请将数据填入 ../data/exp1_trial_run.csv 后运行 --real")


if __name__ == "__main__":
    main()
