#!/usr/bin/env python3
"""
处理实验数据文件 2608.dat / 2608（2）.dat
========================================
两组对转压气机稳态特性数据，处理并生成三线表 LaTeX 代码。

实验条件:
  组1 (2608.dat):      前转子 30Hz×60=1800rpm, 后转子 35Hz×60=2100rpm
  组2 (2608（2）.dat): 前转子 28Hz×60=1680rpm, 后转子 37Hz×60=2220rpm

输出:
  - 终端打印数据摘要
  - 生成 ../reports/tables/table_2608_group1.tex (组1 三线表)
  - 生成 ../reports/tables/table_2608_group2.tex (组2 三线表)
  - 生成 ../reports/tables/table_2608_combined.tex (合并对比表)
  - 绘制特性曲线 ../reports/figures/exp1_2608_characteristics.png

用法:
  python process_2608.py
"""

import sys, os, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

R_AIR = 287.058
D_INLET = 0.502
A_INLET = np.pi * D_INLET**2 / 4
GAMMA = 1.4


def parse_gbk_dat(filepath):
    """解析 GBK 编码的实验室 .dat 文件"""
    with open(filepath, "r", encoding="gbk") as f:
        lines = f.readlines()

    meta = {}
    data_lines = []
    header = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any('一' <= c <= '鿿' for c in stripped):
            if '温度' in stripped:
                parts = stripped.split()
                for i, p in enumerate(parts):
                    if '温度' in p and i + 1 < len(parts):
                        meta['T_atm_C'] = float(re.sub(r'[：:℃°℃]', '', parts[i + 1]))
                    if '压力' in p and i + 1 < len(parts):
                        meta['P_atm'] = float(re.sub(r'[：:Pa]', '', parts[i + 1]))
            if '序号' in stripped:
                header = [c.replace(' ', '') for c in re.split(r'\s+', stripped)]
        elif header is not None:
            vals = re.split(r'\s+', stripped)
            if len(vals) >= 8:
                data_lines.append([float(v) for v in vals])

    short_names = {
        '序号': 'idx', '前机频率': 'f_front', '后机频率': 'f_rear',
        '节流角度': 'throttle_deg',
        '进口总压Pti': 'Pti', '进口静压PiN': 'PiN',
        '出口总压Pte': 'Pte', '出口静压Pe': 'Pe',
        '间隙1Pta1': 'Pta1_1', 'Pta2': 'Pta1_2', 'Pta3': 'Pta1_3',
        '间隙2Ptb1': 'Ptb2_1', 'Ptb2': 'Ptb2_2', 'Ptb3': 'Ptb2_3',
        '进口流量Mass0': 'm_dot_raw',
        '效率ef': 'eff', '总压比PtRatio': 'PtRatio_raw',
        '前机功率': 'Pw_front', '后机功率': 'Pw_rear',
        '间隙1总压Pta': 'Pta1', '静压Pa': 'Pa1',
        '间隙1相对来流角Betaa': 'beta1',
        '间隙2总压Ptb': 'Ptb2', '静压Pb': 'Pb2',
        '间隙2相对来流角Betab': 'beta2',
    }
    header = [short_names.get(h, h) for h in header]
    df = pd.DataFrame(data_lines, columns=header[:len(data_lines[0])])
    return df, meta


def process(df, meta):
    """计算派生参数"""
    T_atm = meta['T_atm_C'] + 273.15
    P_atm = meta['P_atm']
    rho = P_atm / (R_AIR * T_atm)

    df['DeltaP'] = df['Pte'] - df['Pti']
    df['PtRatio'] = df['Pte'] / df['Pti']

    dp_in = np.maximum(df['Pti'] - df['PiN'], 0)
    v_in = np.sqrt(2 * dp_in / rho)
    df['m_dot_calc'] = rho * A_INLET * v_in

    # 计算流量与实测流量的相对误差
    df['m_dot_err'] = np.abs(df['m_dot_calc'] - df['m_dot_raw']) / df['m_dot_raw'] * 100

    # 进口动压
    df['q_in'] = df['Pti'] - df['PiN']

    # 流量系数 φ = v_axial / U (U = πDN/60)
    # 这里用上游转子
    # df['phi'] = v_in / U_front

    return df, T_atm, P_atm, rho


def make_threeline_table(df, caption, label, columns):
    """生成 LaTeX 三线表代码"""
    col_spec = "c" * (len(columns))
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + "}")
    lines.append(r"  \label{" + label + "}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \resizebox{\textwidth}{!}{")
    lines.append(r"    \begin{tabular}{" + col_spec + "}")
    lines.append(r"      \toprule")

    # 表头
    header = " & ".join(columns)
    lines.append(f"    {header} \\\\")
    lines.append(r"      \midrule")

    # 数据行
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                # 总压比保留4位小数（接近1时需要精度）
                if 'PtRatio' in col or '压比' in col:
                    vals.append(f"{v:.4f}")
                elif 'ṁ' in col or 'm_dot' in col.lower() or '流量' in col:
                    vals.append(f"{v:.3f}")
                elif 'P' in col and ('Pa' in col or abs(v) > 90000):
                    vals.append(f"{v:.0f}")
                elif abs(v) < 0.01:
                    vals.append(f"{v:.4f}")
                elif abs(v) < 1:
                    vals.append(f"{v:.3f}")
                elif abs(v) < 100:
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.0f}")
            else:
                vals.append(str(v))
        line = " & ".join(vals)
        lines.append(f"      {line} \\\\")

    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_group_table(df, group_label, condition_text, meta):
    """为单组数据生成详细三线表（7列）"""
    table_cols = ['throttle_deg', 'Pti', 'PiN', 'Pte', 'Pe', 'DeltaP', 'm_dot_raw', 'm_dot_calc', 'm_dot_err']
    col_labels = ['节流阀角度°', '进口总压 (Pa)', '进口静压 (Pa)', '出口总压 (Pa)',
                  '出口静压 (Pa)', '总压升 (Pa)', '流量实测 (kg/s)', '流量计算 (kg/s)', '相对误差 (\%)']

    caption = f"{group_label} 稳态特性数据——{condition_text} (T={meta['T_atm_C']}°C, P={meta['P_atm']}Pa)"
    label = f"tab:{group_label.lower().replace(' ','_')}"

    display_df = df[table_cols].copy()
    display_df.columns = col_labels

    return make_threeline_table(display_df, caption, label, col_labels)


def main():
    files = {
        "组1": ("2608.dat", "前转子1800rpm, 后转子2100rpm"),
        "组2": ("2608（2）.dat", "前转子1680rpm, 后转子2220rpm"),
    }

    all_results = {}

    for label, (fname, condition) in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[警告] 文件不存在: {fpath}")
            continue

        print(f"\n{'='*70}")
        print(f"  {label}: {fname}")
        print(f"  实验条件: {condition}")
        print(f"{'='*70}")

        df_raw, meta = parse_gbk_dat(fpath)
        df, T_atm, P_atm, rho = process(df_raw, meta)

        # 打印摘要
        print(f"\n  T_atm={T_atm:.2f}K, P_atm={P_atm:.0f}Pa, ρ={rho:.4f}kg/m³")
        print(f"  {'节流角':>6} {'ΔP':>7} {'PtRatio':>8} {'ṁ_raw':>7} {'ṁ_calc':>7} "
              f"{'Pta1_avg':>8} {'Ptb2_avg':>8}")
        print(f"  {'-'*65}")
        for _, r in df.iterrows():
            print(f"  {r['throttle_deg']:>6.0f} {r['DeltaP']:>7.0f} {r['PtRatio']:>8.4f} "
                  f"{r['m_dot_raw']:>7.3f} {r['m_dot_calc']:>7.3f} "
                  f"{r['Pta1']:>8.0f} {r['Ptb2']:>8.0f}")

        # 生成三线表
        table_tex = generate_group_table(df, label, condition, meta)
        table_file = os.path.join(TABLE_DIR, f"table_2608_{label}.tex")
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(table_tex)
        print(f"\n  ✓ 三线表已保存: {table_file}")

        # 保存处理后的数据
        df_out = df.copy()
        df_out['group'] = label
        df_out['condition'] = condition
        df_out['T_atm_K'] = T_atm
        df_out['P_atm_Pa'] = P_atm
        all_results[label] = df_out

    # ── 生成合并对比表 ─────────────────────────────
    if len(all_results) == 2:
        print(f"\n{'='*70}")
        print(f"  生成合并对比表")
        print(f"{'='*70}")

        combined_rows = []
        for label, df in all_results.items():
            for _, r in df.iterrows():
                combined_rows.append({
                    '组别': label,
                    '节流角°': r['throttle_deg'],
                    'ΔP (Pa)': r['DeltaP'],
                    '总压比': r['PtRatio'],
                    'ṁ (kg/s)': r['m_dot_raw'],
                    'Pta1 (Pa)': r['Pta1'],
                    'Ptb2 (Pa)': r['Ptb2'],
                })

        combined_df = pd.DataFrame(combined_rows)
        table_tex = make_threeline_table(
            combined_df,
            "两组实验稳态特性数据对比——不同转速组合下的压气机性能",
            "tab:combined_2608",
            ['组别', '节流角°', 'ΔP (Pa)', '总压比', 'ṁ (kg/s)', 'Pta1 (Pa)', 'Ptb2 (Pa)']
        )
        table_file = os.path.join(TABLE_DIR, "table_2608_combined.tex")
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(table_tex)
        print(f"  ✓ 合并表已保存: {table_file}")

    # ── 绘制特性曲线 ─────────────────────────────
    print(f"\n{'='*70}")
    print(f"  绘制特性曲线")
    print(f"{'='*70}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = {'组1': 'o', '组2': 's'}
    for label, df in all_results.items():
        m = markers.get(label, 'x')
        ax.plot(df['m_dot_raw'], df['DeltaP'], f"{m}-", label=label, markersize=7)

    ax.set_xlabel("质量流量 $\\dot{m}$ (kg/s)")
    ax.set_ylabel("总压升 $\\Delta P$ (Pa)")
    ax.set_title("对转压气机稳态特性曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(FIG_DIR, "exp1_2608_characteristics"), ("png",))

    print("\n✅ 处理完成。")


if __name__ == "__main__":
    main()
