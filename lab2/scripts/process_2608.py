#!/usr/bin/env python3
"""
处理压气机稳态特性实验数据（实验一 & 实验三）
实验一：无畸变，2608.dat / 2608（2）.dat
实验三：加畸变片，2608xin-J.dat / 2608（2）xin-J.dat
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
A_INLET = np.pi * D_INLET ** 2 / 4


def parse_gbk_dat(filepath):
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
            if '温度' in stripped:  # 温度
                parts = stripped.split()
                for i, p in enumerate(parts):
                    if '温度' in p and i + 1 < len(parts):
                        meta['T_atm_C'] = float(re.sub(r'[：:℃°]', '', parts[i + 1]))
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
    T_atm = meta['T_atm_C'] + 273.15
    P_atm = meta['P_atm']
    rho = P_atm / (R_AIR * T_atm)
    df['DeltaP'] = df['Pte'] - df['Pti']
    df['PtRatio'] = df['Pte'] / df['Pti']
    dp_in = np.maximum(df['Pti'] - df['PiN'], 0)
    v_in = np.sqrt(2 * dp_in / rho)
    df['m_dot_calc'] = rho * A_INLET * v_in
    df['m_dot_err'] = np.abs(df['m_dot_calc'] - df['m_dot_raw']) / df['m_dot_raw'] * 100
    return df, T_atm, P_atm, rho


def make_threeline_table(df, caption, label, columns):
    col_spec = "c" * len(columns)
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
    header = " & ".join(columns)
    lines.append(f"      {header} \\\\")
    lines.append(r"      \midrule")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
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
        lines.append(f"      {' & '.join(vals)} \\\\")
    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_group_table(df, group_label, condition_text, meta, extra_cols=True):
    if extra_cols:
        table_cols = ['throttle_deg', 'Pti', 'PiN', 'Pte', 'Pe', 'DeltaP',
                      'm_dot_raw', 'm_dot_calc', 'm_dot_err']
        col_labels = ['节流阀角度°', '进口总压 (Pa)', '进口静压 (Pa)', '出口总压 (Pa)',
                      '出口静压 (Pa)', '总压升 (Pa)', '流量实测 (kg/s)',
                      '流量计算 (kg/s)', '相对误差 (\\%)']
    else:
        table_cols = ['throttle_deg', 'Pti', 'PiN', 'Pte', 'Pe', 'DeltaP', 'm_dot_raw']
        col_labels = ['节流阀角度°', '进口总压 (Pa)', '进口静压 (Pa)', '出口总压 (Pa)',
                      '出口静压 (Pa)', '总压升 (Pa)', '质量流量 (kg/s)']

    caption = f"{group_label} 稳态特性数据——{condition_text} (T={meta['T_atm_C']}°C, P={meta['P_atm']}Pa)"
    label = f"tab:{group_label.lower().replace(' ','_')}"

    display_df = df[table_cols].copy()
    display_df.columns = col_labels
    return make_threeline_table(display_df, caption, label, col_labels)


def main():
    exp1_files = {
        "组1": ("2608.dat", "前转子1800rpm, 后转子2100rpm"),
        "组2": ("2608（2）.dat", "前转子1680rpm, 后转子2220rpm"),
    }
    exp3_files = {
        "组1": ("2608xin-J.dat", "前转子1800rpm, 后转子2100rpm（加畸变片）"),
        "组2": ("2608（2）xin-J.dat", "前转子1680rpm, 后转子2220rpm（加畸变片）"),
    }

    import matplotlib.pyplot as plt

    def do_dataset(label, fname, condition):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[警告] 文件不存在: {fpath}")
            return None
        print(f"\n  {label}: {fname} — {condition}")
        df_raw, meta = parse_gbk_dat(fpath)
        df, T_atm, P_atm, rho = process(df_raw, meta)
        print(f"    T={T_atm:.2f}K, P={P_atm:.0f}Pa, ΔPmax={df['DeltaP'].max():.0f}Pa")
        return df, meta, condition

    def save_tables_and_plot(table_prefix, fig_name, fig_title, extra_cols):
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = {'组1': 'o', '组2': 's'}
        for label, (df, meta, condition) in processed.items():
            table_tex = generate_group_table(df, label, condition, meta, extra_cols=extra_cols)
            with open(os.path.join(TABLE_DIR, f"table_{table_prefix}_{label}.tex"), "w", encoding="utf-8") as f:
                f.write(table_tex)
            print(f"    ✓ 表格: table_{table_prefix}_{label}.tex")
            m = markers.get(label, 'x')
            ax.plot(df['m_dot_raw'], df['DeltaP'], f"{m}-", label=label, markersize=7)
        ax.set_xlabel("质量流量 $\\dot{m}$ (kg/s)")
        ax.set_ylabel("总压升 $\\Delta P$ (Pa)")
        ax.set_title(fig_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_style.save_figure(fig, os.path.join(FIG_DIR, fig_name), ("png",))
        print(f"    ✓ 图表: {fig_name}.png")

    # ── 实验一：无畸变 ───────────────────────────
    print(f"\n{'='*70}\n  实验一：无畸变对转压气机稳态特性\n{'='*70}")
    processed = {}
    for label, (fname, condition) in exp1_files.items():
        r = do_dataset(label, fname, condition)
        if r: processed[label] = r
    save_tables_and_plot("2608", "exp1_2608_characteristics",
                         "对转压气机稳态特性曲线（实验一，无畸变）", extra_cols=True)

    # ── 实验三：加畸变片 ─────────────────────────
    print(f"\n{'='*70}\n  实验三：加畸变片对转压气机稳态特性\n{'='*70}")
    processed = {}
    for label, (fname, condition) in exp3_files.items():
        r = do_dataset(label, fname, condition)
        if r: processed[label] = r
    save_tables_and_plot("exp3", "exp3_characteristics",
                         "加畸变片对转压气机稳态特性曲线（实验三）", extra_cols=False)

    print("\n✅ 处理完成。")


if __name__ == "__main__":
    main()
