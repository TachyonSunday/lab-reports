#!/usr/bin/env python3
"""
处理动态压力测量数据（实验二：叶轮机非定常压力参数测量）
=======================================================
数据文件命名: 08-{前机频率}-{后机频率}-{节流角}.txt

数据格式:
  第1行: 采样时间
  第2行: 采样频率
  第3行: 测点描述
  第4行: 列标题 (time[s], AI1-1[Pa], AI1-2[Pa], AI1-3[Pa])
  后续: 制表符分隔的数据

探针布置: 下游转子进口，沿周向布置 3 个动态压力传感器
  - 相邻两片叶片前缘各 1 个
  - 两叶片前缘连线中点 1 个

处理内容:
  1. 时域波形（前 0.05s）
  2. FFT 频谱（0-1500Hz，标注 BPF）
  3. 统计量（均值、RMS、峰峰值）
  4. 四组工况对比
"""

import sys, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# 实验参数
N1_BLADES = 12  # 上游转子叶片数
N2_BLADES = 10  # 下游转子叶片数


def parse_dynamic_file(filepath):
    """解析动态压力数据文件，返回 (metadata, data_array)"""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    meta = {}
    # 行1: 采样时间
    m = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', lines[0])
    meta['time'] = m.group(1) if m else lines[0].strip()

    # 行2: 采样频率
    m = re.search(r'(\d+)', lines[1])
    meta['fs'] = int(m.group(1)) if m else 5120

    # 行3: 测点描述
    meta['probe_desc'] = lines[2].strip()

    # 行4: 列标题
    headers = lines[3].strip().split('\t')
    meta['headers'] = [h.strip() for h in headers if h.strip()]

    # 数据行（从第5行开始，跳过空行）
    data_rows = []
    for line in lines[4:]:
        stripped = line.strip()
        if not stripped:
            continue
        vals = stripped.split('\t')
        if len(vals) >= 4:
            try:
                data_rows.append([float(v) for v in vals[:4]])
            except ValueError:
                continue

    data = np.array(data_rows)
    return meta, data


def compute_statistics(data, fs):
    """计算各通道统计量"""
    stats = {}
    for i in range(1, data.shape[1]):  # 跳过时间列
        ch = data[:, i]
        stats[f'ch{i}'] = {
            'mean': np.mean(ch),
            'rms': np.sqrt(np.mean(ch**2)),
            'std': np.std(ch),
            'peak_to_peak': np.max(ch) - np.min(ch),
            'max': np.max(ch),
            'min': np.min(ch),
        }
    return stats


def compute_fft(data, fs, ch_idx=1):
    """计算单通道 FFT，返回 (freq, magnitude)"""
    sig = data[:, ch_idx] - np.mean(data[:, ch_idx])  # 去均值
    n = len(sig)
    yf = fft(sig)
    xf = fftfreq(n, 1 / fs)
    mask = xf >= 0
    mag = 2.0 / n * np.abs(yf[mask])
    return xf[mask], mag


def make_threeline_table(rows, caption, label, col_labels):
    """生成三线表 LaTeX"""
    col_spec = "c" * len(col_labels)
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
    header = " & ".join(col_labels)
    lines.append(f"      {header} \\\\")
    lines.append(r"      \midrule")
    for row in rows:
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        line = " & ".join(vals)
        lines.append(f"      {line} \\\\")
    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    # 扫描数据文件
    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        m = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)\.txt', fname)
        if m:
            files.append({
                'fname': fname,
                'group': m.group(1),
                'f_front': int(m.group(2)),
                'f_rear': int(m.group(3)),
                'throttle': int(m.group(4)),
                'path': os.path.join(DATA_DIR, fname),
            })

    if not files:
        print("未找到匹配的动态数据文件 (格式: XX-FF-RR-TT.txt)")
        return

    print(f"找到 {len(files)} 个数据文件:" )
    for f in files:
        print(f"  {f['fname']}: 组{f['group']}, 前{f['f_front']}Hz 后{f['f_rear']}Hz, 节流角{f['throttle']}°")

    # ── 处理每个文件 ─────────────────────────────
    all_results = {}
    for f_info in files:
        label = f"前{f_info['f_front']}Hz 后{f_info['f_rear']}Hz {f_info['throttle']}°"
        print(f"\n{'='*70}")
        print(f"  {f_info['fname']} — {label}")
        print(f"{'='*70}")

        meta, data = parse_dynamic_file(f_info['path'])
        fs = meta['fs']
        n_samples = len(data)
        duration = n_samples / fs

        print(f"  采样频率: {fs} Hz, 数据点数: {n_samples}, 时长: {duration:.2f}s")
        print(f"  测点: {meta['headers']}")

        # 计算 BPF
        bpf_front = f_info['f_front'] * N1_BLADES  # freq * blades = BPF
        bpf_rear = f_info['f_rear'] * N2_BLADES
        print(f"  BPF (上游R1, {N1_BLADES}片): {bpf_front} Hz")
        print(f"  BPF (下游R2, {N2_BLADES}片): {bpf_rear} Hz")

        # 统计量
        stats = compute_statistics(data, fs)
        print(f"\n  统计量:")
        print(f"  {'通道':>6} {'均值(Pa)':>10} {'RMS(Pa)':>10} {'峰峰值(Pa)':>10}")
        print(f"  {'-'*40}")
        for ch, s in stats.items():
            print(f"  {ch:>6} {s['mean']:>10.2f} {s['rms']:>10.2f} {s['peak_to_peak']:>10.2f}")

        all_results[label] = {
            'info': f_info,
            'meta': meta,
            'data': data,
            'stats': stats,
            'bpf_front': bpf_front,
            'bpf_rear': bpf_rear,
        }

    # ── 绘制时域图（前 0.05s，四组对比）─────────
    print(f"\n{'='*70}")
    print(f"  绘制图表")
    print(f"{'='*70}")

    fig, axes = plt.subplots(len(files), 1, figsize=(12, 2.5 * len(files)))
    if len(files) == 1:
        axes = [axes]

    colors_probe = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ax, (label, result) in zip(axes, all_results.items()):
        data = result['data']
        t = data[:, 0]
        mask = t <= 0.03  # 前 30ms
        t_plot = t[mask] * 1000  # 转为 ms
        for ch in [1, 2, 3]:
            ax.plot(t_plot, data[mask, ch], linewidth=0.5,
                    color=colors_probe[ch - 1], label=f'探针{ch}')
        ax.set_ylabel("压力 (Pa)")
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=10, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间 (ms)")
    fig.suptitle("动态压力时域信号（前 30 ms）", fontsize=16)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(FIG_DIR, "exp2_dynamic_time_domain"), ("png",))
    print("  ✓ 时域图已保存")

    # ── 频域图：每工况一个子图，包含3个探针 ────
    fig, axes = plt.subplots(len(files), 1, figsize=(12, 2.8 * len(files)))
    if len(files) == 1:
        axes = [axes]

    colors_probe = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ax, (label, result) in zip(axes, all_results.items()):
        data = result['data']
        fs = result['meta']['fs']
        bpf_f = result['bpf_front']
        bpf_r = result['bpf_rear']

        # 三个探针的 FFT
        max_mag = 0
        for ch in [1, 2, 3]:
            xf, mag = compute_fft(data, fs, ch_idx=ch)
            mask = xf <= 1500
            ax.plot(xf[mask], mag[mask], linewidth=0.5,
                    color=colors_probe[ch - 1], label=f'探针{ch}', alpha=0.85)
            max_mag = max(max_mag, mag[mask].max())

        # 标注 BPF（错开高度防重叠）
        bpf_list = [
            (bpf_f, f'BPF R1={bpf_f}Hz', '#d62728', 0.85),
            (bpf_r, f'BPF R2={bpf_r}Hz', '#9467bd', 0.70),
        ]
        for bpf, bpf_label, color, y_pos in bpf_list:
            for h in [1, 2, 3]:
                f_h = bpf * h
                if f_h <= 1500:
                    ax.axvline(f_h, color=color, linestyle='--', alpha=0.35, linewidth=0.6)
            ax.annotate(bpf_label, xy=(bpf, max_mag * y_pos),
                       fontsize=7, color=color, fontweight='bold')

        ax.set_ylabel("幅值")
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=10, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("频率 (Hz)")
    fig.suptitle("动态压力频域图（FFT 幅值谱，0-1500 Hz）", fontsize=16)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(FIG_DIR, "exp2_dynamic_frequency_domain"), ("png",))
    print("  ✓ 频域图已保存")

    # ── 统计量表格 ──────────────────────────────
    rows = []
    for label, result in all_results.items():
        info = result['info']
        stats = result['stats']
        for ch_name, s in stats.items():
            rows.append([label, ch_name, s['mean'], s['rms'], s['peak_to_peak']])

    table_tex = make_threeline_table(
        rows,
        "动态压力测量统计量——四组工况对比",
        "tab:dynamic_stats",
        ['工况', '通道', '均值 (Pa)', 'RMS (Pa)', '峰峰值 (Pa)']
    )
    with open(os.path.join(TABLE_DIR, "table_dynamic_stats.tex"), "w", encoding="utf-8") as f:
        f.write(table_tex)
    print("  ✓ 统计量表已保存")

    # ── BPF 汇总表 ──────────────────────────────
    rows = []
    for label, result in all_results.items():
        info = result['info']
        bpf_f = result['bpf_front']
        bpf_r = result['bpf_rear']
        # 找 BPF 附近的实际峰值
        xf, mag = compute_fft(result['data'], result['meta']['fs'], ch_idx=1)
        for bpf, rotor in [(bpf_f, 'R1(上游)'), (bpf_r, 'R2(下游)')]:
            idx = np.argmin(np.abs(xf - bpf))
            actual_f = xf[idx]
            actual_mag = mag[idx]
            # 在 ±5Hz 范围内找实际峰值
            nearby = (xf >= bpf - 5) & (xf <= bpf + 5)
            if nearby.any():
                peak_idx = np.argmax(mag[nearby])
                actual_f = xf[nearby][peak_idx]
                actual_mag = mag[nearby][peak_idx]
            rows.append([label, rotor, bpf, actual_f, actual_mag])

    table_tex = make_threeline_table(
        rows,
        "叶片通过频率 (BPF) 分析",
        "tab:bpf_analysis",
        ['工况', '转子', '理论BPF (Hz)', '实测峰值频率 (Hz)', '峰值幅值']
    )
    with open(os.path.join(TABLE_DIR, "table_bpf_analysis.tex"), "w", encoding="utf-8") as f:
        f.write(table_tex)
    print("  ✓ BPF 分析表已保存")

    print("\n✅ 动态数据处理完成。")


if __name__ == "__main__":
    main()
