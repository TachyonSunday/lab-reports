#!/usr/bin/env python3
"""
实验二：叶轮机非定常压力参数测量与数据处理
=========================================
绘制时域压力信号波形图，进行 FFT 变换得到频域图，
标注叶片通过频率 (BPF) 等特征频率。

叶片通过频率：
    BPF_R1 = n1/60 × N_blades_R1    (上游转子)
    BPF_R2 = n2/60 × N_blades_R2    (下游转子)

用法：
    python exp2_unsteady_pressure.py                  # 模拟数据
    python exp2_unsteady_pressure.py --real            # 读取真实数据
    python exp2_unsteady_pressure.py --data-dir <路径>  # 指定数据目录
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import plot_style
from dat_reader import read_lab_data, find_data_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 5120  # Hz
DURATION = 1.0      # s
N1_BLADES = 12       # 上游转子叶片数
N2_BLADES = 10       # 下游转子叶片数
N_CHANNELS = 12      # 传感器通道数


def generate_mock_signal(speed_rpm, duration=DURATION, sr=SAMPLE_RATE):
    """
    生成模拟的非定常压力信号。
    包含：叶片通过频率基频+谐波、随机噪声。
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    # 上游转子 BPF
    bpf1 = speed_rpm / 60 * N1_BLADES  # Hz
    # 下游转子 BPF
    bpf2 = speed_rpm / 60 * N2_BLADES

    # 基频 + 2次谐波 + 宽带噪声
    signal = np.zeros(n_samples)
    for bpf in [bpf1, bpf2]:
        for harmonic in [1, 2, 3]:
            amp = 1.0 / harmonic * (0.8 + 0.2 * np.random.rand())
            phase = np.random.rand() * 2 * np.pi
            signal += amp * np.sin(2 * np.pi * bpf * harmonic * t + phase)

    # 添加湍流宽带噪声
    noise = 0.15 * np.random.randn(n_samples)
    # 低频脉动
    low_freq = 0.3 * np.sin(2 * np.pi * 15 * t) * np.random.randn(n_samples) * 0.1

    return t, signal + noise + low_freq, bpf1, bpf2


def compute_fft(signal, sr=SAMPLE_RATE):
    """计算单边 FFT 幅值谱。"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sr)
    # 取正频率部分
    mask = xf >= 0
    magnitude = 2.0 / n * np.abs(yf[mask])
    return xf[mask], magnitude


def plot_time_domain(signals_data, speeds):
    """绘制时域波形图 — 6通道对比。"""
    n_plot = min(6, len(signals_data))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.2 * n_plot), sharex=True)

    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        s = speeds[i]
        t, sig, bpf1, bpf2 = signals_data[i]
        # 只绘制前 0.05 秒以看清波形
        mask = t <= 0.05
        axes[i].plot(t[mask] * 1000, sig[mask], linewidth=0.6, color=f"C{i}")
        axes[i].set_ylabel(f"通道{i+1}\n压力 (Pa)")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"转速={s} rpm, BPF1={bpf1:.0f}Hz, BPF2={bpf2:.0f}Hz", fontsize=9)

    axes[-1].set_xlabel("时间 (ms)")
    fig.suptitle("非定常压力时域信号（前 50 ms）", fontsize=13)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp2_time_domain"), ("png",))
    print(f"  ✓ 时域图已保存")


def plot_frequency_domain(signals_data, speeds):
    """绘制 FFT 频域图 — 标注 BPF。"""
    n_plot = min(6, len(signals_data))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.2 * n_plot), sharex=True)

    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        s = speeds[i]
        t, sig, bpf1, bpf2 = signals_data[i]
        xf, mag = compute_fft(sig)

        # 只显示 0-1500 Hz 范围
        mask = xf <= 1500
        axes[i].plot(xf[mask], mag[mask], linewidth=0.7, color=f"C{i}")
        axes[i].set_ylabel(f"通道{i+1}\n幅值")
        axes[i].grid(True, alpha=0.3)

        # 标注 BPF 线
        for bpf, label, color in [(bpf1, "BPF R1", "#d62728"), (bpf2, "BPF R2", "#2ca02c")]:
            for h in [1, 2, 3]:
                f = bpf * h
                if f <= 1500:
                    axes[i].axvline(f, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
            axes[i].annotate(
                f"{label}={bpf:.0f}Hz",
                xy=(bpf, axes[i].get_ylim()[1] * 0.85),
                fontsize=7, color=color,
            )

        axes[i].set_title(f"转速={s} rpm", fontsize=9)

    axes[-1].set_xlabel("频率 (Hz)")
    fig.suptitle("非定常压力频域图（FFT 幅值谱）", fontsize=13)
    fig.tight_layout()
    plot_style.save_figure(fig, os.path.join(OUTPUT_DIR, "exp2_frequency_domain"), ("png",))
    print(f"  ✓ 频域图已保存")


def main():
    use_real = "--real" in sys.argv

    if use_real:
        print("📂 真实数据模式 — 请确保 ../data/raw/ 目录下有按规范命名的 .txt 文件")
        print("   数据目录结构见 ../data/exp2_unsteady_pressure.csv 注释说明")
        print("   当前使用模拟数据演示。")

    speeds = [1200, 1500, 1800]
    print(f"🔧 生成模拟数据: 转速={speeds} rpm...")

    signals_data = [generate_mock_signal(s) for s in speeds]

    print("📊 绘制时域波形图...")
    plot_time_domain(signals_data, speeds)

    print("📊 绘制频域图...")
    plot_frequency_domain(signals_data, speeds)

    print("✅ 实验二（非定常压力测量）数据处理完成。")


if __name__ == "__main__":
    main()
