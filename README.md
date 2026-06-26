# 航空发动机/叶轮机械原理 实验报告模板与数据处理

西北工业大学 动力与能源学院 实验报告 LaTeX 模板及 Python 数据处理脚本。

## 结构

```
lab/
├── lab1/                     # 航空发动机实践（5个实验）
│   ├── data/                 # 实验数据（CSV 模板 + 原始 .dat）
│   ├── scripts/              # Python 数据处理脚本
│   └── reports/              # LaTeX 报告 + 输出图表
│       ├── lab1_report.tex   # 主报告（编译入口）
│       ├── chapters/         # 各实验章节
│       ├── tables/           # 自动生成的三线表
│       └── figures/          # 自动生成的图表
├── lab2/                     # 叶轮机械原理（8个实验）
│   ├── data/
│   ├── scripts/
│   └── reports/
│       ├── lab2_report.tex
│       ├── chapters/
│       ├── tables/
│       └── figures/
└── common/                   # 共享资源
    ├── plot_style.py         # matplotlib 统一样式
    ├── report_template.tex   # LaTeX 通用模板
    └── dat_reader.py         # .dat/.csv 数据读取器
```

## 使用

### 编译报告

需要 XeLaTeX（推荐 TinyTeX）：

```bash
cd lab1/reports && xelatex lab1_report.tex && xelatex lab1_report.tex
```

### 数据处理

```bash
# 模拟数据演示
cd lab2/scripts && python3 process_2608.py

# 真实数据（将 .dat 文件放入 data/ 后）
cd lab1/scripts && python3 exp1_trial_run.py --real
```

## 依赖

- Python 3 + pandas, numpy, scipy, matplotlib
- XeLaTeX (TeX Live / TinyTeX) + ctex, booktabs, siunitx, caption, fancyhdr, enumitem

## 许可

本项目的代码和模板文件采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议授权。
