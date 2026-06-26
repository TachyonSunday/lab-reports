"""
通用实验数据文件读取器
========================
支持实验室常见的 .dat 格式和 .csv 格式，自动检测分隔符和注释行。

支持的格式：
  1. CSV (.csv)：逗号分隔，首行为列名
  2. 表格式 .dat：空格/制表符/逗号分隔的列数据
  3. 注释式 .dat：# % // 开头的行视为注释
  4. 纯数值 .dat：无表头，纯数值矩阵

用法：
    from dat_reader import read_lab_data

    # 自动检测格式
    df = read_lab_data("data/exp1.dat")

    # 指定列名（当文件无表头时）
    df = read_lab_data("data/exp1.dat", columns=["P","T","F","n1","n2"])

    # 也兼容 CSV
    df = read_lab_data("data/exp1.csv")

    # 查找数据文件（dat 优先）
    path = find_data_file("data/", "exp1")
"""

import os
import re
import pandas as pd
import numpy as np

# 注释符号
COMMENT_PREFIXES = ("#", "%", "//", ";", "!")


def find_data_file(data_dir, base_name):
    """
    在 data_dir 中查找数据文件，按优先级：.dat > .txt > .csv

    返回文件路径，找不到返回 None。
    """
    for ext in (".dat", ".txt", ".csv"):
        path = os.path.join(data_dir, f"{base_name}{ext}")
        if os.path.exists(path):
            return path
    # 尝试在 raw/ 子目录查找
    for ext in (".dat", ".txt"):
        path = os.path.join(data_dir, "raw", f"{base_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def _detect_separator(lines):
    """
    从一组数据行中自动检测分隔符。
    返回 (sep, is_fixed_width)
    """
    if len(lines) < 2:
        return ",", False

    # 统计每种分隔符在各行出现的次数一致性
    candidates = [
        (",", ","),
        ("\t", "\\t"),
        (r"\s+", "whitespace"),
    ]

    best_sep = ","
    best_score = -1

    for sep_pattern, _ in candidates:
        if sep_pattern == r"\s+":
            # 空格分隔：各行的字段数应基本一致
            counts = [len(re.split(r"\s+", l.strip())) for l in lines if l.strip()]
        else:
            counts = [len(l.split(sep_pattern)) for l in lines if l.strip()]

        if not counts:
            continue

        # 评分：众数的出现频率
        from collections import Counter
        counter = Counter(counts)
        most_common_count, most_common_freq = counter.most_common(1)[0]
        score = most_common_freq / len(counts)

        if score > best_score and most_common_count >= 2:
            best_score = score
            best_sep = sep_pattern

    return best_sep, (best_sep == r"\s+")


def _read_lines_without_comments(filepath, encoding="utf-8"):
    """读取文件，跳过注释行和空行，返回 (所有行, 数据行列表)"""
    # 尝试多种编码
    for enc in [encoding, "gbk", "gb2312", "latin-1", "utf-8-sig"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                all_lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法识别文件编码: {filepath}")

    data_lines = []
    for line in all_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in COMMENT_PREFIXES):
            continue
        # 跳过纯标题/说明行（全是非数字字符且包含中文或字母）
        data_lines.append(stripped)

    return all_lines, data_lines


def read_lab_data(filepath, columns=None, encoding="utf-8"):
    """
    读取实验数据文件，自动检测格式。

    参数:
        filepath:   数据文件路径（支持 .dat .txt .csv）
        columns:    列名列表。为 None 时尝试自动检测表头。
        encoding:   文件编码，默认 utf-8

    返回:
        pandas DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    # CSV 文件直接用 pandas 读取
    if ext == ".csv":
        # 跳过注释行
        with open(filepath, "r", encoding=encoding) as f:
            header_line = None
            for line in f:
                stripped = line.strip()
                if stripped and not any(stripped.startswith(p) for p in COMMENT_PREFIXES):
                    header_line = stripped
                    break
        if header_line and "," in header_line:
            return pd.read_csv(filepath, comment="#", encoding=encoding)
        else:
            # 无表头 CSV
            df = pd.read_csv(filepath, comment="#", header=None, encoding=encoding)
            if columns:
                df.columns = columns
            return df

    # .dat / .txt 文件
    _, data_lines = _read_lines_without_comments(filepath, encoding)

    if not data_lines:
        raise ValueError(f"数据文件无有效数据行: {filepath}")

    sep, is_fixed = _detect_separator(data_lines[:50])

    # 判断首行是否为表头（包含非数值内容）
    first_line = data_lines[0]
    has_header = False
    if columns is None:
        # 用正则检查首行是否包含非数值字符（字母/中文）
        tokens = re.split(sep, first_line) if not is_fixed else re.split(r"\s+", first_line.strip())
        numeric_count = 0
        for t in tokens:
            t = t.strip('"\'')
            try:
                float(t)
                numeric_count += 1
            except ValueError:
                pass
        # 如果超过一半的 token 不是纯数字，认为是表头
        has_header = numeric_count < len(tokens) * 0.5

    # 构建 pandas 读取参数
    if has_header:
        # 首行即表头
        header_line_idx = 0
        for i, line in enumerate(data_lines):
            if line == first_line:
                header_line_idx = i
                break
        # 将数据行写入临时内容
        text = "\n".join(data_lines)
        from io import StringIO
        if sep == r"\s+":
            df = pd.read_csv(StringIO(text), sep=sep, engine="python")
        else:
            df = pd.read_csv(StringIO(text), sep=sep)
    else:
        # 无表头
        text = "\n".join(data_lines)
        from io import StringIO
        if sep == r"\s+":
            df = pd.read_csv(StringIO(text), sep=sep, header=None, engine="python")
        else:
            df = pd.read_csv(StringIO(text), sep=sep, header=None)

        if columns:
            if len(columns) == len(df.columns):
                df.columns = columns
            else:
                # 列数不匹配时，对齐
                n = min(len(columns), len(df.columns))
                new_cols = list(df.columns)
                new_cols[:n] = columns[:n]
                df.columns = new_cols

    # 清理列名中的空白字符
    df.columns = [str(c).strip().replace(" ", "_").replace("(", "_").replace(")", "").replace("°", "deg")
                  for c in df.columns]

    return df


# ── 测试 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=== dat_reader 自测 ===\n")

    # 测试1：空格分隔的 .dat
    content1 = """# 实验数据 2024-01-15
    # 转速=1200rpm
    speed throttle P_in P_out
    1200 170 101325 102500
    1200 270 101320 103200
    1200 320 101318 103800
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False, encoding="utf-8") as f:
        f.write(content1)
        tmp1 = f.name

    df1 = read_lab_data(tmp1)
    print(f"测试1 — 空格分隔.dat (有表头):")
    print(df1)
    print()

    # 测试2：无表头 .dat + 手动指定列名
    content2 = """-25 100123 101456 100789 105000 100000
    -20 100234 101478 100812 105000 100000
    -15 100345 101489 100856 105000 100000
    0 100456 101500 100901 105000 100000
    15 100567 101511 100945 105000 100000
    25 100678 101522 100989 105000 100000
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False, encoding="utf-8") as f:
        f.write(content2)
        tmp2 = f.name

    df2 = read_lab_data(tmp2, columns=["alpha", "P1", "P2", "P3", "Pt", "Ps"])
    print(f"测试2 — 无表头.dat (手动指定列名):")
    print(df2)
    print()

    # 测试3：CSV 兼容
    content3 = """a,b,c
    1,2,3
    4,5,6
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content3)
        tmp3 = f.name

    df3 = read_lab_data(tmp3)
    print(f"测试3 — CSV 兼容:")
    print(df3)
    print()

    os.unlink(tmp1)
    os.unlink(tmp2)
    os.unlink(tmp3)
    print("✅ 所有测试通过")
