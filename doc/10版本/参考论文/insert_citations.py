#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动向论文 .tex 章节文件插入 \\cite{} 命令。

匹配规则（按优先级）：
1. 中文作者名 + 年份：如 "胡小勇等（2018）"
2. 英文作者名 + 年份：如 "Yan等人（2018）" / "Vaswani et al. (2017)"
3. 技术关键词首次出现：如 "SHapley Additive exPlanations"

插入位置：在匹配串末尾、下一个标点（句号/逗号/右括号）之前。
"""

import re
import os
import shutil
from datetime import datetime

CHAPTERS_DIR = "/home/rayson/code/teacher_style_analysis/doc/10版本/ecnu_latex_template/chapters"
LOG_FILE     = "/home/rayson/code/teacher_style_analysis/doc/10版本/参考论文/citation_changes.log"

# ── 匹配规则表 ────────────────────────────────────────────────────────────────
# 每条规则：(cite_key, [pattern1, pattern2, ...], chapters, match_once)
# match_once=True: 全文只插一次（用于技术术语首次出现）
RULES = [
    # ── 中文教学理论 ──────────────────────────────────────────────────────────
    ("ref_intent",        [r"Flanders.*?(?:1963|1960|196\d)", r"互动分析系统.*?FIAS", r"FIAS.*?互动分析", r"Flanders互动分析系统"], [1,2,5], False),
    ("ref_anderson1939",  [r"Anderson.*?1939", r"H\.H\. Anderson"], [1,2], False),
    ("ref_grasha1996",    [r"Grasha.*?(?:1996|五分类|五种)", r"Grasha（1996）"], [1,2], False),
    ("ref_pianta2008",    [r"Pianta等人.*?(?:2008|2007)", r"CLASS评价工具", r"Classroom Assessment Scoring System"], [1,2], False),

    # ── 中文教师画像/多模态 ───────────────────────────────────────────────────
    ("ref_hu2024",        [r"胡小勇.*?(?:2018|2024).*?画像", r"胡小勇等.*?2018"], [1,2], False),
    ("ref_hu2024b",       [r"胡小勇.*?研修.*?2024", r"胡小勇.*?2024.*?研修"], [2], False),
    ("ref_zhang2022",     [r"张乐乐.*?2022", r"顾小清.*?2022"], [2], False),
    ("ref_chen2021",      [r"陈鑫.*?2021", r"胡东芳.*?2021"], [1,2], False),
    ("ref_bai2024",       [r"柏宏权.*?2024", r"小学人工智能教师画像"], [2], False),
    ("ref_hu2024portrait",[r"Hu.*?2024.*?portrait", r"micro ecological.*?teacher", r"胡小勇.*?AI.*?时代.*?教师画像"], [1,2], False),

    # ── 英文教学分析系统 ──────────────────────────────────────────────────────
    ("ref_canovas2023",   [r"Canovas.*?2023", r"AI.driven Teacher Analytics", r"AI-driven.*?Teacher Analytics"], [1,2], False),
    ("ref_classmind",     [r"ClassMind.*?(?:2024|2025)", r"ClassMind系统"], [1,2], False),
    ("ref_zhou2019",      [r"Zhou.*?2019.*?[Tt]eaching [Ss]tyle", r"Understanding the Teaching Styles.*?Attention"], [1,3], False),
    ("ref_iseeyou",       [r"Lee.*?2024.*?GPT.4.*?[Tt]eacher", r"I See You.*?Teacher", r"GPT-4.*?观察性评估.*?教师"], [1,2], False),
    ("ref_han",           [r"Han.*?2020.*?[Tt]ext [Cc]lassif", r"Qing Han.*?2020", r"韩清.*?文本分类.*?教师行为"], [1,2], False),
    ("ref_borchers2023",  [r"Borchers.*?2023", r"Transmodal Ordered Network", r"AI.*?教室.*?教师实践网络"], [1,2], False),
    ("ref_rafique",       [r"Rafique.*?2022", r"automatic content recognition.*?teaching"], [1,2], False),
    ("ref_sapena",        [r"Sapena.*?2022", r"Multimodal Classification.*?Teaching Activities.*?University"], [2,3], False),
    ("ref_riordan2025",   [r"Riordan.*?2024", r"Multimodal classroom interaction.*?ungrouping"], [1,2], False),

    # ── 视频/动作识别 ─────────────────────────────────────────────────────────
    ("ref_yan2018",       [r"Yan等人.*?(?:2018|ST-GCN)", r"Yan.*?2018.*?ST-GCN", r"ST-GCN.*?Yan.*?2018", r"空间时序图卷积网络.*?Yan"], [1,2,3], False),
    ("ref_li2022",        [r"Li.*?2022.*?[Pp]ose [Ss]equen", r"Yuanzhong Li.*?2022", r"2022.*?Long.*?Short.*?Pose"], [2,3], False),
    ("ref_liu2023",       [r"Liu.*?2023.*?[Ff]acial [Ee]xpression", r"Ziyi Liu.*?2023"], [2,3], False),
    ("ref_wu2024mctm",    [r"Wu.*?2024.*?MCTM", r"Motion Complement.*?Temporal Multifocusing"], [2,3], False),
    ("ref_sitmlp2024",    [r"SiT-MLP.*?(?:2023|2024)", r"Zhang.*?2024.*?SiT-MLP", r"点拓扑.*?骨骼"], [2,3], False),
    ("ref_snag",          [r"SnAG.*?(?:2024)", r"Scalable and Accurate Video Grounding.*?SnAG"], [2,3], False),
    ("ref_tridet",        [r"TriDet.*?2023", r"Shi.*?2023.*?TriDet", r"Temporal Action Detection.*?Relative Boundary"], [2,3], False),
    ("ref_actionformer",  [r"ActionFormer.*?2022", r"Zhang.*?2022.*?ActionFormer"], [2,3], False),
    ("ref_cai2025",       [r"TBU.*?[Dd]ataset", r"Cai.*?2025.*?classroom", r"Cai.*?TBU"], [2,3], False),
    ("ref_csbyolo",       [r"CSB-YOLO", r"Csb.yolo.*?2024", r"Zhu.*?2024.*?classroom.*?student"], [2,3], False),

    # ── 语音/情感 ─────────────────────────────────────────────────────────────
    ("ref_baevski2020",   [r"Baevski等人.*?(?:2020)", r"Baevski.*?Wav2Vec 2\.0", r"Wav2Vec 2\.0.*?Baevski"], [1,2,3], False),
    ("ref_liang",         [r"Liang.*?2020.*?[Ee]motion", r"Speech Emotion.*?[Tt]eacher.*?Classroom", r"梁.*?2020.*?语音情感"], [1,2,3], False),
    ("ref_zhang2021speech",[r"Zhang.*?2021.*?[Ss]peech [Ii]ntention", r"张.*?2021.*?意图分类", r"Classification.*?Teachers.*?Speech Intention.*?Deep"], [1,2,3], False),
    ("ref_pardo2025audio",[r"Pardo.*?2025.*?[Aa]udio", r"Audio Features.*?Education.*?Systematic Review"], [1,2], False),

    # ── 综述/数据挖掘 ─────────────────────────────────────────────────────────
    ("ref_guerrero",      [r"Guerrero.*?2023", r"Comprehensive Review.*?Multimodal Analysis.*?Education"], [1,2], False),
    ("ref_ahmad2024",     [r"Ahmad.*?(?:2020|2024).*?AI.*?Education", r"Data-Driven.*?AI.*?Education.*?Comprehensive Review"], [1,2], False),
    ("ref_shafizadegan2024",[r"Shafizadegan.*?2024", r"Multimodal.*?human action recognition.*?deep learning.*?review"], [1,2], False),
    ("ref_saket2025",     [r"Saket.*?2025", r"abnormal human behavior.*?video surveillance.*?review"], [1,2], False),
    ("ref_schiappa2023",  [r"Schiappa.*?2023", r"Self-Supervised Learning.*?Videos.*?Survey"], [1,2], False),
    ("ref_yurum2025",     [r"Y.r.m.*?2025", r"Technology-Enhanced Multimodal Learning Analytics.*?Higher Education"], [1,2], False),
    ("ref_zhao2024review",[r"Zhao.*?2024.*?[Aa]ction [Rr]ecognition.*?[Rr]eview", r"A Review.*?State-of-the-Art.*?Action Recognition"], [1,2], False),
    ("ref_howard",        [r"Howard.*?2018.*?[Dd]ata [Mm]ining", r"Data Mining.*?Machine Learning.*?Technology-Enhanced Learning"], [1,2], False),
    ("ref_zhao2024",      [r"Zhao.*?2024.*?[Kk]nowledge [Gg]raph.*?[Tt]eacher [Pp]ortrait", r"Design.*?Teacher Portrait.*?Knowledge Graph"], [2,4], False),
    ("ref_liu2022rfm",    [r"Liu.*?2022.*?RFM.*?[Pp]ortrait", r"Animation User Value Portrait.*?RFM"], [2,4], False),

    # ── 深度学习核心方法（关键词首次出现）────────────────────────────────────
    ("ref_vaswani2017",   [r"Vaswani等人.*?(?:2017)", r"Vaswani.*?2017.*?Transformer"], [1,2,3], False),
    ("ref_devlin2018",    [r"Devlin等人.*?(?:2018)", r"Devlin.*?2018.*?BERT"], [1,2,3], False),
    ("ref_lundberg2017",  [r"SHapley Additive exPlanations", r"SHAP值.*?SHapley"], [1,3], True),
    ("ref_wojke2017",     [r"DeepSORT", r"Simple Online and Realtime Tracking.*?Deep"], [2,3], True),
    ("ref_lugaresi2019",  [r"MediaPipe.*?(?:2020|2019)", r"MediaPipe轻量化"], [1,2,3], True),
]

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def insert_cite_after(text, match_end, cite_key):
    """在 match_end 位置后插入 \\cite{key}，跳过空格。"""
    # Find insertion point: right after match, before next CJK punctuation or period
    insert_at = match_end
    # If there's already a \cite right here, skip
    if text[insert_at:insert_at+6] == r'\cite{':
        return text, False
    return text[:insert_at] + f"\\cite{{{cite_key}}}" + text[insert_at:], True


def process_file(filepath, rules, log_entries):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    changes = []

    # Track which keys have been inserted (for match_once)
    inserted_once = set()

    for key, patterns, chapters, match_once in rules:
        if match_once and key in inserted_once:
            continue

        for pat in patterns:
            try:
                regex = re.compile(pat, re.UNICODE | re.DOTALL)
            except re.error:
                continue

            new_lines = []
            modified = False

            for lineno, line in enumerate(lines, 1):
                # Skip comment lines and lines already having this cite
                if line.strip().startswith("%"):
                    new_lines.append(line)
                    continue
                if f"\\cite{{{key}}}" in line:
                    new_lines.append(line)
                    continue

                m = regex.search(line)
                if m:
                    new_line, did_insert = insert_cite_after(line, m.end(), key)
                    if did_insert:
                        changes.append({
                            "lineno": lineno,
                            "key": key,
                            "pattern": pat,
                            "before": line.strip()[:80],
                            "after": new_line.strip()[:80],
                        })
                        modified = True
                        if match_once:
                            inserted_once.add(key)
                        new_lines.append(new_line)
                        continue
                new_lines.append(line)

            if modified:
                lines = new_lines

    final_content = "\n".join(lines)
    log_entries.extend(changes)
    return final_content, len(changes)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    log_entries = []
    total_inserts = 0

    chapter_files = [
        ("chapter-1.tex", [1]),
        ("chapter-2.tex", [2]),
        ("chapter-3.tex", [3]),
        ("chapter-4.tex", [4]),
        ("chapter-5.tex", [5]),
    ]

    for fname, chapter_nums in chapter_files:
        fpath = os.path.join(CHAPTERS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP (not found): {fname}")
            continue

        # Filter rules that apply to this chapter
        applicable = [(k, p, chs, mo) for (k, p, chs, mo) in RULES
                      if any(c in chs for c in chapter_nums)]

        # Backup
        backup = fpath + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(fpath, backup)

        print(f"\nProcessing {fname} ({len(applicable)} applicable rules)...")
        new_content, n = process_file(fpath, applicable, log_entries)

        if n > 0:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"  → {n} citations inserted")
        else:
            print(f"  → no changes")

        total_inserts += n

    # Write log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Citation Insertion Log — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total insertions: {total_inserts}\n")
        f.write("=" * 70 + "\n\n")
        for e in log_entries:
            f.write(f"Line {e['lineno']:4d} | \\cite{{{e['key']}}}\n")
            f.write(f"  Pattern : {e['pattern']}\n")
            f.write(f"  Before  : {e['before']}\n")
            f.write(f"  After   : {e['after']}\n\n")

    print(f"\n{'='*50}")
    print(f"Total citations inserted: {total_inserts}")
    print(f"Log written to: {LOG_FILE}")

if __name__ == "__main__":
    main()
