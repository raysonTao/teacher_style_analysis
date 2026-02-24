#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考文献验证脚本
- 从 references_generated.bib 中解析每条 \bibitem
- 通过 Semantic Scholar API 查询论文（英文）
- 通过 CrossRef API 作为备用
- 计算标题相似度验证引用准确性
- 输出验证报告 references_verified.md
"""

import urllib.request
import urllib.parse
import json
import re
import time
import difflib
import os

# ── 路径配置 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIB_FILE = os.path.join(BASE_DIR, "references_generated.bib")
OUT_BIB  = os.path.join(BASE_DIR, "references_verified.bib")
REPORT   = os.path.join(BASE_DIR, "verification_report.md")

# ── 相似度阈值 ────────────────────────────────────────────────────────────────
THRESHOLD_PASS  = 0.80  # ≥80% 视为通过
THRESHOLD_WARN  = 0.55  # 55%~80% 警告
# <55% 视为失败（未找到或不匹配）

# ── API 工具 ──────────────────────────────────────────────────────────────────
def http_get(url, headers=None, timeout=12):
    h = {"User-Agent": "RefVerifier/1.0 (academic research)"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def semantic_scholar_search(title, max_retries=4):
    """Search Semantic Scholar by title, return best match or None."""
    q = urllib.parse.quote(title[:200])
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={q}&limit=3"
        "&fields=title,authors,year,venue,journal,externalIds,publicationVenue"
    )
    for attempt in range(max_retries):
        try:
            data = http_get(url)
            if data.get("data"):
                return data["data"]  # list of candidates
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 8 * (attempt + 1)
                print(f"    [429] rate limit, wait {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] {e}")
                break
        except Exception as e:
            print(f"    [ERR] {e}")
            break
    return None

def crossref_search(title, max_retries=3):
    """Search CrossRef by title, return best match or None."""
    q = urllib.parse.quote(title[:200])
    url = (
        "https://api.crossref.org/works"
        f"?query.title={q}&rows=3&select=title,author,published,container-title,DOI"
    )
    for attempt in range(max_retries):
        try:
            data = http_get(url, headers={"mailto": "student@ecnu.edu.cn"})
            items = data.get("message", {}).get("items", [])
            if items:
                return items
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"    [CrossRef 429] wait {wait}s...")
                time.sleep(wait)
            else:
                break
        except Exception as e:
            print(f"    [CrossRef ERR] {e}")
            break
    return None

# ── 字符串工具 ────────────────────────────────────────────────────────────────
def normalize(s):
    """Lowercase, strip punctuation/whitespace for comparison."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def best_match(candidates, query_title):
    """Return (best_item, score) from a list of candidates."""
    best, best_score = None, 0.0
    for c in candidates:
        t = c.get("title") or ""
        if isinstance(t, list):   # CrossRef returns list
            t = t[0] if t else ""
        s = similarity(query_title, t)
        if s > best_score:
            best, best_score = c, s
    return best, best_score

# ── 解析 .bib ─────────────────────────────────────────────────────────────────
def parse_bib(path):
    """
    Parse \bibitem{key} ... entries.
    Returns list of dict: {key, raw, title_guess, is_chinese, is_todo}
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # Split on \bibitem boundaries
    parts = re.split(r"(?=\\bibitem\{)", text)
    entries = []
    for part in parts:
        part = part.strip()
        if not part.startswith(r"\bibitem"):
            continue
        m = re.match(r"\\bibitem\{([^}]+)\}\s*(.*)", part, re.DOTALL)
        if not m:
            continue
        key  = m.group(1).strip()
        body = m.group(2).strip()
        is_todo    = "% TODO" in body
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in body)

        # Guess title: for English, first quoted or italic segment, or 2nd sentence
        title_guess = ""
        if is_chinese:
            # Chinese: first Chinese phrase before [J] or period
            cm = re.search(r'[\u4e00-\u9fff][^\[。.]{5,60}', body)
            title_guess = cm.group(0).strip() if cm else ""
        else:
            # English: text inside \textit{...} if present
            tm = re.search(r'\\textit\{([^}]+)\}', body)
            if tm:
                pass  # that's venue, not title
            # Title is usually after "key} Author. (year). TITLE. venue"
            # Heuristic: sentence after "). "
            tm2 = re.search(r'\(\d{4}\)\.\s+(.+?)\.\s+\\textit', body, re.DOTALL)
            if tm2:
                title_guess = tm2.group(1).strip()
            else:
                # fallback: 2nd segment split by ". "
                segs = [s.strip() for s in body.split(". ") if len(s.strip()) > 10]
                title_guess = segs[1] if len(segs) > 1 else segs[0] if segs else ""

        entries.append({
            "key":          key,
            "raw":          body,
            "title_guess":  title_guess,
            "is_chinese":   is_chinese,
            "is_todo":      is_todo,
        })
    return entries

# ── 格式化 API 结果为标准 \bibitem ────────────────────────────────────────────
def format_from_ss(key, paper):
    authors = paper.get("authors", [])
    auth_str = ", ".join(a["name"] for a in authors[:6])
    if len(authors) > 6:
        auth_str += ", et al."
    year    = paper.get("year", "n.d.")
    title   = paper.get("title", "")
    venue   = ""
    if paper.get("journal") and paper["journal"].get("name"):
        venue = paper["journal"]["name"]
    elif paper.get("venue"):
        venue = paper["venue"]
    doi = ""
    ext = paper.get("externalIds", {})
    if ext.get("DOI"):
        doi = f" DOI: {ext['DOI']}."
    return (
        f"\\bibitem{{{key}}} {auth_str}. ({year}). {title}."
        f" \\textit{{{venue}}}.{doi}"
    )

def format_from_cr(key, item):
    authors = item.get("author", [])
    auth_parts = []
    for a in authors[:6]:
        name = a.get("family", "") + (", " + a.get("given", "") if a.get("given") else "")
        auth_parts.append(name.strip(", "))
    auth_str = ", ".join(auth_parts)
    if len(authors) > 6:
        auth_str += ", et al."
    titles = item.get("title", [""])
    title = titles[0] if titles else ""
    pub = item.get("published", {}).get("date-parts", [[None]])[0]
    year = pub[0] if pub else "n.d."
    venue = ""
    ct = item.get("container-title", [])
    if ct:
        venue = ct[0]
    doi = item.get("DOI", "")
    doi_str = f" DOI: {doi}." if doi else ""
    return (
        f"\\bibitem{{{key}}} {auth_str}. ({year}). {title}."
        f" \\textit{{{venue}}}.{doi_str}"
    )

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  参考文献验证脚本")
    print("=" * 60)

    entries = parse_bib(BIB_FILE)
    print(f"\n解析到 {len(entries)} 条条目\n")

    results = []   # list of result dicts
    verified_bibs = []  # final corrected bibitem lines

    for i, entry in enumerate(entries):
        key   = entry["key"]
        title = entry["title_guess"]
        print(f"[{i+1:02d}/{len(entries)}] {key}")
        print(f"        title_guess: {title[:70]}")

        r = {
            "key":        key,
            "original":   entry["raw"],
            "title_in":   title,
            "is_chinese": entry["is_chinese"],
            "is_todo":    entry["is_todo"],
            "status":     "SKIP",
            "score":      None,
            "api_title":  None,
            "api_source": None,
            "corrected":  None,
        }

        # ── Skip TODO entries ──────────────────────────────────────────────
        if entry["is_todo"]:
            print("        → SKIP (TODO)")
            r["status"] = "TODO"
            results.append(r)
            verified_bibs.append(f"\\bibitem{{{key}}} % TODO: 未能自动获取，请手动补充\n")
            continue

        # ── Skip Chinese entries (SS has limited coverage) ─────────────────
        if entry["is_chinese"]:
            print("        → SKIP (中文，人工核验)")
            r["status"] = "CHINESE_SKIP"
            results.append(r)
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")
            continue

        if not title or len(title) < 8:
            print("        → SKIP (无法提取标题)")
            r["status"] = "NO_TITLE"
            results.append(r)
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")
            continue

        # ── Try Semantic Scholar ───────────────────────────────────────────
        time.sleep(1.5)  # polite delay
        candidates = semantic_scholar_search(title)
        api_source = "SemanticScholar"

        if not candidates:
            # Fallback: CrossRef
            print("        SS not found, trying CrossRef...")
            time.sleep(1.0)
            candidates_cr = crossref_search(title)
            if candidates_cr:
                # convert CrossRef format
                best, score = best_match(
                    [{"title": c.get("title", [""])[0], "_cr": c}
                     for c in candidates_cr],
                    title
                )
                api_source = "CrossRef"
                if best and score >= THRESHOLD_WARN:
                    cr_item = best["_cr"]
                    api_title = (cr_item.get("title") or [""])[0]
                    r["api_title"]  = api_title
                    r["api_source"] = api_source
                    r["score"]      = score
                    corrected = format_from_cr(key, cr_item)
                    r["corrected"]  = corrected
                    status = "PASS" if score >= THRESHOLD_PASS else "WARN"
                    r["status"] = status
                    print(f"        CrossRef → {status} (score={score:.2f})")
                    print(f"        api_title: {api_title[:70]}")
                    bib_line = corrected if score >= THRESHOLD_PASS else entry["raw"]
                    verified_bibs.append(f"\\bibitem{{{key}}} {bib_line}\n")
                    results.append(r)
                    continue
            r["status"] = "NOT_FOUND"
            print("        → NOT FOUND on SS or CrossRef")
            results.append(r)
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")
            continue

        best, score = best_match(candidates, title)
        if best is None:
            r["status"] = "NOT_FOUND"
            results.append(r)
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")
            continue

        api_title = best.get("title", "")
        r["api_title"]  = api_title
        r["api_source"] = api_source
        r["score"]      = score

        if score >= THRESHOLD_PASS:
            corrected = format_from_ss(key, best)
            r["corrected"] = corrected
            r["status"]    = "PASS"
            verified_bibs.append(f"\\bibitem{{{key}}} {corrected}\n")
        elif score >= THRESHOLD_WARN:
            corrected = format_from_ss(key, best)
            r["corrected"] = corrected
            r["status"]    = "WARN"
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")  # keep original
        else:
            r["status"] = "FAIL"
            verified_bibs.append(f"\\bibitem{{{key}}} {entry['raw']}\n")

        print(f"        → {r['status']} (score={score:.2f})")
        print(f"        api_title: {api_title[:70]}")
        results.append(r)

    # ── Write verified .bib ───────────────────────────────────────────────────
    with open(OUT_BIB, "w", encoding="utf-8") as f:
        f.write("% ============================================================\n")
        f.write("% 参考文献条目 — 已验证版本\n")
        f.write("% 生成时间: 2026-02-24\n")
        f.write("% ============================================================\n\n")
        for line in verified_bibs:
            f.write(line)
            f.write("\n")

    # ── Write Markdown report ─────────────────────────────────────────────────
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ["PASS", "WARN", "FAIL", "NOT_FOUND", "TODO",
                        "CHINESE_SKIP", "NO_TITLE"]}

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# 参考文献验证报告\n\n")
        f.write(f"生成时间：2026-02-24  \n")
        f.write(f"来源文件：`references_generated.bib`  \n\n")

        f.write("## 总览\n\n")
        f.write(f"| 状态 | 数量 | 说明 |\n")
        f.write(f"|------|------|------|\n")
        f.write(f"| ✅ PASS     | {counts.get('PASS',0)} | 相似度 ≥ 80%，已用API结果更新格式 |\n")
        f.write(f"| ⚠️ WARN     | {counts.get('WARN',0)} | 相似度 55%~80%，保留原格式，建议人工核验 |\n")
        f.write(f"| ❌ FAIL     | {counts.get('FAIL',0)} | 相似度 < 55%，未自动替换 |\n")
        f.write(f"| 🔍 NOT_FOUND| {counts.get('NOT_FOUND',0)} | API 未返回结果 |\n")
        f.write(f"| 📝 TODO     | {counts.get('TODO',0)} | 原始文件中标记为待补充 |\n")
        f.write(f"| 🇨🇳 中文     | {counts.get('CHINESE_SKIP',0)} | 中文论文，跳过API验证 |\n")
        f.write(f"| ❓ NO_TITLE | {counts.get('NO_TITLE',0)} | 无法提取标题 |\n\n")

        f.write("---\n\n")
        f.write("## 逐条结果\n\n")

        for r in results:
            status_icon = {
                "PASS": "✅", "WARN": "⚠️", "FAIL": "❌",
                "NOT_FOUND": "🔍", "TODO": "📝",
                "CHINESE_SKIP": "🇨🇳", "NO_TITLE": "❓",
            }.get(r["status"], "❓")

            f.write(f"### {status_icon} `{r['key']}`\n\n")
            f.write(f"**状态：** {r['status']}")
            if r["score"] is not None:
                f.write(f"  （相似度：{r['score']:.2%}，来源：{r['api_source']}）")
            f.write("\n\n")

            f.write(f"**原始条目：**\n```\n{r['original']}\n```\n\n")

            if r["api_title"]:
                f.write(f"**API 返回标题：** {r['api_title']}\n\n")

            if r["corrected"] and r["status"] == "PASS":
                f.write(f"**已更新为：**\n```\n{r['corrected']}\n```\n\n")
            elif r["corrected"] and r["status"] == "WARN":
                f.write(f"**建议格式（未自动替换）：**\n```\n{r['corrected']}\n```\n\n")

            if r["status"] in ("FAIL", "NOT_FOUND"):
                f.write(f"> ⚠️ 建议手动通过 Google Scholar 搜索标题验证。\n\n")

            f.write("---\n\n")

    print("\n" + "=" * 60)
    print("  验证完成")
    print("=" * 60)
    print(f"  ✅ PASS     : {counts.get('PASS',0)}")
    print(f"  ⚠️  WARN     : {counts.get('WARN',0)}")
    print(f"  ❌ FAIL     : {counts.get('FAIL',0)}")
    print(f"  🔍 NOT_FOUND: {counts.get('NOT_FOUND',0)}")
    print(f"  📝 TODO     : {counts.get('TODO',0)}")
    print(f"  🇨🇳 中文     : {counts.get('CHINESE_SKIP',0)}")
    print(f"\n  已写入: {OUT_BIB}")
    print(f"  报告  : {REPORT}")

if __name__ == "__main__":
    main()
