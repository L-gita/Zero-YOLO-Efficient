# -*- coding: utf-8 -*-
"""
把 Markdown 渲染成一个自包含的 HTML 文件（离线可看，中文字体、表格、代码块、公式占位）。
用法：
    python md_to_html.py 输入.md [输出.html]
不指定输出时，输出到输入文件同目录、同名 .html。
"""
import os
import re
import sys

import markdown

# LaTeX 里常见的符号 → Unicode，让不做 KaTeX 渲染时也能看懂
MATH_REPL = [
    (r"\\alpha", "α"), (r"\\cdot", "·"), (r"\\times", "×"), (r"\\leq", "≤"), (r"\\geq", "≥"),
    (r"\\left|", "|"), (r"\\right|", "|"), (r"\\left\(", "("), (r"\\right\)", ")"),
    (r"\\sum_\{([^{}]*)\}", r"Σ_\1"), (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\mathcal\{L\}", "L"), (r"\^2", "²"), (r"\^\{2\}", "²"),
]

CSS = """
:root { --fg:#1f2328; --bg:#ffffff; --muted:#57606a; --line:#d0d7de; --code:#f6f8fa; --accent:#0969da; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 32px 40px 80px; max-width: 980px;
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", system-ui, sans-serif;
  font-size: 16px; line-height: 1.75; color: var(--fg); background: var(--bg);
  word-wrap: break-word;
}
h1,h2,h3,h4 { line-height: 1.3; margin: 28px 0 12px; }
h1 { font-size: 30px; padding-bottom: 10px; border-bottom: 2px solid var(--line); }
h2 { font-size: 23px; padding-bottom: 6px; border-bottom: 1px solid var(--line); margin-top: 36px; }
h3 { font-size: 19px; }
h4 { font-size: 17px; color: var(--muted); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
p, li { margin: 8px 0; }
blockquote {
  margin: 16px 0; padding: 10px 16px; color: var(--muted);
  border-left: 4px solid var(--line); background: #fafbfc; border-radius: 4px;
}
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 88%; background: var(--code); padding: 2px 5px; border-radius: 4px;
}
pre {
  background: var(--code); border: 1px solid var(--line); border-radius: 6px;
  padding: 14px 16px; overflow-x: auto; line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: 13px; }
table { border-collapse: collapse; margin: 16px 0; width: 100%; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 7px 12px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) td { background: #fcfcfd; }
hr { border: 0; border-top: 1px solid var(--line); margin: 32px 0; }
.math {
  display: block; text-align: center; font-family: "Cambria Math", "Times New Roman", serif;
  font-size: 19px; background: #f6f8fa; border-radius: 6px; padding: 12px; margin: 16px 0;
  overflow-x: auto;
}
.toc { background: #fafbfc; border: 1px solid var(--line); border-radius: 6px; padding: 12px 20px; margin: 20px 0; }
.toc ul { margin: 6px 0; padding-left: 22px; }
.footer { margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }
@media print { body { padding: 0; max-width: none; } pre, table { page-break-inside: avoid; } }
"""


def latex_to_unicode(s: str) -> str:
    for pat, rep in MATH_REPL:
        s = re.sub(pat, rep, s)
    # x^2 之类统一点
    s = re.sub(r"\^(\d)", r"^\1", s)
    return s.strip()


def convert(md_path: str, html_path: str) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 先把 $$...$$ 独立公式抽出来，避免 markdown 把里面的 * _ 当成强调
    blocks = []

    def stash(m):
        blocks.append(latex_to_unicode(m.group(1)))
        return f"\n\n@@MATH{len(blocks)-1}@@\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", stash, text, flags=re.S)
    # 行内 $...$ 也换成 <code> 更安全
    text = re.sub(r"(?<!\$)\$([^$\n]{1,120})\$(?!\$)", lambda m: f"`{latex_to_unicode(m.group(1))}`", text)

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
        extension_configs={"toc": {"title": "目录"}},
    )

    for i, b in enumerate(blocks):
        body = body.replace(f"<p>@@MATH{i}@@</p>", f'<div class="math">{b}</div>')
        body = body.replace(f"@@MATH{i}@@", f'<div class="math">{b}</div>')

    title = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = title.group(1).strip() if title else os.path.basename(md_path)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{body}
<div class="footer">由 <code>{os.path.basename(md_path)}</code> 渲染 · 生成时间 {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK -> {html_path}  ({len(html)/1024:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".html"
    convert(src, dst)
