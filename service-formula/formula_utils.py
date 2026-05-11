import re
import xml.etree.ElementTree as ET
from typing import List

from latex2mathml.converter import convert as latex2mathml

try:
    import requests as _requests
except ImportError:
    _requests = None


def trim_math_delimiters(formula: str) -> str:
    if not formula:
        return formula
    s = formula.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 2:
        return s[1:-1].strip()
    return s


def latex_to_mathml(latex_formula: str) -> str:
    if not latex_formula or not latex_formula.strip():
        return ""

    pmml_full = latex2mathml(latex_formula)
    try:
        root = ET.fromstring(pmml_full)
        semantics = root.find("{http://www.w3.org/1998/Math/MathML}semantics")
        if semantics is not None:
            mrow = semantics.find("{http://www.w3.org/1998/Math/MathML}mrow")
            pmml_body = ET.tostring(mrow if mrow is not None else semantics, encoding="unicode")
        else:
            pmml_body = pmml_full
    except ET.ParseError:
        pmml_body = pmml_full

    return (
        f'<math xmlns="http://www.w3.org/1998/Math/MathML" '
        f'alttext="{latex_formula}" class="ltx_Math" display="block">'
        f"<semantics>{pmml_body}</semantics></math>"
    )


def extract_formulas(text: str) -> List[str]:
    if not text:
        return []

    # Minimum character length for a string to be treated as a formula.
    _MIN_FORMULA_LEN = 1

    placeholder = "__ESCAPED_DOLLAR__"
    working = text.replace(r"\$", placeholder)

    extracted: List[str] = []
    extracted.extend(re.findall(r"\$\$(.+?)\$\$", working, flags=re.DOTALL))
    without_display = re.sub(r"\$\$(.+?)\$\$", "", working, flags=re.DOTALL)
    extracted.extend(re.findall(r"\$([^$]+?)\$", without_display))

    if not extracted:
        candidate = re.sub(r"\\text\{[^}]*\}", "", text).strip()
        math_indicators = [
            r"[a-zA-Z]\^[\d{]", r"[a-zA-Z]_[\d{]", r"\\frac\{", r"\\sqrt[\[{]",
            r"\\sum\b", r"\\int\b", r"[+\-*/=<>≤≥≠]",
        ]
        if any(re.search(p, candidate) for p in math_indicators) and len(candidate) > _MIN_FORMULA_LEN:
            extracted.append(candidate)

    cleaned: List[str] = []
    for formula in extracted:
        formula = formula.strip().replace(placeholder, r"\$")
        if len(formula) <= 1 or formula.isdigit() or formula.isspace() or formula.isalpha():
            continue
        cleaned.append(formula)

    return list(dict.fromkeys(cleaned))


def scrape_mse_post_text(thread_id: int, max_chars: int = 1500) -> str:
    """Scrape the question and top answers text from a Math Stack Exchange post."""
    if _requests is None:
        return ""
    try:
        from bs4 import BeautifulSoup
        url = f"https://math.stackexchange.com/q/{thread_id}"
        headers = {"User-Agent": "MathMexFormulaHandler/0.1"}
        resp = _requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        parts: List[str] = []

        # Question body
        question_body = soup.select_one(".question .s-prose")
        if question_body:
            parts.append("Question:\n" + question_body.get_text(separator=" ", strip=True))

        # Top answers (up to 2)
        for answer in soup.select(".answer .s-prose")[:2]:
            parts.append("Answer:\n" + answer.get_text(separator=" ", strip=True))

        text = "\n\n".join(parts).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception as err:
        return f"(could not fetch post text: {err})"
