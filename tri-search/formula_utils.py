import re
import xml.etree.ElementTree as ET
from typing import List

from latex2mathml.converter import convert as latex2mathml


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
        # Edit min formula length here!!!
        if any(re.search(p, candidate) for p in math_indicators) and len(candidate) > 1:
            extracted.append(candidate)

    cleaned: List[str] = []
    for formula in extracted:
        formula = formula.strip().replace(placeholder, r"\$")
        if len(formula) <= 1 or formula.isdigit() or formula.isspace() or formula.isalpha():
            continue
        cleaned.append(formula)

    return list(dict.fromkeys(cleaned))
