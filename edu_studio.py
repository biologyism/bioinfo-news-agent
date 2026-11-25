import base64
import html
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, features

BASE_DIR = Path(__file__).resolve().parent
LIBRARY_PATH = BASE_DIR / "knowledge" / "bioinfo_library.json"
OVERLAY_PATH = BASE_DIR / "assets" / "brand_overlay.png"
FONTS_DIR = BASE_DIR / "assets" / "fonts"

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]


def get_gemini_keys() -> list[str]:
    raw = os.getenv("GEMINI_API_KEY", "").strip()
    if not raw:
        return []
    keys = [k.strip() for k in re.split(r"[,;\s\n]+", raw) if k.strip()]
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
HAS_RAQM = features.check("raqm")
if not HAS_RAQM:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except Exception:
        arabic_reshaper = None
        get_display = None


def to_fa_digits(val: Any) -> str:
    return str(val).translate(PERSIAN_DIGITS)


def shape_text(text: str) -> str:
    if not text:
        return ""
    if HAS_RAQM:
        return text
    if arabic_reshaper and get_display:
        return get_display(arabic_reshaper.reshape(text))
    return text


def load_library() -> dict[str, Any]:
    if not LIBRARY_PATH.exists():
        return {"repos": [], "parent_directories": [], "github_topics": [], "iran_persian_searches": []}
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))


def score_library_items(lib: dict[str, Any], search_terms: list[str], target_cats: list[str]) -> dict[str, Any]:
    terms = [t.lower().strip() for t in search_terms if len(t.strip()) >= 2]
    cats_l = [c.lower().strip() for c in target_cats if c.strip()]

    scored_repos = []
    for r in lib.get("repos", []):
        blob = f"{r.get('cat','')} {r.get('repo','')} {r.get('desc','')} {r.get('lang','')} {r.get('kind','')}".lower()
        score = 0.0
        for c in cats_l:
            if c in (r.get("cat") or "").lower():
                score += 12.0
        for t in terms:
            if t in blob:
                score += 6.0 if t in (r.get("repo") or "").lower() else 4.0
        if score > 0:
            scored_repos.append((score, r))

    scored_repos.sort(key=lambda x: x[0], reverse=True)
    top_repos = [x[1] for x in scored_repos[:14]]
    if len(top_repos) < 6:
        # Add core/education repos as helpful baseline
        extras = [r for r in lib.get("repos", []) if r.get("cat") in {"Core bioinformatics", "Education, tutorials & benchmark"}]
        for e in extras:
            if e not in top_repos:
                top_repos.append(e)
            if len(top_repos) >= 10:
                break

    scored_topics = []
    for t_item in lib.get("github_topics", []):
        tb = f"{t_item.get('topic','')} {t_item.get('label','')}".lower()
        sc = sum(5 for t in terms if t in tb)
        if sc > 0:
            scored_topics.append((sc, t_item))
    scored_topics.sort(key=lambda x: x[0], reverse=True)

    scored_iran = []
    for ir in lib.get("iran_persian_searches", []):
        ib = f"{ir.get('title','')} {ir.get('desc','')}".lower()
        sc = sum(4 for t in terms if t in ib)
        if sc > 0 or any(w in ib for w in ["آموزش", "ایران", "فارسی", "مخازن"]):
            scored_iran.append((sc, ir))
    scored_iran.sort(key=lambda x: x[0], reverse=True)

    return {
        "repos": top_repos,
        "parent_directories": lib.get("parent_directories", [])[:4],
        "topics": [x[1] for x in scored_topics[:5]] or lib.get("github_topics", [])[:4],
        "iran_searches": [x[1] for x in scored_iran[:4]],
    }


def fetch_github_repo_meta(repo_slug: str) -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "IranBioinfoEduStudio/1.0"}
    gh_tok = os.getenv("GITHUB_TOKEN", "").strip()
    if gh_tok:
        headers["Authorization"] = f"Bearer {gh_tok}"
    try:
        r = requests.get(f"https://api.github.com/repos/{repo_slug}", headers=headers, timeout=4)
        if r.status_code == 200:
            d = r.json()
            return {
                "repo": repo_slug,
                "stars": d.get("stargazers_count", 0),
                "description": d.get("description") or "",
                "topics": (d.get("topics") or [])[:6],
                "url": d.get("html_url") or f"https://github.com/{repo_slug}",
            }
    except Exception:
        pass
    return {"repo": repo_slug, "stars": None, "description": "", "topics": [], "url": f"https://github.com/{repo_slug}"}


def plan_search_for_edu(user_query: str, topic_hint: str, all_cats: list[str]) -> dict[str, Any]:
    prompt = (
        "Map this Persian/English bioinformatics educational content request to relevant library categories and keywords.\n"
        f'User request: "{user_query or topic_hint}"\n'
        f"Available library categories: {json.dumps(all_cats, ensure_ascii=False)}\n"
        'Return ONLY JSON: {"target_categories": [str], "search_keywords": [str], "is_library_lookup_only": bool}'
    )
    # Use high-quota Flash-Lite (500 RPD) or Groq for query planning so 3.8-flash (20 RPD) is saved for writing
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            gr = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=12,
            )
            if gr.status_code == 200:
                return _extract_json_obj(gr.json()["choices"][0]["message"]["content"])
        except Exception:
            pass

    gemini_keys = get_gemini_keys()
    if gemini_keys:
        for m in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
            for api_key in gemini_keys:
                try:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                        params={"key": api_key},
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                        },
                        timeout=14,
                    )
                    if r.status_code == 200:
                        return _extract_json_obj(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                    if r.status_code == 404:
                        break
                except Exception:
                    continue

    tokens = re.split(r"[\s,،]+", f"{topic_hint} {user_query}".lower())
    return {"target_categories": [], "search_keywords": [t for t in tokens if len(t) >= 3], "is_library_lookup_only": False}


def _extract_json_obj(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_edu_json_with_failover(
    system_instruction: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    json_hint: str,
) -> dict[str, Any]:
    gemini_keys = get_gemini_keys()
    preferred = os.getenv("GEMINI_MODEL", "").strip()
    models_to_try = ([preferred] if preferred else []) + [m for m in GEMINI_MODELS if m != preferred]

    last_err = None
    if gemini_keys:
        for model in models_to_try:
            gen_cfg: dict[str, Any] = {
                "temperature": 0.4,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            }
            if any(tag in model for tag in ("3.8", "3.6", "3-flash")):
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "low"}

            payload = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": gen_cfg,
            }
            for k_idx, api_key in enumerate(gemini_keys, 1):
                try:
                    print(f"[INFO] Edu Studio calling Gemini: {model} (key #{k_idx})")
                    resp = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": api_key},
                        json=payload,
                        timeout=32,
                    )
                    if resp.status_code == 400 and "thinkingConfig" in gen_cfg:
                        gen_cfg.pop("thinkingConfig", None)
                        resp = requests.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            params={"key": api_key},
                            json=payload,
                            timeout=32,
                        )
                    if resp.status_code in {404, 503}:
                        last_err = f"HTTP {resp.status_code} on {model}: {resp.text[:180]}"
                        print(f"[WARN] Edu Gemini {model} -> {resp.status_code} (skipping model)")
                        break
                    if resp.status_code in {400, 403, 429}:
                        last_err = f"HTTP {resp.status_code} ({model} key #{k_idx}): {resp.text[:180]}"
                        print(f"[WARN] {last_err}")
                        continue
                    resp.raise_for_status()
                    text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return _extract_json_obj(text_out)
                except Exception as exc:
                    last_err = f"{model} key #{k_idx}: {exc}"
                    print(f"[WARN] Edu Gemini failed ({last_err})")

    # Multi-Provider Failover: Groq (fast!) -> Cloudflare Workers AI (with max_tokens=4096) -> OpenRouter
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    cf_acc = os.getenv("CF_ACCOUNT_ID", "").strip()
    cf_tok = os.getenv("CF_AI_TOKEN", "").strip()
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    providers = []
    if groq_key:
        providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "llama-3.3-70b-versatile", True))
        providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "openai/gpt-oss-120b", True))
    if cf_acc and cf_tok:
        cf_ep = f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1/chat/completions"
        providers.append((cf_ep, cf_tok, "@cf/meta/llama-3.3-70b-instruct-fp8-fast", False))
        providers.append((cf_ep, cf_tok, "@cf/openai/gpt-oss-120b", False))
    if or_key:
        providers.append(("https://openrouter.ai/api/v1/chat/completions", or_key, "openrouter/free", False))

    for ep, key, model, use_jm in providers:
        try:
            print(f"[INFO] Edu Studio failover -> {model}")
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            body: dict[str, Any] = {
                "model": model,
                "temperature": 0.35,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system_instruction + json_hint},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if use_jm:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(ep, headers=headers, json=body, timeout=40)
            if r.status_code == 400 and use_jm:
                body.pop("response_format", None)
                r = requests.post(ep, headers=headers, json=body, timeout=40)
            if r.status_code != 200:
                last_err = f"{model} HTTP {r.status_code}: {r.text[:200]}"
                print(f"[WARN] {last_err}")
                continue
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if not content:
                last_err = f"{model} returned empty content"
                print(f"[WARN] {last_err}")
                continue
            return _extract_json_obj(content)
        except Exception as exc:
            last_err = f"{model}: {exc}"
            print(f"[WARN] Edu failover {model} failed: {exc}")

    raise RuntimeError(f"All AI providers failed in Edu Studio: {last_err}")


PALETTES: dict[str, dict[str, str]] = {
    "bioinfo_teal": {
        "name_fa": "بیوانفورماتیک ژنومیکس",
        "bg": "#F8FAFA",
        "card": "#FFFFFF",
        "primary": "#0D9488",
        "accent": "#E87A42",
        "secondary": "#5E9DB8",
        "text": "#2E3438",
        "muted": "#5E666B",
        "blob1": "#B2E5DF",
        "blob2": "#FBD3BC",
    },
    "pastel_science": {
        "name_fa": "پاستل علمی",
        "bg": "#FAF6F8",
        "card": "#FFFFFF",
        "primary": "#C9739D",
        "accent": "#73A9D8",
        "secondary": "#E38B75",
        "text": "#3A3438",
        "muted": "#6D666A",
        "blob1": "#F5D7E3",
        "blob2": "#D6E8F8",
    },
    "smart_minimal": {
        "name_fa": "مینیمال هوشمند",
        "bg": "#F9F5EC",
        "card": "#FFFFFF",
        "primary": "#5E9DB8",
        "accent": "#8FAFA5",
        "secondary": "#B9C7B8",
        "text": "#34383B",
        "muted": "#646B70",
        "blob1": "#DDECF2",
        "blob2": "#E5EFE6",
    },
    "lab_citrus": {
        "name_fa": "مرکبات آزمایشگاهی",
        "bg": "#FFF9EA",
        "card": "#FFFFFF",
        "primary": "#4C9DA8",
        "accent": "#EAAE32",
        "secondary": "#65A98B",
        "text": "#34383B",
        "muted": "#646B70",
        "blob1": "#D5EDF0",
        "blob2": "#FBE9B9",
    },
    "terracotta_mist": {
        "name_fa": "سفال و مه",
        "bg": "#FAF3EE",
        "card": "#FFFFFF",
        "primary": "#A95F4B",
        "accent": "#8CB8B6",
        "secondary": "#CF896B",
        "text": "#393330",
        "muted": "#6E635E",
        "blob1": "#F1D8CC",
        "blob2": "#D5E8E7",
    },
    "transparent_gel": {
        "name_fa": "ژل شفاف",
        "bg": "#F2F9F8",
        "card": "#FFFFFF",
        "primary": "#75BDB5",
        "accent": "#A78BC6",
        "secondary": "#6D9CCB",
        "text": "#333A3E",
        "muted": "#5F6A70",
        "blob1": "#D7EFEA",
        "blob2": "#E5DCF3",
    },
    "polar_sunset": {
        "name_fa": "غروب قطبی",
        "bg": "#FFF6F1",
        "card": "#FFFFFF",
        "primary": "#D66F68",
        "accent": "#8CBBD5",
        "secondary": "#ED927F",
        "text": "#3A3333",
        "muted": "#6B6060",
        "blob1": "#FAD4CB",
        "blob2": "#D9EAF5",
    },
}


def _hex_to_rgb(h: str, fallback: tuple[int, int, int] = (13, 148, 136)) -> tuple[int, int, int]:
    s = (h or "").strip().lstrip("#")
    if len(s) == 6:
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except Exception:
            pass
    return fallback


def _blend_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(c1[0] * (1 - t) + c2[0] * t),
        int(c1[1] * (1 - t) + c2[1] * t),
        int(c1[2] * (1 - t) + c2[2] * t),
    )


def _extract_ref_image_profile(file_id: str, bot_token: str) -> dict[str, Any]:
    if not file_id or not bot_token:
        return {}
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getFile",
            params={"file_id": file_id},
            timeout=10,
        )
        fp = r.json().get("result", {}).get("file_path") if r.status_code == 200 else None
        if not fp:
            return {}
        img_resp = requests.get(f"https://api.telegram.org/file/bot{bot_token}/{fp}", timeout=15)
        if img_resp.status_code != 200:
            return {}
        im = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        w, h = im.size
        ratio = h / max(1, w)
        aspect = "story" if ratio >= 1.55 else ("square" if 0.85 <= ratio <= 1.15 else "post")

        small = im.resize((96, 96), Image.Resampling.BILINEAR)
        # Estimate background from corners
        corners = [
            small.getpixel((4, 4)),
            small.getpixel((91, 4)),
            small.getpixel((4, 91)),
            small.getpixel((91, 91)),
        ]
        bg_avg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))
        bg_lum = 0.299 * bg_avg[0] + 0.587 * bg_avg[1] + 0.114 * bg_avg[2]
        theme_mode = "dark" if bg_lum < 110 else "light"

        # Quantize to find vivid accent colors
        q = small.quantize(colors=10).convert("RGB")
        counts = sorted(q.getcolors(maxcolors=32) or [], reverse=True)
        vivid: list[str] = []
        for _, rgb in counts:
            sat = max(rgb) - min(rgb)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            if sat >= 32 and 45 <= lum <= 225:
                hx = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                if hx not in vivid:
                    vivid.append(hx)
        return {
            "detected_aspect": aspect,
            "theme_mode": theme_mode,
            "bg_hex": f"#{bg_avg[0]:02X}{bg_avg[1]:02X}{bg_avg[2]:02X}",
            "primary_hex": vivid[0] if vivid else "",
            "accent_hex": vivid[1] if len(vivid) > 1 else (vivid[0] if vivid else ""),
        }
    except Exception as exc:
        print(f"[WARN] Ref image Pillow inspect failed: {exc}")
        return {}


def resolve_design_profile(
    query_text: str,
    aspect_ratio: str = "post",
    design_spec_raw: str = "",
    bot_token: str = "",
) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    if design_spec_raw:
        try:
            parsed = json.loads(design_spec_raw)
            if isinstance(parsed, dict):
                spec = parsed
        except Exception:
            pass

    q = (query_text or "").lower()
    # Aspect ratio resolution: explicit query/input wins over ref aspect
    ar = (aspect_ratio or "").strip().lower()
    if any(k in q for k in ("استوری", "story", "1920", "۹:۱۶", "9:16")):
        ar = "story"
    elif any(k in q for k in ("مربع", "square", "۱:۱", "1:1")):
        ar = "square"
    elif any(k in q for k in ("پست", "post", "کاروسل", "1350", "۴:۵", "4:5")):
        ar = "post"
    elif ar not in {"post", "story", "square"}:
        ar = str(spec.get("canvas_aspect") or "post").lower()
        if ar not in {"post", "story", "square"}:
            ar = "post"

    dims = {"post": (1080, 1350), "story": (1080, 1920), "square": (1080, 1080)}[ar]

    # Pick named palette if mentioned
    palette_key = "bioinfo_teal"
    if any(k in q for k in ("ژل شفاف", "یاس", "بنفش")):
        palette_key = "transparent_gel"
    elif any(k in q for k in ("غروب قطبی", "مرجانی", "گلبهی")):
        palette_key = "polar_sunset"
    elif any(k in q for k in ("سفال", "آجری", "خردلی")):
        palette_key = "terracotta_mist"
    elif any(k in q for k in ("مرکبات", "لیمو", "کهربایی")):
        palette_key = "lab_citrus"
    elif any(k in q for k in ("مینیمال هوشمند", "کاغذی")):
        palette_key = "smart_minimal"
    elif any(k in q for k in ("پاستل", "صورتی", "دختر")):
        palette_key = "pastel_science"
    else:
        # Rotate cleanly by query hash across the top light palettes for freshness
        keys_pool = ["bioinfo_teal", "pastel_science", "transparent_gel", "polar_sunset", "lab_citrus", "terracotta_mist", "smart_minimal"]
        palette_key = keys_pool[sum(ord(c) for c in (query_text or "bio")) % len(keys_pool)]

    palette = dict(PALETTES[palette_key])
    theme_mode = str(spec.get("theme_mode") or "light").lower()
    has_grid = bool(spec.get("has_grid_bg", True))
    has_cta = bool(spec.get("has_cta_pill", ar == "story"))
    layout_pref = str(spec.get("layout_style") or "").strip().lower()
    visual_style_en = str(spec.get("visual_style_en") or "").strip()

    # If user attached a reference photo, verify/fill from Pillow if needed
    ref_file_id = str(spec.get("ref_file_id") or "").strip()
    if ref_file_id and bot_token and (not spec.get("bg_hex") or not spec.get("primary_hex")):
        px = _extract_ref_image_profile(ref_file_id, bot_token)
        for k in ("bg_hex", "primary_hex", "accent_hex", "theme_mode"):
            if px.get(k) and not spec.get(k):
                spec[k] = px[k]
        if not aspect_ratio and px.get("detected_aspect"):
            ar = px["detected_aspect"]
            dims = {"post": (1080, 1350), "story": (1080, 1920), "square": (1080, 1080)}[ar]

    # Apply extracted reference colors cleanly
    if spec.get("bg_hex") and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(spec["bg_hex"]).strip()):
        palette["bg"] = str(spec["bg_hex"]).strip()
    if spec.get("card_hex") and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(spec["card_hex"]).strip()):
        palette["card"] = str(spec["card_hex"]).strip()
    if spec.get("primary_hex") and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(spec["primary_hex"]).strip()):
        palette["primary"] = str(spec["primary_hex"]).strip()
    if spec.get("accent_hex") and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(spec["accent_hex"]).strip()):
        palette["accent"] = str(spec["accent_hex"]).strip()
    if spec.get("text_hex") and re.fullmatch(r"#[0-9A-Fa-f]{6}", str(spec["text_hex"]).strip()):
        palette["text"] = str(spec["text_hex"]).strip()

    if theme_mode == "dark":
        if not spec.get("bg_hex"):
            palette["bg"] = "#0B1224"
        if not spec.get("card_hex"):
            palette["card"] = "#141F3C"
        if not spec.get("text_hex"):
            palette["text"] = "#F1F5F9"
        palette["muted"] = "#CBD5E1"
        palette["blob1"] = palette["primary"]
        palette["blob2"] = palette["accent"]
    else:
        bg_rgb = _hex_to_rgb(palette["bg"], (248, 250, 250))
        p_rgb = _hex_to_rgb(palette["primary"], (13, 148, 136))
        a_rgb = _hex_to_rgb(palette["accent"], (232, 122, 66))
        b1 = _blend_rgb((255, 255, 255), p_rgb, 0.25)
        b2 = _blend_rgb((255, 255, 255), a_rgb, 0.25)
        palette["blob1"] = f"#{b1[0]:02X}{b1[1]:02X}{b1[2]:02X}"
        palette["blob2"] = f"#{b2[0]:02X}{b2[1]:02X}{b2[2]:02X}"
        # Guarantee dark readable text on light backgrounds
        bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        if bg_lum > 180 and not spec.get("text_hex"):
            palette["text"] = "#2E3438"

    return {
        "aspect": ar,
        "width": dims[0],
        "height": dims[1],
        "theme_mode": theme_mode,
        "has_grid": has_grid,
        "has_cta": has_cta,
        "layout_pref": layout_pref,
        "visual_style_en": visual_style_en,
        "palette": palette,
        "is_reference_clone": bool(spec),
    }


def generate_edu_carousel(
    user_query: str,
    topic_hint: str,
    slide_count: int,
    context_bundle: dict[str, Any],
    reply_context: str = "",
    design_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prof = design_profile or resolve_design_profile(user_query)
    is_story = prof["aspect"] == "story"
    style_note = (
        f"\nسبک بصری مرجع انتخابی کاربر: {prof['visual_style_en']}"
        if prof.get("visual_style_en")
        else ""
    )

    if slide_count == 1:
        structure_rules = (
            "1. **تعداد دقیق اسلایدها: ۱ اسلاید مستقل و کامل (`role: \"story_card\"`):**\n"
            "   - `headline`: تیتر اصلی کوتاه و کوبنده (حداکثر ۷ کلمه).\n"
            "   - `highlight_phrase`: ۱ تا ۳ کلمه کلیدی از تیتر (یا زیرتیتر رنگی) برای هایلایت رنگی دقیقاً مثل سبک استوری‌های حرفه‌ای آکادمی.\n"
            "   - `body_points`: **۳ تا ۵ بولت کوتاه** به فرمت `عنوان کوتاه: توضیح ۵ تا ۸ کلمه‌ای` (مثلاً `ردیابی ژن‌های بیماری‌زا: کشف کلید اصلی بیماری‌ها`).\n"
            "   - `cta_text`: ۱ جمله کوتاه دعوت به اقدام برای پایین کارت (مثلاً «برای دریافت آموزش کامل، کلمه R را ریپلای کنید» یا «این کارت رو برای هم‌رشته‌ای‌ات بفرست»).\n"
            "   - `tool_spotlight`: خالی `\"\"` بگذار مگر یک ابزار ضروری باشد.\n"
            "   - `visual_prompt`: توصیف انگلیسی یک **شیء یا مفهوم علمی سه‌بعدی شیشه‌ای/پاستلی مینیمال** دقیقاً مرتبط با موضوع."
        )
    else:
        structure_rules = (
            f"1. **تعداد دقیق اسلایدها: {slide_count} اسلاید (حتماً دقیقاً {slide_count} آبجکت در آرایه `slides` بساز!):**\n"
            "   - اسلاید ۱ (`role: \"cover\"`): تیتر کوتاه و کنجکاوکننده (حداکثر ۸ کلمه، غیرزرد) + `highlight_phrase` (۲ تا ۳ کلمه کلیدی تیتر برای رنگ تأکیدی) + ۱ زیرتیتر کوتاه در `body_points` + `cta_text: \"\"`.\n"
            f"   - اسلایدهای ۲ تا {slide_count - 1} (`role: \"content\"`): هر اسلاید یک تکهٔ کلیدی را با ۲ تا ۳ جمله یا بولت خیلی کوتاه (هر کدام ۶ تا ۱۲ کلمه) بگوید + `highlight_phrase` (عبارت کلیدی داخل متن برای هایلایت رنگی).\n"
            f"   - اسلاید آخر {slide_count} (`role: \"summary\"`): جمع‌بندی شفاف + ۲ نکته عملی + `cta_text` (دعوت به ذخیره/ارسال یا ریپلای)."
        )

    system_instruction = f"""تو طراح محتوای گرافیکی و استراتژیست علمی «آکادمی بیوانفورماتیک ایران» هستی.
مخاطب: کاربران باهوش و کم‌حوصله در ایران (دانشجویان زیست، پزشکی، بیوانفورماتیک و AI).

قوانین طلایی نگارش برای اسلاید ({'استوری ۹:۱۶' if is_story else 'پست ۴:۵'}):
{structure_rules}
2. **هیچ حس زردی و کلیک‌بیت نداشته باشد:** کلمات زرد («معجزه»، «باورنکردنی»، «انفجار») ممنوع! لحن علمیِ باوقار، واضح، موجز و خودمونی باشد.
3. **کمتر تخصصی و کاملاً مفهوم:** اصطلاحات سنگین انگلیسی را بدون توضیح نیاور؛ مفهوم را خیلی مستقیم و شفاف بنویس.
4. **ممنوعیت ایموجی در عکس:** داخل `headline`، `highlight_phrase`، `badge`، `body_points`، `cta_text` و `tool_spotlight` به هیچ وجه ایموجی نگذار تا فونت کلمه/پینار خراب نشود (ایموجی را فقط در `telegram_caption` بگذار).
5. `telegram_caption`: کپشن کوتاه و شیک (۳ تا ۴ خط + ۴ هشتگ تخصصی).{style_note}

خروجی فقط JSON معتبر مطابق اسکیما باشد."""

    source_note = f"\nمتن مرجع انتخابی کاربر برای تبدیل به اسلاید:\n{reply_context[:3500]}\n" if reply_context else ""
    user_prompt = (
        f"درخواست کاربر: {user_query or topic_hint}\n"
        f"تعداد دقیق اسلایدها: {slide_count}\n"
        f"فرمت ابعادی: {prof['aspect']} ({prof['width']}x{prof['height']})\n"
        f"{source_note}\n"
        f"منابع مرتبط برای دقت علمی (فقط در حد نیاز استفاده شود):\n"
        f"{json.dumps(context_bundle, ensure_ascii=False)}"
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "post_title": {"type": "STRING"},
            "slides": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "slide_number": {"type": "INTEGER"},
                        "role": {"type": "STRING"},
                        "badge": {"type": "STRING"},
                        "headline": {"type": "STRING"},
                        "highlight_phrase": {"type": "STRING"},
                        "body_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "tool_spotlight": {"type": "STRING"},
                        "cta_text": {"type": "STRING"},
                        "visual_prompt": {"type": "STRING"},
                    },
                    "required": [
                        "slide_number",
                        "role",
                        "badge",
                        "headline",
                        "highlight_phrase",
                        "body_points",
                        "tool_spotlight",
                        "cta_text",
                        "visual_prompt",
                    ],
                },
            },
            "telegram_caption": {"type": "STRING"},
            "selected_repos": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "repo": {"type": "STRING"},
                        "why_persian": {"type": "STRING"},
                    },
                    "required": ["repo", "why_persian"],
                },
            },
        },
        "required": ["post_title", "slides", "telegram_caption", "selected_repos"],
    }
    json_hint = (
        '\nReturn ONLY valid JSON: {"post_title": str, "slides": [{"slide_number": int, "role": "cover"|"content"|"summary"|"story_card", '
        '"badge": str, "headline": str, "highlight_phrase": str, "body_points": [str], "tool_spotlight": str, "cta_text": str, "visual_prompt": str}], '
        '"telegram_caption": str, "selected_repos": [{"repo": str, "why_persian": str}]}.'
    )
    return call_edu_json_with_failover(system_instruction, user_prompt, schema, json_hint)


def generate_cf_art(prompt: str, design_profile: dict[str, Any] | None = None) -> bytes | None:
    account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
    ai_token = os.getenv("CF_AI_TOKEN", "").strip()
    if not account_id or not ai_token or not prompt:
        return None
    prof = design_profile or {}
    theme = prof.get("theme_mode", "light")
    pal = prof.get("palette") or PALETTES["bioinfo_teal"]
    style_extra = prof.get("visual_style_en") or ""

    if theme == "dark":
        clean_p = (
            f"3D translucent glowing scientific illustration of {prompt.strip()}, "
            f"color accents {pal.get('primary', '#38BDF8')} and {pal.get('accent', '#10B981')}, "
            f"isolated on deep dark navy studio backdrop, {style_extra}, ultra clean, no text, no letters, no watermark"
        )[:1500]
    else:
        clean_p = (
            f"3D translucent frosted glass and smooth matte pastel scientific render of {prompt.strip()}, "
            f"color accents {pal.get('primary', '#0D9488')} and {pal.get('accent', '#E87A42')}, "
            f"centered isolated object on pure solid #FFFFFF white studio background, soft subtle drop shadow, "
            f"minimalist editorial medical visualization, {style_extra}, strictly no text, no letters, no labels, no watermark"
        )[:1500]
    try:
        r = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell",
            headers={"Authorization": f"Bearer {ai_token}", "Content-Type": "application/json"},
            json={"prompt": clean_p, "steps": 4},
            timeout=35,
        )
        if r.status_code == 200:
            b64 = r.json().get("result", {}).get("image")
            if b64:
                return base64.b64decode(b64)
    except Exception as exc:
        print(f"[WARN] CF art failed: {exc}")
    return None


def wrap_text_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = (text or "").strip().split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), shape_text(trial), font=font, direction="rtl" if HAS_RAQM else None)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def strip_emojis_for_card(text: str) -> str:
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]", "", text or "")
    # Replace em-dash / en-dash / ellipsis with ASCII-safe equivalents present in all Persian fonts (e.g. Pinar)
    cleaned = cleaned.replace("—", "-").replace("–", "-").replace("…", "...")
    return cleaned.strip()


def apply_academy_footer(canvas: Image.Image) -> Image.Image:
    if not OVERLAY_PATH.exists():
        return canvas
    overlay = Image.open(OVERLAY_PATH).convert("RGBA")
    # Crop bottom 185px band (DNA icon + bilingual brand name) from 1280x720 and scale cleanly to 1080px wide
    band = overlay.crop((0, 535, overlay.width, overlay.height))
    new_h = round(band.height * (canvas.width / band.width))
    band = band.resize((canvas.width, new_h), Image.Resampling.LANCZOS)
    rgba = canvas.convert("RGBA")
    rgba.alpha_composite(band, (0, canvas.height - new_h))
    return rgba.convert("RGB")


def _pick_font_path(candidates: list[Path], keywords: tuple[str, ...]) -> Path | None:
    for kw in keywords:
        for p in candidates:
            if kw in p.stem.lower():
                return p
    return candidates[0] if candidates else None


def _load_font_with_weight(path: Path, size: int, weight: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), size)
    try:
        axes = font.get_variation_axes()
        if axes:
            vals = []
            for ax in axes:
                tag = (ax.get("tag") or "").decode("ascii", "ignore") if isinstance(ax.get("tag"), bytes) else str(ax.get("tag") or "")
                name = (ax.get("name") or "").decode("ascii", "ignore") if isinstance(ax.get("name"), bytes) else str(ax.get("name") or "")
                if tag.lower() == "wght" or "weight" in name.lower():
                    v = max(ax["minimum"], min(ax["maximum"], weight))
                else:
                    v = ax["default"]
                vals.append(v)
            font.set_variation_by_axes(vals)
    except Exception:
        pass
    return font


def _is_variable_font(path: Path) -> bool:
    try:
        return bool(ImageFont.truetype(str(path), 24).get_variation_axes())
    except Exception:
        return "vf" in path.stem.lower() or "variable" in path.stem.lower()


def load_slide_fonts(role: str, query_hint: str = "") -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf")))
    custom_fonts = [p for p in all_fonts if not p.stem.lower().startswith("vazirmatn")]

    hint_l = (query_hint or "").lower()
    chosen_pool = custom_fonts
    if custom_fonts:
        if any(k in hint_l for k in ("پینار", "pinar")):
            pinar = [p for p in custom_fonts if "pinar" in p.stem.lower()]
            if pinar:
                chosen_pool = pinar
        elif any(k in hint_l for k in ("کلمه", "kalameh")):
            kalameh = [p for p in custom_fonts if "kalameh" in p.stem.lower()]
            if kalameh:
                chosen_pool = kalameh
        else:
            kalameh = [p for p in custom_fonts if "kalameh" in p.stem.lower()]
            pinar = [p for p in custom_fonts if "pinar" in p.stem.lower()]
            if kalameh and pinar:
                chosen_pool = kalameh + pinar

    eb_path = _pick_font_path(chosen_pool, ("black", "extrabold", "heavy", "900", "800", "bold", "700")) or (FONTS_DIR / "Vazirmatn-ExtraBold.ttf")
    b_path = _pick_font_path(chosen_pool, ("bold", "700", "semibold", "600", "medium", "500")) or (FONTS_DIR / "Vazirmatn-Bold.ttf")
    r_path = _pick_font_path(chosen_pool, ("regular", "400", "medium", "light", "normal")) or (FONTS_DIR / "Vazirmatn-Regular.ttf")

    # If user uploaded only a static bold display font (no regular weight), keep Vazirmatn-Regular for crisp small body text
    if chosen_pool and r_path == eb_path and not _is_variable_font(r_path) and not any(k in r_path.stem.lower() for k in ("regular", "400", "medium")):
        vaz_reg = FONTS_DIR / "Vazirmatn-Regular.ttf"
        if vaz_reg.exists():
            r_path = vaz_reg

    return (
        _load_font_with_weight(eb_path, 52 if role == "cover" else 42, 850),
        _load_font_with_weight(b_path, 30, 700),
        _load_font_with_weight(r_path, 28, 400),
        _load_font_with_weight(b_path, 23, 700),
    )


def _remove_white_backdrop(img: Image.Image) -> Image.Image:
    """Convert near-white studio background to smooth alpha so 3D renders float over the grid/card."""
    rgba = img.convert("RGBA")
    px = rgba.getdata()
    new_px = []
    for r, g, b, a in px:
        mn = min(r, g, b)
        mx = max(r, g, b)
        sat = mx - mn
        if mn >= 238 and sat <= 18:
            # Smooth transition from 238..252 -> alpha 255..0
            fade = max(0, min(255, int(255 * (252 - mn) / 14)))
            new_px.append((r, g, b, min(a, fade)))
        else:
            new_px.append((r, g, b, a))
    rgba.putdata(new_px)
    return rgba


def _draw_light_brand_background(W: int, H: int, prof: dict[str, Any]) -> Image.Image:
    pal = prof["palette"]
    theme = prof["theme_mode"]
    bg_rgb = _hex_to_rgb(pal["bg"], (248, 250, 250))
    canvas = Image.new("RGBA", (W, H), (*bg_rgb, 255))

    # Soft ambient pastel glows (corners + behind hero area)
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)
    b1 = _hex_to_rgb(pal.get("blob1") or pal["primary"])
    b2 = _hex_to_rgb(pal.get("blob2") or pal["accent"])
    alpha_g = 75 if theme == "dark" else 145
    gdraw.ellipse([(-int(W * 0.18), int(H * 0.08)), (int(W * 0.55), int(H * 0.48))], fill=(*b1, alpha_g))
    gdraw.ellipse([(int(W * 0.45), int(H * 0.55)), (int(W * 1.18), int(H * 0.96))], fill=(*b2, alpha_g))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=95))
    canvas = Image.alpha_composite(canvas, glow_layer)

    # Subtle architectural grid (like the academy reference posts & stories)
    if prof.get("has_grid", True):
        grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gr = ImageDraw.Draw(grid)
        step = 72
        line_col = (255, 255, 255, 16) if theme == "dark" else (140, 145, 155, 24)
        for x in range(36, W, step):
            gr.line([(x, 0), (x, H)], fill=line_col, width=1)
        for y in range(36, H, step):
            gr.line([(0, y), (W, y)], fill=line_col, width=1)
        canvas = Image.alpha_composite(canvas, grid)

    return canvas


def _draw_footer_and_dots(
    canvas: Image.Image,
    num: int,
    total_slides: int,
    prof: dict[str, Any],
    font_sm: ImageFont.FreeTypeFont,
) -> Image.Image:
    W, H = canvas.size
    pal = prof["palette"]
    theme = prof["theme_mode"]
    if theme == "dark":
        canvas_rgb = apply_academy_footer(canvas.convert("RGB"))
        return canvas_rgb

    draw = ImageDraw.Draw(canvas)
    kw_dir = {"direction": "rtl"} if HAS_RAQM else {}
    muted_rgb = _hex_to_rgb(pal.get("muted") or "#5E666B")
    prim_rgb = _hex_to_rgb(pal.get("primary") or "#0D9488")

    footer_y = H - 56
    # Subtle divider line
    draw.line([(56, footer_y - 26), (W - 56, footer_y - 26)], fill=(*muted_rgb, 40), width=1)

    # Left: clean academy bilingual label
    draw.text((60, footer_y), "@ir_bioinformtic_academy", font=font_sm, fill=(*muted_rgb, 210), anchor="lm")
    draw.text((W - 60, footer_y), shape_text("آکادمی بیوانفورماتیک ایران"), font=font_sm, fill=(*muted_rgb, 225), anchor="rm", **kw_dir)

    # Center: minimal slide progress dots when > 1 slide
    if total_slides > 1:
        dots_n = min(total_slides, 12)
        dot_gap = 18
        total_w = (dots_n - 1) * dot_gap
        start_x = (W - total_w) // 2
        active_idx = min(dots_n - 1, num - 1)
        for i in range(dots_n):
            cx = start_x + i * dot_gap
            r = 6 if i == active_idx else 4
            col = (*prim_rgb, 230) if i == active_idx else (*muted_rgb, 80)
            draw.ellipse([(cx - r, footer_y - r), (cx + r, footer_y + r)], fill=col)

    return canvas.convert("RGB")


def _paste_3d_visual(
    canvas: Image.Image,
    art_bytes: bytes | None,
    box: tuple[int, int, int, int],
    theme_mode: str,
) -> Image.Image:
    if not art_bytes:
        return canvas
    try:
        raw = Image.open(io.BytesIO(art_bytes)).convert("RGBA")
        bx, by, bw, bh = box
        if theme_mode != "dark":
            raw = _remove_white_backdrop(raw)
            fitted = ImageOps.contain(raw, (bw, bh), method=Image.Resampling.LANCZOS)
            px = bx + (bw - fitted.width) // 2
            py = by + (bh - fitted.height) // 2
            canvas.alpha_composite(fitted, (px, py))
        else:
            fitted = ImageOps.fit(raw, (bw, bh), method=Image.Resampling.LANCZOS)
            mask = Image.new("L", (bw, bh), 0)
            ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (bw, bh)], radius=36, fill=245)
            canvas.paste(fitted, (bx, by), mask)
    except Exception:
        pass
    return canvas


def render_carousel_slide(
    slide: dict[str, Any],
    total_slides: int,
    art_bytes: bytes | None,
    font_hint: str = "",
    design_profile: dict[str, Any] | None = None,
) -> bytes:
    prof = design_profile or resolve_design_profile(font_hint)
    W, H = int(prof["width"]), int(prof["height"])
    pal = prof["palette"]
    theme = prof["theme_mode"]
    role = (slide.get("role") or "content").lower()
    num = int(slide.get("slide_number") or 1)
    is_story = H >= 1700

    prim_rgb = _hex_to_rgb(pal["primary"])
    acc_rgb = _hex_to_rgb(pal["accent"])
    text_rgb = _hex_to_rgb(pal["text"])
    muted_rgb = _hex_to_rgb(pal.get("muted") or "#5E666B")
    card_rgb = _hex_to_rgb(pal.get("card") or "#FFFFFF")

    font_eb, font_b, font_r, font_sm = load_slide_fonts("cover" if role in {"cover", "story_card"} else "content", font_hint)
    if is_story:
        # Scale headings slightly larger for tall 1080x1920 story canvas
        font_eb = ImageFont.truetype(font_eb.path, int(font_eb.size * 1.08))
        font_b = ImageFont.truetype(font_b.path, int(font_b.size * 1.05))
        font_r = ImageFont.truetype(font_r.path, int(font_r.size * 1.05))

    canvas = _draw_light_brand_background(W, H, prof)
    draw = ImageDraw.Draw(canvas, "RGBA")
    kw_dir = {"direction": "rtl"} if HAS_RAQM else {}

    headline = strip_emojis_for_card(slide.get("headline") or "")
    hl_phrase = strip_emojis_for_card(slide.get("highlight_phrase") or "")
    badge_txt = strip_emojis_for_card(slide.get("badge") or "")
    pts = [strip_emojis_for_card(p) for p in (slide.get("body_points") or []) if strip_emojis_for_card(p)]
    cta_text = strip_emojis_for_card(slide.get("cta_text") or "")
    spotlight = strip_emojis_for_card(slide.get("tool_spotlight") or "")

    # Choose layout cleanly: story_card, cover_editorial, content_split, content_centered, summary_cta
    layout = prof.get("layout_pref") or ""
    if role == "story_card" or (total_slides == 1 and is_story):
        layout = "story_card"
    elif role == "cover":
        layout = "story_card" if is_story else "cover_editorial"
    elif role == "summary":
        layout = "summary_cta"
    elif layout not in {"content_split", "content_centered", "arch_card"}:
        layout = "content_split" if (num % 2 == 1 and not is_story) else "content_centered"

    card_fill = (*card_rgb, 228 if theme == "dark" else 222)
    card_outline = (255, 255, 255, 40) if theme == "dark" else (*prim_rgb, 65)

    def draw_translucent_card(coords: list[tuple[int, int]], radius: int) -> None:
        nonlocal canvas, draw
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle(coords, radius=radius, fill=card_fill, outline=card_outline, width=2)
        canvas = Image.alpha_composite(canvas, ov)
        draw = ImageDraw.Draw(canvas)

    # Top subtle header brand mark (for stories/covers matching Biocan/IRBIOINF style)
    if is_story or role in {"cover", "story_card"}:
        draw.text((W // 2, 52), "IRBIOINF  •  ACADEMY", font=font_sm, fill=(*muted_rgb, 210), anchor="mm")

    if layout == "story_card":
        # Inspired by R story & اقساطی story: soft translucent rounded card + overlapping 3D visual + colored title + itemized points + bottom CTA pill
        card_top = int(H * 0.21)
        card_bot = H - (225 if cta_text else 125)
        draw_translucent_card([(68, card_top), (W - 68, card_bot)], radius=56)

        # Overlapping 3D visual at top center of card
        vis_size = int(W * 0.44)
        canvas = _paste_3d_visual(canvas, art_bytes, ((W - vis_size) // 2, int(H * 0.075), vis_size, vis_size), theme)
        draw = ImageDraw.Draw(canvas)

        y_cur = int(H * 0.33)
        head_lines = wrap_text_lines(draw, headline, font_eb, W - 170)[:2]
        for idx_l, line in enumerate(head_lines):
            col = (*prim_rgb, 255) if (idx_l == len(head_lines) - 1 and len(head_lines) > 1) or (hl_phrase and hl_phrase in line) else (*text_rgb, 255)
            draw.text((W // 2, y_cur), shape_text(line), font=font_eb, fill=col, anchor="ma", **kw_dir)
            y_cur += int(font_eb.size * 1.44)

        if hl_phrase and hl_phrase not in headline:
            draw.text((W // 2, y_cur + 6), shape_text(hl_phrase), font=font_b, fill=(*acc_rgb, 255), anchor="ma", **kw_dir)
            y_cur += int(font_b.size * 1.60)

        # Thin divider
        y_cur += 20
        div_col = _blend_rgb(card_rgb, prim_rgb, 0.35)
        draw.line([(160, y_cur), (W - 160, y_cur)], fill=(*div_col, 255), width=2)
        y_cur += 44

        # Distribute points evenly inside remaining card height
        max_pts = 5 if is_story else 3
        shown_pts = pts[:max_pts]
        avail_h = max(240, (card_bot - 50) - y_cur)
        est_item_h = 115 if any(":" in p or "：" in p for p in shown_pts) else 85
        step_gap = max(28, min(72, (avail_h - len(shown_pts) * est_item_h) // max(1, len(shown_pts))))
        for idx_p, pt in enumerate(shown_pts):
            if y_cur > card_bot - 85:
                break
            if ":" in pt or "：" in pt:
                sep = "：" if "：" in pt else ":"
                lbl, desc = pt.split(sep, 1)
                lbl_col = (*prim_rgb, 255) if idx_p % 2 == 0 else (*acc_rgb, 255)
                draw.text((W // 2, y_cur), shape_text(f"{lbl.strip()}:"), font=font_b, fill=lbl_col, anchor="ma", **kw_dir)
                y_cur += int(font_b.size * 1.44)
                for dl in wrap_text_lines(draw, desc.strip(), font_r, W - 170)[:2]:
                    draw.text((W // 2, y_cur), shape_text(dl), font=font_r, fill=(*text_rgb, 255), anchor="ma", **kw_dir)
                    y_cur += int(font_r.size * 1.48)
                y_cur += step_gap
            else:
                for dl in wrap_text_lines(draw, pt, font_r, W - 170)[:2]:
                    draw.text((W // 2, y_cur), shape_text(dl), font=font_r, fill=(*text_rgb, 255), anchor="ma", **kw_dir)
                    y_cur += int(font_r.size * 1.48)
                y_cur += step_gap

        if cta_text:
            pill_y1 = card_bot + 28
            pill_y2 = pill_y1 + 78
            draw.rounded_rectangle([(84, pill_y1), (W - 84, pill_y2)], radius=39, fill=(*acc_rgb, 255))
            cta_lines = wrap_text_lines(draw, cta_text, font_sm, W - 220)[:1]
            draw.text((W // 2, (pill_y1 + pill_y2) // 2), shape_text(cta_lines[0]), font=font_sm, fill=(255, 255, 255, 255), anchor="mm", **kw_dir)

    elif layout == "cover_editorial":
        # Inspired by "انسان عظمتی به اندازه تمامی کهکشان‌ها": bold right-aligned Kalameh headline + 3D render on left
        canvas = _paste_3d_visual(canvas, art_bytes, (40, int(H * 0.14), int(W * 0.48), int(H * 0.68)), theme)
        draw = ImageDraw.Draw(canvas)

        y_cur = int(H * 0.19)
        head_lines = wrap_text_lines(draw, headline, font_eb, int(W * 0.48))[:5]
        for idx_l, line in enumerate(head_lines):
            use_col = (*prim_rgb, 255) if (idx_l > 0 or (hl_phrase and hl_phrase in line)) else (*text_rgb, 255)
            draw.text((W - 68, y_cur), shape_text(line), font=font_eb, fill=use_col, anchor="ra", **kw_dir)
            y_cur += int(font_eb.size * 1.48)

        y_cur += 24
        draw.text((W - 68, y_cur), "•  •  •", font=font_b, fill=(*muted_rgb, 200), anchor="ra")
        y_cur += 54

        if hl_phrase and hl_phrase not in headline:
            draw.text((W - 68, y_cur), shape_text(hl_phrase), font=font_b, fill=(*acc_rgb, 255), anchor="ra", **kw_dir)
            y_cur += int(font_b.size * 1.48)

        for pt in pts[:2]:
            for dl in wrap_text_lines(draw, pt, font_r, int(W * 0.46))[:3]:
                draw.text((W - 68, y_cur), shape_text(dl), font=font_r, fill=(*text_rgb, 255), anchor="ra", **kw_dir)
                y_cur += int(font_r.size * 1.48)
            y_cur += 14

    elif layout == "content_split":
        # Inspired by "ژنومیکس Genomics" & "کروموزوم X": pill badge/title + crisp text on right, 3D artwork on left
        draw_translucent_card([(46, 78), (W - 46, H - 108)], radius=48)
        canvas = _paste_3d_visual(canvas, art_bytes, (64, int(H * 0.24), int(W * 0.42), int(H * 0.56)), theme)
        draw = ImageDraw.Draw(canvas)

        # Top right pill headline or badge
        pill_txt = badge_txt or headline
        if len(pill_txt) <= 22:
            pill_w = min(int(W * 0.50), max(280, len(pill_txt) * 24 + 90))
            draw.rounded_rectangle([(W - 76 - pill_w, 132), (W - 76, 208)], radius=38, fill=(*prim_rgb, 255))
            draw.text((W - 76 - pill_w // 2, 170), shape_text(pill_txt), font=font_b, fill=(255, 255, 255, 255), anchor="mm", **kw_dir)
            y_cur = 240
            if pill_txt != headline:
                for hl in wrap_text_lines(draw, headline, font_eb, int(W * 0.48))[:3]:
                    draw.text((W - 78, y_cur), shape_text(hl), font=font_eb, fill=(*text_rgb, 255), anchor="ra", **kw_dir)
                    y_cur += int(font_eb.size * 1.38)
                y_cur += 18
        else:
            y_cur = 148
            for hl in wrap_text_lines(draw, headline, font_eb, int(W * 0.52))[:3]:
                draw.text((W - 78, y_cur), shape_text(hl), font=font_eb, fill=(*text_rgb, 255), anchor="ra", **kw_dir)
                y_cur += int(font_eb.size * 1.38)
            y_cur += 22

        for idx_p, pt in enumerate(pts[:3]):
            if y_cur > H - 230:
                break
            lines = wrap_text_lines(draw, pt, font_r, int(W * 0.45))[:4]
            col = (*prim_rgb, 255) if (hl_phrase and hl_phrase in pt) else (*text_rgb, 255)
            for l in lines:
                draw.text((W - 78, y_cur), shape_text(l), font=font_r, fill=col, anchor="ra", **kw_dir)
                y_cur += int(font_r.size * 1.50)
            y_cur += 20

        if hl_phrase and not any(hl_phrase in p for p in pts[:3]) and y_cur < H - 210:
            draw.text((W - 78, y_cur + 6), shape_text(hl_phrase), font=font_b, fill=(*acc_rgb, 255), anchor="ra", **kw_dir)

    else:
        # "content_centered" or "summary_cta": Inspired by "سیستم ایمنی قوی‌تر" & "سفری تا خورشید"
        draw_translucent_card([(52, 74), (W - 52, H - 108)], radius=52)

        y_cur = 118
        head_lines = wrap_text_lines(draw, headline, font_eb, W - 150)[:2]
        for hl in head_lines:
            draw.text((W // 2, y_cur), shape_text(hl), font=font_eb, fill=(*text_rgb, 255), anchor="ma", **kw_dir)
            y_cur += int(font_eb.size * 1.36)

        # Rounded soft pastel stage for 3D visual
        stage_top = y_cur + 18
        stage_h = int(H * (0.34 if is_story else 0.36))
        stage_bg = _blend_rgb(card_rgb, prim_rgb, 0.10) if theme != "dark" else _blend_rgb(card_rgb, prim_rgb, 0.18)
        draw.rounded_rectangle([(120, stage_top), (W - 120, stage_top + stage_h)], radius=48, fill=(*stage_bg, 255))
        canvas = _paste_3d_visual(canvas, art_bytes, (140, stage_top + 12, W - 280, stage_h - 24), theme)
        draw = ImageDraw.Draw(canvas)

        y_cur = stage_top + stage_h + 38
        for idx_p, pt in enumerate(pts[:3]):
            if y_cur > H - 220:
                break
            is_hl = bool(hl_phrase and hl_phrase in pt) or (role == "summary" and idx_p == 0)
            wrapped = wrap_text_lines(draw, pt, font_r, W - 140)[:3]
            if is_hl and len(wrapped) == 1:
                pill_h = int(font_r.size * 1.65)
                draw.rounded_rectangle([(110, y_cur - 6), (W - 110, y_cur + pill_h)], radius=18, fill=(*prim_rgb, 255))
                draw.text((W // 2, y_cur + pill_h // 2 - 4), shape_text(wrapped[0]), font=font_b, fill=(255, 255, 255, 255), anchor="mm", **kw_dir)
                y_cur += pill_h + 18
            else:
                col = (*acc_rgb, 255) if is_hl else (*text_rgb, 255)
                for l in wrapped:
                    draw.text((W // 2, y_cur), shape_text(l), font=font_r, fill=col, anchor="ma", **kw_dir)
                    y_cur += int(font_r.size * 1.50)
                y_cur += 16

        if cta_text:
            pill_y1 = H - 205
            draw.rounded_rectangle([(96, pill_y1), (W - 96, pill_y1 + 72)], radius=36, fill=(*acc_rgb, 255))
            cta_l = wrap_text_lines(draw, cta_text, font_sm, W - 240)[0]
            draw.text((W // 2, pill_y1 + 36), shape_text(cta_l), font=font_sm, fill=(255, 255, 255, 255), anchor="mm", **kw_dir)

    if spotlight and layout != "story_card":
        draw = ImageDraw.Draw(canvas)
        sp_y = H - 185 if not cta_text else H - 270
        sp_bg = _blend_rgb(card_rgb, prim_rgb, 0.12)
        sp_border = _blend_rgb(card_rgb, prim_rgb, 0.45)
        draw.rounded_rectangle([(84, sp_y), (W - 84, sp_y + 52)], radius=20, fill=(*sp_bg, 255), outline=(*sp_border, 255), width=2)
        draw.text((W // 2, sp_y + 26), shape_text(f"ابزار گیت‌هاب: {spotlight[:55]}"), font=font_sm, fill=(*prim_rgb, 255), anchor="mm", **kw_dir)

    final_rgb = _draw_footer_and_dots(canvas, num, total_slides, prof, font_sm)
    buf = io.BytesIO()
    final_rgb.save(buf, format="JPEG", quality=93, optimize=True)
    return buf.getvalue()


def send_tg_album(bot_token: str, chat_id: str, slide_images: list[bytes]) -> None:
    if not slide_images:
        return
    if len(slide_images) == 1:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendPhoto",
            data={"chat_id": chat_id},
            files={"photo": ("slide_1.jpg", slide_images[0], "image/jpeg")},
            timeout=30,
        )
        return

    for chunk_start in range(0, len(slide_images), 10):
        chunk = slide_images[chunk_start : chunk_start + 10]
        if len(chunk) == 1:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                data={"chat_id": chat_id},
                files={"photo": (f"slide_{chunk_start + 1}.jpg", chunk[0], "image/jpeg")},
                timeout=30,
            )
            continue
        media = []
        files = {}
        for idx, img_bytes in enumerate(chunk, 1):
            key = f"slide{idx}"
            media.append({"type": "photo", "media": f"attach://{key}"})
            files[key] = (f"slide_{chunk_start + idx}.jpg", img_bytes, "image/jpeg")

        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
            data={"chat_id": chat_id, "media": json.dumps(media)},
            files=files,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"[WARN] sendMediaGroup HTTP {resp.status_code}: {resp.text[:250]}")


def _clean_tg_html(text: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", text or "", flags=re.I)
    s = re.sub(r"</p\s*>", "\n\n", s, flags=re.I)
    s = re.sub(r"<p\s*>", "", s, flags=re.I)
    # Strip any unsupported HTML tags except Telegram-allowed ones
    s = re.sub(r"<(?!/?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|a)(?:\s|>))[^>]+>", "", s, flags=re.I)
    return s.strip()


def send_tg_text(bot_token: str, chat_id: str, html_text: str) -> None:
    cleaned = _clean_tg_html(html_text)
    for chunk_start in range(0, len(cleaned), 3900):
        chunk = cleaned[chunk_start : chunk_start + 3900]
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if r.status_code != 200:
            print(f"[WARN] sendMessage HTML HTTP {r.status_code}: {r.text[:200]} — retrying plain text")
            plain = re.sub(r"<[^>]+>", "", chunk)
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": plain, "disable_web_page_preview": True},
                timeout=20,
            )


def format_editable_slides_text(carousel: dict[str, Any]) -> str:
    lines = [f"🎓 <b>{html.escape(carousel.get('post_title') or 'پست آموزشی بیوانفورماتیک')}</b>\n"]
    for s in carousel.get("slides") or []:
        n = to_fa_digits(s.get("slide_number") or 1)
        lines.append(f"📌 <b>اسلاید {n}: {html.escape(str(s.get('headline') or ''))}</b>")
        pts = s.get("body_points") or []
        if isinstance(pts, str):
            pts = [pts]
        for bp in pts:
            lines.append(f"• {html.escape(str(bp))}")
        if s.get("tool_spotlight"):
            lines.append(f"🛠 <i>ابزار: {html.escape(str(s['tool_spotlight']))}</i>")
        lines.append("")

    cap = (carousel.get("telegram_caption") or "").strip()
    if cap:
        lines.append("📝 <b>کپشن آماده انتشار:</b>")
        lines.append(cap)
    lines.append(
        "\n⭕️ معین شیرین کام\n"
        "آکادمی بیوانفورماتیک ایران 🖥⭐️\n"
        "@ir_bioinformtic_academy"
    )
    return "\n".join(lines)


def format_library_links_message(carousel: dict[str, Any], matched: dict[str, Any]) -> str:
    lines = ["📚 <b>منابع واقعی انتخاب‌شده از کتابخانه ۱۵۰ ریپوی آکادمی:</b>\n"]
    selected = carousel.get("selected_repos") or []
    seen_slugs = set()
    for item in selected[:8]:
        slug = (item.get("repo") or "").strip()
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        url = f"https://github.com/{slug}"
        lines.append(f'🔹 <a href="{html.escape(url, quote=True)}"><b>{html.escape(slug)}</b></a> — {html.escape(item.get("why_persian") or "")}')

    if len(seen_slugs) < 4:
        for r in matched.get("repos", [])[:5]:
            slug = r.get("repo") or ""
            if slug and slug not in seen_slugs:
                seen_slugs.add(slug)
                lines.append(f'🔹 <a href="https://github.com/{html.escape(slug, quote=True)}"><b>{html.escape(slug)}</b></a> — {html.escape(r.get("desc") or "")}')

    topics = matched.get("topics") or []
    if topics:
        t_links = " | ".join(
            f'<a href="{html.escape(t["url"], quote=True)}">#{html.escape(t["topic"])}</a>'
            for t in topics[:4]
        )
        lines.append(f"\n🏷 <b>جست‌وجوی زنده در GitHub Topics:</b>\n{t_links}")

    iran = matched.get("iran_searches") or []
    if iran:
        lines.append("\n🇮🇷 <b>جست‌وجوی منابع فارسی و ایرانی مرتبط:</b>")
        for ir in iran[:3]:
            lines.append(f'• <a href="{html.escape(ir["url"], quote=True)}">{html.escape(ir["title"])}</a>')

    return "\n".join(lines)


def generate_edu_guide(
    user_query: str,
    topic_hint: str,
    item_count: int,
    context_bundle: dict[str, Any],
    channel: str = "telegram",
) -> dict[str, Any]:
    """Generate crisp, short, itemized, unambiguous Persian educational content (with optional 4-angle mode only when requested)."""
    is_insta = channel == "instagram"
    wants_all_angles = bool(re.search(r"(۴\s*مدل|چهار\s*مدل|۴\s*زاویه|همه\s*مدل|وایرال)", user_query or "", re.I))
    channel_rule = (
        "قالب «اینستاگرامی»: فوق‌العاده موجز و اسکن‌پذیر! هر خط حداکثر ۱۰–۱۲ کلمه، بدون حاشیه، + در پایان یک «📝 کپشن کوتاه اینستاگرام» (نهایتاً ۳ خط + هشتگ)."
        if is_insta
        else "قالب «تلگرامی» (پیش‌فرض): کوتاه، کوبنده، موردی (۱، ۲، ۳...)، کاملاً مفهوم و بدون ابهام."
    )
    angle_rule = (
        "کاربر صریحاً چند مدل خواسته است؛ موضوع را در ۴ زاویه کوتاه (چک‌لیست، باور غلط vs واقعیت، نقشه راه، مقایسه) بنویس."
        if wants_all_angles
        else "یک ساختار **یکدست، کوتاه، موردی و مستقیم** بنویس (بدون تقسیم اجباری به ۴ مدل مختلف!)."
    )

    system_instruction = f"""تو استراتژیست و مدرس ارشد «آکادمی بیوانفورماتیک ایران» هستی.
مخاطب تو در ایران امروز حوصلهٔ انشا، داستان‌سرایی دور از اصل موضوع، و اصطلاحات سخت دانشگاهی را ندارد؛ در عین حال به محتوای زرد و پوچ هم بی‌اعتماد است.

اصول قطعی نگارش:
1. **خیلی کوتاه، کوبنده، موردی، واضح و کاملاً مفهوم:** بدون مقدمه‌چینی اضافی؛ ۱–۲ خط معرفیِ شفافِ کانسپت ← ۳ تا ۴ مورد دقیق و شماره‌دار (`🔹 <b>۱. ...</b>`) ← ۱ خط جمع‌بندی روشن (`💡 <b>خلاصه:</b> ...`).
2. **ضد زرد و ضد کلیشه:** کلمات زرد («معجزه»، «باورنکردنی»، «انفجار») و انگلیسی بی‌ربط اکیداً ممنوع! لحن علمیِ تمیز، دقیق و خودمونیِ باکلاس باشد.
3. **به زبان ساده:** اصطلاحات سخت و ترجمه‌نشده نیاور؛ اگر یک کلمه تخصصی لازم بود، همان لحظه به زبان عادی روشنش کن.
4. {angle_rule}
5. {channel_rule}
6. **ایموجی به‌جا:** از ایموجی‌های تمیز (`🧬`, `🔹`, `💡`, `⚡️`, `🎯`) برای خوانایی استفاده کن و فقط تگ‌های مجاز تلگرام (`<b>`, `<i>`, `<u>`, `<blockquote>`, `<code>`) را به کار ببر.

خروجی فقط JSON معتبر مطابق اسکیما باشد."""

    user_prompt = (
        f"درخواست دقیق کاربر: {user_query or topic_hint}\n"
        f"موضوع هدف: {topic_hint}\n"
        f"کانال هدف: {channel}\n"
        f"تعداد درخواستی (در صورت نیاز به گام/مورد): {item_count}\n\n"
        f"منابع علمی پشتیبان از کتابخانه آکادمی (صرفاً برای دقت علمی، لازم نیست همه را لیست کنی):\n"
        f"{json.dumps(context_bundle, ensure_ascii=False)}"
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "body_html": {"type": "STRING"},
            "selected_repos": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "repo": {"type": "STRING"},
                        "why_persian": {"type": "STRING"},
                    },
                    "required": ["repo", "why_persian"],
                },
            },
        },
        "required": ["title", "body_html", "selected_repos"],
    }
    json_hint = (
        '\nReturn ONLY valid JSON: {"title": str, "body_html": str, '
        '"selected_repos": [{"repo": str, "why_persian": str}]}.'
    )
    return call_edu_json_with_failover(system_instruction, user_prompt, schema, json_hint)


def revise_reply_message(reply_context: str, user_instruction: str) -> dict[str, Any]:
    """Strictly revise/edit the exact message the user replied to, following their exact correction."""
    system_instruction = """تو دستیار ویراستاری و محتوای «آکادمی بیوانفورماتیک ایران» هستی.
کاربر روی یک پیامِ مشخص ریپلای زده و دقیقاً از تو خواسته همان پیام را اصلاح یا بازنویسی کنی.

قوانین اکید:
1. **فقط و فقط روی همان متن مرجع (`reply_context`) کار کن** — موضوع را عوض نکن و چیز بی‌ربطی تولید نکن!
2. **دقیقاً اصلاحیهٔ درخواستی کاربر را اعمال کن** (مثلاً کوتاه‌تر کردن، خودمونی‌تر کردن، حذف/اضافه بخش، تبدیل به اسلاید اینستا، اضافه کردن ایموجی، تغییر تیتر، ساده‌تر کردن اصطلاحات، و...).
3. لحن باید حرفه‌ای، دقیق، غیرزرد، شفاف و با ایموجی‌های به‌جا باشد.
4. فقط تگ‌های مجاز HTML تلگرام (`<b>`, `<i>`, `<u>`, `<blockquote>`, `<code>`) را به کار ببر.

خروجی فقط JSON معتبر مطابق اسکیما باشد."""

    user_prompt = (
        f"متن پیامی که کاربر روی آن ریپلای زده:\n"
        f"\"\"\"\n{reply_context}\n\"\"\"\n\n"
        f"دستور اصلاحی کاربر:\n\"{user_instruction}\""
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "body_html": {"type": "STRING"},
        },
        "required": ["title", "body_html"],
    }
    json_hint = '\nReturn ONLY valid JSON: {"title": str, "body_html": str}.'
    return call_edu_json_with_failover(system_instruction, user_prompt, schema, json_hint)


def format_guide_message(guide: dict[str, Any]) -> str:
    title = (guide.get("title") or "آموزش تخصصی بیوانفورماتیک").strip()
    body = (guide.get("body_html") or "").strip()
    return (
        f"🎓 <b>{title}</b>\n\n"
        f"{body}\n\n"
        "⭕️ معین شیرین کام\n"
        "آکادمی بیوانفورماتیک ایران 🖥⭐️\n"
        "@ir_bioinformtic_academy"
    )


def format_edited_message(edited: dict[str, Any]) -> str:
    title = (edited.get("title") or "").strip()
    body = (edited.get("body_html") or "").strip()
    header = f"✏️ <b>{title}</b>\n\n" if title and not body.startswith(f"<b>{title}") else ""
    footer = (
        "\n\n⭕️ معین شیرین کام\n"
        "آکادمی بیوانفورماتیک ایران 🖥⭐️\n"
        "@ir_bioinformtic_academy"
    )
    if "@ir_bioinformtic_academy" in body:
        footer = ""
    return f"{header}{body}{footer}"


def run_edu_studio(
    custom_topic: str,
    custom_query: str,
    slide_count: int,
    bot_token: str,
    target_chat_ids: list[str],
    status_msg_id: str,
    dry_run: bool,
    progress_fn: Any,
    with_images: bool = False,
    reply_context: str = "",
    edit_mode: bool = False,
    aspect_ratio: str = "post",
    design_spec_raw: str = "",
) -> None:
    t0 = time.time()

    # 1) Direct Reply-to-Edit mode: modify the exact replied message cleanly and return
    if edit_mode and reply_context:
        if not dry_run:
            progress_fn(bot_token, target_chat_ids, status_msg_id, 55, "در حال اعمال دقیق اصلاحیه روی پیام انتخابی شما...")
        edited = revise_reply_message(reply_context, custom_query)
        edited_msg = format_edited_message(edited)
        if dry_run:
            print(f"[DRY_RUN] Reply Edit mode completed in {time.time() - t0:.1f}s")
            print(edited_msg)
            return
        progress_fn(bot_token, target_chat_ids, status_msg_id, 92, "اصلاحیه آماده شد؛ در حال ارسال به تلگرام...")
        for cid in target_chat_ids:
            send_tg_text(bot_token, cid, edited_msg)
        progress_fn(bot_token, target_chat_ids, status_msg_id, 100, "اصلاحیه دقیق روی پیام شما اعمال و ارسال شد!", done=True)
        return

    lib = load_library()
    all_cats = sorted({r.get("cat", "") for r in lib.get("repos", []) if r.get("cat")})

    sources_only = (custom_query or "").startswith("__SOURCES_ONLY__")
    clean_query = (custom_query or "").replace("__SOURCES_ONLY__", "", 1).strip()
    wants_sources = sources_only or bool(re.search(r"(منابع|منبع|ریپو|لینک\s*گیت[‌ ]?هاب|با\s*منابع|with\s*sources)", clean_query, re.I))

    search_seed = reply_context[:500] if (sources_only and reply_context) else (clean_query or custom_topic)

    if not dry_run:
        progress_fn(bot_token, target_chat_ids, status_msg_id, 35, "در حال جست‌وجو بین ۱۵۰ ریپوی گیت‌هاب و کتابخانه آکادمی...")

    plan = plan_search_for_edu(search_seed, custom_topic, all_cats)
    matched = score_library_items(
        lib,
        search_terms=(plan.get("search_keywords") or []) + re.split(r"[\s,،]+", custom_topic or ""),
        target_cats=plan.get("target_categories") or [],
    )

    if sources_only:
        links_msg = format_library_links_message({}, matched)
        if dry_run:
            print(f"[DRY_RUN] Sources-only mode completed in {time.time() - t0:.1f}s")
            print(links_msg)
            return
        progress_fn(bot_token, target_chat_ids, status_msg_id, 92, "منابع و ریپوهای مرتبط استخراج شدند؛ در حال ارسال...")
        for cid in target_chat_ids:
            send_tg_text(bot_token, cid, links_msg)
        progress_fn(bot_token, target_chat_ids, status_msg_id, 100, "منابع واقعی از کتابخانه آکادمی ارسال شد!", done=True)
        return

    with ThreadPoolExecutor(max_workers=5) as ex:
        enriched_repos = list(ex.map(fetch_github_repo_meta, [r["repo"] for r in matched["repos"][:4]]))

    context_bundle = {
        "matched_curated_repos": matched["repos"][:7],
        "live_github_metadata": [
            {"repo": e["repo"], "stars": e.get("stars"), "desc": (e.get("description") or "")[:120]}
            for e in enriched_repos
        ],
        "parent_awesome_lists": matched["parent_directories"][:2],
        "github_topics": matched["topics"][:3],
    }

    is_instagram = bool(re.search(r"(اینستا|اینستاگرام|اسلاید|کاروسل|instagram|slide|carousel)", clean_query, re.I))
    channel = "instagram" if (is_instagram or with_images) else "telegram"

    if not with_images:
        if not dry_run:
            progress_fn(bot_token, target_chat_ids, status_msg_id, 55, "در حال نگارش دقیق آموزش با ۴ زاویه استراتژیک (بدون تصویر)...")

        guide = generate_edu_guide(clean_query, custom_topic, slide_count, context_bundle, channel=channel)
        guide_msg = format_guide_message(guide)
        links_msg = format_library_links_message(guide, matched) if wants_sources else ""

        if dry_run:
            print(f"[DRY_RUN] Edu Guide mode ({channel}, NO images) completed in {time.time() - t0:.1f}s")
            print(guide_msg)
            if links_msg:
                print("\n" + links_msg)
            return

        progress_fn(bot_token, target_chat_ids, status_msg_id, 92, "محتوای آموزشی آماده شد؛ در حال ارسال به تلگرام...")
        for cid in target_chat_ids:
            send_tg_text(bot_token, cid, guide_msg)
            if links_msg:
                send_tg_text(bot_token, cid, links_msg)
        progress_fn(bot_token, target_chat_ids, status_msg_id, 100, "محتوای آموزشی آماده و ارسال شد!", done=True)
        return

    if not dry_run:
        progress_fn(bot_token, target_chat_ids, status_msg_id, 55, f"در حال طراحی و نگارش محتوای {to_fa_digits(slide_count)} اسلاید آموزشی...")

    design_profile = resolve_design_profile(
        clean_query or custom_topic,
        aspect_ratio=aspect_ratio,
        design_spec_raw=design_spec_raw,
        bot_token=bot_token,
    )
    carousel = generate_edu_carousel(
        clean_query,
        custom_topic,
        slide_count,
        context_bundle,
        reply_context=reply_context,
        design_profile=design_profile,
    )
    slides = (carousel.get("slides") or [])[:slide_count]

    dim_label = f"{to_fa_digits(design_profile['width'])}×{to_fa_digits(design_profile['height'])}"
    if not dry_run:
        progress_fn(bot_token, target_chat_ids, status_msg_id, 75, f"در حال رندر گرافیکی {to_fa_digits(len(slides))} اسلاید با ابعاد {dim_label}...")

    # Generate distinct 3D scientific illustrations (up to 4 unique visuals for multi-slide carousels)
    unique_prompts: list[str] = []
    for idx, sl in enumerate(slides):
        vp = (sl.get("visual_prompt") or "").strip() or f"{custom_topic} 3D bioinformatics illustration"
        if idx < 4:
            unique_prompts.append(vp)

    art_pool: list[bytes | None] = []
    if unique_prompts:
        with ThreadPoolExecutor(max_workers=min(4, len(unique_prompts))) as ex:
            art_pool = list(ex.map(lambda p: generate_cf_art(p, design_profile), unique_prompts))

    rendered_slides: list[bytes] = []
    for idx, sl in enumerate(slides):
        bg = art_pool[idx % len(art_pool)] if art_pool else None
        if not bg and art_pool:
            bg = next((a for a in art_pool if a), None)
        rendered_slides.append(
            render_carousel_slide(
                sl,
                len(slides),
                bg,
                font_hint=clean_query,
                design_profile=design_profile,
            )
        )

    if dry_run:
        out_dir = BASE_DIR / "tmp_edu_preview"
        out_dir.mkdir(exist_ok=True)
        for i, b in enumerate(rendered_slides, 1):
            (out_dir / f"slide_{i}.jpg").write_bytes(b)
        print(f"[DRY_RUN] Rendered {len(rendered_slides)} slides ({dim_label}) to {out_dir} in {time.time() - t0:.1f}s")
        print(format_editable_slides_text(carousel))
        if wants_sources:
            print("\n" + format_library_links_message(carousel, matched))
        return

    progress_fn(bot_token, target_chat_ids, status_msg_id, 92, "اسلایدهای تصویری آماده شدند؛ در حال ارسال به تلگرام...")
    for cid in target_chat_ids:
        send_tg_album(bot_token, cid, rendered_slides)
        send_tg_text(bot_token, cid, format_editable_slides_text(carousel))
        if wants_sources:
            send_tg_text(bot_token, cid, format_library_links_message(carousel, matched))

    progress_fn(
        bot_token,
        target_chat_ids,
        status_msg_id,
        100,
        f"پست آموزشی {to_fa_digits(len(slides))} اسلایدی ({dim_label}) آماده و ارسال شد!",
        done=True,
    )
