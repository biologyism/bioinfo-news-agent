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
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

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
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}

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
                    if resp.status_code == 404:
                        last_err = f"HTTP 404 on {model}: {resp.text[:180]}"
                        print(f"[WARN] Edu Gemini {model} -> 404")
                        break
                    if resp.status_code in {400, 403, 429, 503}:
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


def generate_edu_carousel(
    user_query: str,
    topic_hint: str,
    slide_count: int,
    context_bundle: dict[str, Any],
    reply_context: str = "",
) -> dict[str, Any]:
    system_instruction = f"""تو استراتژیست محتوای علمی «آکادمی بیوانفورماتیک ایران» هستی.
مخاطب: کاربران باهوش و کم‌حوصله در ایران (دانشجویان زیست، پزشکی، بیوانفورماتیک و علاقه‌مندان AI).

قوانین طلایی لحن (بسیار حیاتی):
1. **هیچ حس زردی و کلیک‌بیت نداشته باشد:** کلمات زرد مثل «معجزه»، «باورنکردنی»، «۱۰ سال صبر تموم شد»، «انفجار علم» یا «حتماً ببینید» اکیداً ممنوع! لحن باید **خودمونیِ باکلاس، دقیق، تیز و حرفه‌ای** باشد.
2. **فوق‌العاده موجز و کوبنده:** هر بولت نهایتاً ۸ تا ۱۲ کلمه! بدون کش‌دادن، بدون مقدمهٔ طولانی و بدون اصطلاحات سنگین انگلیسی (مثل BindAffinity یا BWT). اگر یک مفهوم تخصصی لازم بود، در همان نیم‌خط با یک مثالِ عادی توضیحش بده.
3. **تعداد دقیق اسلایدها: {slide_count} اسلاید:**
   - اسلاید ۱ (role: "cover"): تیتر کوتاه و کنجکاوکننده (حداکثر ۸ کلمه، غیرزرد) + زیرتیتر ۱ خطی + `visual_prompt` انگلیسی تمیز.
   - اسلایدهای ۲ تا {slide_count - 1} (role: "content"): هر اسلاید یک تکهٔ کلیدی را با ۲ بولت کوتاه و کوبنده بگوید. `tool_spotlight` را فقط اگر ۱ ابزار خیلی ضروری است کوتاه بنویس، وگرنه خالی `""` بگذار تا اسلاید شلوغ نشود.
   - اسلاید آخر (role: "summary"): جمع‌بندی ۱ خطی + یک سؤال حرفه‌ای برای کامنت/ذخیره.
4. `telegram_caption`: **کپشن اینستاگرام خیلی کوتاه و شیک** (نهایتاً ۳ تا ۴ خط کوتاه + یک دلیل منطقی برای Save/Share + ۴ تا ۵ هشتگ تخصصی، با ایموجی‌های به‌جا — اصلاً طولانی نباشد!).
5. **ممنوعیت ایموجی در عکس:** داخل `headline`، `badge`، `body_points` و `tool_spotlight` ایموجی نگذار تا فونت کلمه/پینارِ عکس خراب نشود (ایموجی را فقط در `telegram_caption` بگذار).

خروجی فقط JSON معتبر مطابق اسکیما باشد."""

    source_note = f"\nمتن مرجع انتخابی کاربر برای تبدیل به اسلاید:\n{reply_context[:2500]}\n" if reply_context else ""
    user_prompt = (
        f"درخواست کاربر: {user_query or topic_hint}\n"
        f"تعداد دقیق اسلایدها: {slide_count}\n"
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
                        "body_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "tool_spotlight": {"type": "STRING"},
                        "visual_prompt": {"type": "STRING"},
                    },
                    "required": [
                        "slide_number",
                        "role",
                        "badge",
                        "headline",
                        "body_points",
                        "tool_spotlight",
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
        '\nReturn ONLY valid JSON: {"post_title": str, "slides": [{"slide_number": int, "role": "cover"|"content"|"summary", '
        '"badge": str, "headline": str, "body_points": [str], "tool_spotlight": str, "visual_prompt": str}], '
        '"telegram_caption": str, "selected_repos": [{"repo": str, "why_persian": str}]}.'
    )
    return call_edu_json_with_failover(system_instruction, user_prompt, schema, json_hint)


def generate_cf_art(prompt: str) -> bytes | None:
    account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
    ai_token = os.getenv("CF_AI_TOKEN", "").strip()
    if not account_id or not ai_token or not prompt:
        return None
    clean_p = (
        f"{prompt.strip()}, dark futuristic biomedical editorial illustration, deep navy and glowing cyan tones, "
        "clean composition with uncluttered dark negative space, strictly no text, no letters, no watermark"
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


def render_carousel_slide(slide: dict[str, Any], total_slides: int, cover_bg: bytes | None, font_hint: str = "") -> bytes:
    W, H = 1080, 1350
    role = (slide.get("role") or "content").lower()
    num = int(slide.get("slide_number") or 1)

    font_eb, font_b, font_r, font_sm = load_slide_fonts(role, font_hint)

    # Background base: deep scientific navy gradient
    canvas = Image.new("RGB", (W, H), (10, 16, 34))
    draw = ImageDraw.Draw(canvas)
    for y in range(H):
        t = y / H
        r = int(10 + 8 * t)
        g = int(17 + 12 * t)
        b = int(38 + 20 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    if cover_bg:
        try:
            bg_img = Image.open(io.BytesIO(cover_bg)).convert("RGB")
            if role == "cover":
                bg_fit = ImageOps.fit(bg_img, (W, 760), method=Image.Resampling.LANCZOS, centering=(0.5, 0.4))
                canvas.paste(bg_fit, (0, 0))
                # Smooth fade from illustration into deep navy card area
                fade = Image.new("RGBA", (W, 460), (0, 0, 0, 0))
                fdraw = ImageDraw.Draw(fade)
                for y in range(460):
                    alpha = int(255 * ((y / 460) ** 1.5))
                    fdraw.line([(0, y), (W, y)], fill=(12, 19, 40, alpha))
                rgba = canvas.convert("RGBA")
                rgba.alpha_composite(fade, (0, 320))
                canvas = rgba.convert("RGB")
            else:
                # Subtle top science strip for content slides
                strip = ImageOps.fit(bg_img, (W, 250), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                strip = strip.point(lambda p: int(p * 0.45))
                canvas.paste(strip, (0, 0))
                fade = Image.new("RGBA", (W, 180), (0, 0, 0, 0))
                fdraw = ImageDraw.Draw(fade)
                for y in range(180):
                    alpha = int(255 * (y / 180))
                    fdraw.line([(0, y), (W, y)], fill=(12, 19, 40, alpha))
                rgba = canvas.convert("RGBA")
                rgba.alpha_composite(fade, (0, 75))
                canvas = rgba.convert("RGB")
        except Exception:
            pass

    draw = ImageDraw.Draw(canvas)
    kw_dir = {"direction": "rtl"} if HAS_RAQM else {}

    # Top slide pill badge
    badge_txt = strip_emojis_for_card(slide.get("badge") or f"اسلاید {to_fa_digits(num)} از {to_fa_digits(total_slides)}")
    pill_color = (16, 185, 129) if role == "summary" else (56, 189, 248)
    draw.rounded_rectangle([(W - 390, 56), (W - 64, 112)], radius=24, fill=(18, 30, 60), outline=pill_color, width=2)
    draw.text((W - 227, 84), shape_text(badge_txt), font=font_sm, fill=pill_color, anchor="mm", **kw_dir)

    # Headline
    headline = strip_emojis_for_card(slide.get("headline") or "")
    if role == "cover":
        y_cursor = 640
        head_lines = wrap_text_lines(draw, headline, font_eb, W - 140)
        for line in head_lines[:3]:
            draw.text((W - 72, y_cursor), shape_text(line), font=font_eb, fill=(255, 255, 255), anchor="ra", **kw_dir)
            y_cursor += 78
        draw.line([(W - 240, y_cursor + 12), (W - 72, y_cursor + 12)], fill=(56, 189, 248), width=6)
        y_cursor += 44

        for pt in (slide.get("body_points") or [])[:2]:
            for l in wrap_text_lines(draw, strip_emojis_for_card(pt), font_b, W - 150)[:3]:
                draw.text((W - 72, y_cursor), shape_text(l), font=font_b, fill=(195, 215, 245), anchor="ra", **kw_dir)
                y_cursor += 46
            y_cursor += 12
    else:
        y_cursor = 165
        head_lines = wrap_text_lines(draw, headline, font_eb, W - 144)
        for line in head_lines[:3]:
            draw.text((W - 72, y_cursor), shape_text(line), font=font_eb, fill=(125, 211, 252), anchor="ra", **kw_dir)
            y_cursor += 64
        y_cursor += 28

        # Bullet cards
        for pt in (slide.get("body_points") or [])[:3]:
            clean_pt = strip_emojis_for_card(pt)
            wrapped = wrap_text_lines(draw, clean_pt, font_r, W - 200)
            box_h = max(92, len(wrapped) * 44 + 42)
            if y_cursor + box_h > 980:
                break
            draw.rounded_rectangle([(68, y_cursor), (W - 68, y_cursor + box_h)], radius=20, fill=(20, 31, 60), outline=(44, 62, 108), width=2)
            # Cyan vertical accent bar on right side
            draw.rounded_rectangle([(W - 84, y_cursor + 18), (W - 74, y_cursor + box_h - 18)], radius=4, fill=pill_color)
            ty = y_cursor + 22
            for l in wrapped:
                draw.text((W - 104, ty), shape_text(l), font=font_r, fill=(238, 242, 255), anchor="ra", **kw_dir)
                ty += 44
            y_cursor += box_h + 22

    # Tool Spotlight Box positioned cleanly below content without excessive empty gap
    spotlight = strip_emojis_for_card(slide.get("tool_spotlight") or "")
    if spotlight:
        spot_lines = wrap_text_lines(draw, f"ابزار پیشنهادی گیت‌هاب: {spotlight}", font_sm, W - 190)[:2]
        s_h = 34 + len(spot_lines) * 36
        s_y = min(max(y_cursor + 32, 780 if role != "cover" else y_cursor + 24), 1150 - s_h)
        draw.rounded_rectangle([(68, s_y), (W - 68, s_y + s_h)], radius=18, fill=(12, 42, 58), outline=(52, 211, 153), width=2)
        sy = s_y + 18
        for l in spot_lines:
            draw.text((W - 96, sy), shape_text(l), font=font_sm, fill=(167, 243, 208), anchor="ra", **kw_dir)
            sy += 36

    canvas = apply_academy_footer(canvas)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92, optimize=True)
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

    media = []
    files = {}
    for idx, img_bytes in enumerate(slide_images[:10], 1):
        key = f"slide{idx}"
        media.append({"type": "photo", "media": f"attach://{key}"})
        files[key] = (f"slide_{idx}.jpg", img_bytes, "image/jpeg")

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

    carousel = generate_edu_carousel(clean_query, custom_topic, slide_count, context_bundle, reply_context=reply_context)
    slides = (carousel.get("slides") or [])[:slide_count]

    if not dry_run:
        progress_fn(bot_token, target_chat_ids, status_msg_id, 75, f"در حال رندر گرافیکی {to_fa_digits(len(slides))} اسلاید آماده انتشار (۱۰۸۰×۱۳۵۰)...")

    cover_prompt = (slides[0].get("visual_prompt") if slides else "") or f"{custom_topic} scientific illustration"
    slide_prompt_2 = (slides[1].get("visual_prompt") if len(slides) > 1 else cover_prompt) or cover_prompt
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut1 = ex.submit(generate_cf_art, cover_prompt)
        fut2 = ex.submit(generate_cf_art, slide_prompt_2)
        bg1 = fut1.result()
        bg2 = fut2.result()

    rendered_slides: list[bytes] = []
    for idx, sl in enumerate(slides):
        bg = bg1 if idx == 0 else (bg2 or bg1)
        rendered_slides.append(render_carousel_slide(sl, len(slides), bg, font_hint=clean_query))

    if dry_run:
        out_dir = BASE_DIR / "tmp_edu_preview"
        out_dir.mkdir(exist_ok=True)
        for i, b in enumerate(rendered_slides, 1):
            (out_dir / f"slide_{i}.jpg").write_bytes(b)
        print(f"[DRY_RUN] Rendered {len(rendered_slides)} slides to {out_dir} in {time.time() - t0:.1f}s")
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

    progress_fn(bot_token, target_chat_ids, status_msg_id, 100, f"پست آموزشی {to_fa_digits(len(slides))} اسلایدی با تصویر آماده و ارسال شد!", done=True)
