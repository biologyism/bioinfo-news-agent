import csv
import html
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

BASE_DIR = Path(__file__).resolve().parent
SOURCES_PATH = BASE_DIR / "sources.json"
STATE_DIR = BASE_DIR / "state"
STATE_PATH = STATE_DIR / "seen_articles.json"

IRST = timezone(timedelta(hours=3, minutes=30))

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 "
    "IranBioinformaticsAgent/1.0"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}

GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ocid",
    "rss",
}

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def to_persian_digits(value: Any) -> str:
    return str(value).translate(PERSIAN_DIGITS)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_qs = {k: v for k, v in qs.items() if k.lower() not in TRACKING_PARAMS}
        new_query = urlencode(cleaned_qs, doseq=True)
        path = parsed.path.rstrip("/") or "/"
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", new_query, ""))
    except Exception:
        return url


def normalize_title_tokens(title: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (title or "").lower())
    stop = {
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with",
        "by", "from", "at", "is", "are", "as", "new", "using", "via", "how",
    }
    return {w for w in cleaned.split() if len(w) > 2 and w not in stop}


def titles_are_similar(t1: str, t2: str) -> bool:
    s1 = normalize_title_tokens(t1)
    s2 = normalize_title_tokens(t2)
    if not s1 or not s2:
        return False
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return (inter / union) >= 0.55 or (inter >= 4 and inter / min(len(s1), len(s2)) >= 0.75)


def clean_html_text(raw: str, max_len: int = 1200) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return text[:max_len]


TZINFOS = {
    "EST": -18000,
    "EDT": -14400,
    "CST": -21600,
    "CDT": -18000,
    "MST": -25200,
    "MDT": -21600,
    "PST": -28800,
    "PDT": -25200,
    "GMT": 0,
    "UTC": 0,
}


def parse_entry_date(entry: Any) -> datetime | None:
    for attr in ("published", "updated", "created"):
        val = entry.get(attr)
        if val:
            try:
                dt = date_parser.parse(val, tzinfos=TZINFOS)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    for attr in ("published_parsed", "updated_parsed"):
        val = entry.get(attr)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def load_sources() -> dict[str, Any]:
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    sheet_csv_url = os.getenv("GOOGLE_SHEET_CSV_URL", "").strip()
    if sheet_csv_url:
        try:
            resp = requests.get(sheet_csv_url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            reader = csv.DictReader(io.StringIO(resp.text))
            added = 0
            for row in reader:
                url = (row.get("url") or "").strip()
                if not url:
                    continue
                enabled = (row.get("enabled") or "true").strip().lower() not in {"false", "0", "no"}
                config["rss_feeds"].append(
                    {
                        "name": (row.get("name") or url).strip(),
                        "category": (row.get("category") or "custom").strip(),
                        "url": url,
                        "weight": float(row.get("weight") or 1.25),
                        "enabled": enabled,
                    }
                )
                added += 1
            print(f"[INFO] Loaded {added} extra sources from GOOGLE_SHEET_CSV_URL.")
        except Exception as exc:
            print(f"[WARN] Could not load GOOGLE_SHEET_CSV_URL: {exc}")

    return config


def load_state(retention_days: int) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    today_irst = datetime.now(IRST).strftime("%Y-%m-%d")
    default_state = {
        "daily_date": today_irst,
        "daily_post_count": 0,
        "seen": {},
    }
    if not STATE_PATH.exists():
        return default_state

    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return default_state

    if state.get("daily_date") != today_irst:
        state["daily_date"] = today_irst
        state["daily_post_count"] = 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cleaned_seen = {}
    for url, meta in state.get("seen", {}).items():
        try:
            ts = date_parser.parse(meta.get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                cleaned_seen[url] = meta
        except Exception:
            continue
    state["seen"] = cleaned_seen
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fetch_single_rss(feed_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    url = feed_cfg["url"]
    name = feed_cfg["name"]
    category = feed_cfg.get("category", "general")
    weight = float(feed_cfg.get("weight", 1.0))
    items: list[dict[str, Any]] = []

    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"[WARN] Feed HTTP {resp.status_code}: {name}")
            return items
        parsed = feedparser.parse(resp.content)
        for entry in parsed.entries[:35]:
            link = normalize_url(entry.get("link") or "")
            title = clean_html_text(entry.get("title") or "", max_len=300)
            if not link or not title:
                continue

            summary_raw = entry.get("summary") or entry.get("description") or ""
            if not summary_raw and entry.get("content"):
                summary_raw = entry["content"][0].get("value", "")
            summary = clean_html_text(summary_raw, max_len=1400)

            authors = []
            if entry.get("authors"):
                authors = [a.get("name", "") for a in entry["authors"] if a.get("name")]
            elif entry.get("author"):
                authors = [entry.get("author")]

            pub_dt = parse_entry_date(entry)
            source_name = name
            if "news.google.com" in url and " - " in title:
                parts = title.rsplit(" - ", 1)
                if len(parts[1]) <= 60:
                    title = parts[0].strip()
                    source_name = f"{parts[1].strip()} (via Google News)"

            items.append(
                {
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "category": category,
                    "weight": weight,
                    "summary": summary,
                    "authors": ", ".join(authors[:5]),
                    "published_utc": pub_dt.isoformat() if pub_dt else "",
                    "published_dt": pub_dt,
                }
            )
    except Exception as exc:
        print(f"[WARN] Error fetching {name}: {exc}")
    return items


def fetch_pubmed_query(query_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    term = query_cfg["term"]
    retmax = int(query_cfg.get("retmax", 12))
    weight = float(query_cfg.get("weight", 1.4))
    category = query_cfg.get("category", "bioinformatics")
    items: list[dict[str, Any]] = []

    try:
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        r1 = requests.get(
            esearch_url,
            params={
                "db": "pubmed",
                "term": term,
                "retmax": retmax,
                "sort": "date",
                "retmode": "json",
            },
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        r1.raise_for_status()
        id_list = r1.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return items

        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r2 = requests.get(
            efetch_url,
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        r2.raise_for_status()
        root = ET.fromstring(r2.content)

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID") or ""
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            if not pmid or not title:
                continue

            journal = article.findtext(".//Journal/Title") or article.findtext(".//Journal/ISOAbbreviation") or "PubMed"
            abs_parts = ["".join(el.itertext()).strip() for el in article.findall(".//Abstract/AbstractText")]
            abstract = " ".join(abs_parts)[:1600]

            author_names = []
            for au in article.findall(".//AuthorList/Author")[:4]:
                ln = au.findtext("LastName") or au.findtext("CollectiveName")
                if ln:
                    author_names.append(ln)

            pub_year = article.findtext(".//JournalIssue/PubDate/Year") or article.findtext(".//ArticleDate/Year")
            pub_month = article.findtext(".//JournalIssue/PubDate/Month") or article.findtext(".//ArticleDate/Month") or "01"
            pub_day = article.findtext(".//JournalIssue/PubDate/Day") or article.findtext(".//ArticleDate/Day") or "01"
            pub_dt = None
            if pub_year:
                try:
                    pub_dt = date_parser.parse(f"{pub_year}-{pub_month}-{pub_day}").replace(tzinfo=timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

            pmc_id = ""
            for aid in article.findall(".//ArticleIdList/ArticleId"):
                if aid.attrib.get("IdType") == "pmc":
                    pmc_id = aid.text or ""
                    break

            oa_note = " | Open Access (PMC)" if pmc_id else ""
            items.append(
                {
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": f"{journal}{oa_note}",
                    "category": category,
                    "weight": weight,
                    "summary": abstract,
                    "authors": ", ".join(author_names),
                    "published_utc": pub_dt.isoformat() if pub_dt else "",
                    "published_dt": pub_dt,
                }
            )
    except Exception as exc:
        print(f"[WARN] PubMed query failed ({query_cfg.get('name')}): {exc}")
    return items


def enrich_article_snippet(item: dict[str, Any]) -> None:
    url = item["url"]
    if "news.google.com" in url or "pubmed.ncbi.nlm.nih.gov" in url:
        return
    if len(item.get("summary", "")) >= 450:
        return
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            return
        soup = BeautifulSoup(resp.text, "html.parser")
        meta_desc = ""
        for selector in [
            ("meta", {"name": "description"}),
            ("meta", {"property": "og:description"}),
            ("meta", {"name": "dc.description"}),
        ]:
            tag = soup.find(selector[0], selector[1])
            if tag and tag.get("content"):
                meta_desc = tag["content"].strip()
                break

        paragraphs = [
            re.sub(r"\s+", " ", p.get_text(" ", strip=True))
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 70
        ]
        body_excerpt = " ".join(paragraphs[:5])[:1200]
        combined = f"{meta_desc} {body_excerpt}".strip()
        if len(combined) > len(item.get("summary", "")):
            item["summary"] = combined[:1500]
    except Exception:
        pass


def score_and_deduplicate(
    raw_items: list[dict[str, Any]],
    seen_map: dict[str, Any],
    lookback_hours: int,
    priority_keywords: dict[str, int],
) -> list[dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - timedelta(hours=lookback_hours)

    seen_titles = [meta.get("title", "") for meta in seen_map.values() if meta.get("title")]

    fresh: list[dict[str, Any]] = []
    for item in raw_items:
        url = item["url"]
        if url in seen_map:
            continue
        pub_dt = item.get("published_dt")
        if pub_dt and pub_dt < cutoff_dt:
            continue
        if any(titles_are_similar(item["title"], old_t) for old_t in seen_titles):
            continue
        fresh.append(item)

    merged: list[dict[str, Any]] = []
    for item in fresh:
        matched_cluster = None
        for existing in merged:
            if item["url"] == existing["url"] or titles_are_similar(item["title"], existing["title"]):
                matched_cluster = existing
                break
        if matched_cluster:
            matched_cluster["buzz_count"] += 1
            if item["source"] not in matched_cluster["also_reported_by"]:
                matched_cluster["also_reported_by"].append(item["source"])
            if len(item["summary"]) > len(matched_cluster["summary"]):
                matched_cluster["summary"] = item["summary"]
            if item["weight"] > matched_cluster["weight"]:
                matched_cluster["weight"] = item["weight"]
        else:
            item["buzz_count"] = 1
            item["also_reported_by"] = [item["source"]]
            merged.append(item)

    for item in merged:
        text_blob = f"{item['title']} {item['summary']}".lower()
        kw_score = sum(pts for kw, pts in priority_keywords.items() if kw.lower() in text_blob)

        recency_bonus = 2.0
        if item.get("published_dt"):
            age_hours = max(0.0, (now_utc - item["published_dt"]).total_seconds() / 3600.0)
            recency_bonus = max(0.0, 6.0 - (age_hours / 6.0))

        buzz_bonus = (item["buzz_count"] - 1) * 4.0
        cat_bonus = 3.0 if item["category"] in {"bioinformatics", "ai_bio"} else 1.0
        item["pre_score"] = round((kw_score + recency_bonus + buzz_bonus + cat_bonus) * item["weight"], 2)

    merged.sort(key=lambda x: x["pre_score"], reverse=True)
    return merged


def call_gemini(candidates: list[dict[str, Any]], max_posts: int) -> list[dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    preferred_model = os.getenv("GEMINI_MODEL", "").strip()
    models_to_try = ([preferred_model] if preferred_model else []) + [
        m for m in GEMINI_FALLBACK_MODELS if m != preferred_model
    ]

    compact_articles = []
    for idx, item in enumerate(candidates):
        pub_str = ""
        if item.get("published_dt"):
            pub_str = item["published_dt"].astimezone(IRST).strftime("%Y-%m-%d %H:%M IRST")
        compact_articles.append(
            {
                "id": idx,
                "title": item["title"],
                "source": item["source"],
                "also_reported_by": item.get("also_reported_by", []),
                "category": item["category"],
                "authors": item.get("authors", ""),
                "published": pub_str or item.get("published_utc", ""),
                "url": item["url"],
                "summary": item.get("summary", "")[:1100],
            }
        )

    today_str = datetime.now(IRST).strftime("%Y-%m-%d")

    system_instruction = """تو سردبیر ارشد علمی «آکادمی بیوانفورماتیک ایران» هستی.
وظیفه‌ات انتخاب داغ‌ترین، مهم‌ترین و معتبرترین اخبار از بین لیست مقالات و خبرهای ورودی و نگارش پست فارسی دقیقاً به سبک ژورنالیسم علمی حرفه‌ای است.

قوانین انتخاب خبر:
1. اولویت اول: بیوانفورماتیک، ژنومیکس، تحلیل تک‌سلولی، طراحی پروتئین، AI در زیست‌شناسی و کشف دارو، و ابزارهای جدید محاسباتی.
2. اولویت دوم: کشفیات بزرگ زیست‌شناسی (Nature/Science/Cell/Quanta) و تحولات مهم هوش مصنوعی (OpenAI, Anthropic, DeepMind, Hugging Face, مدل‌ها و پژوهش‌های شاخص).
3. از بین ورودی‌ها، حداکثر تعداد درخواستی خبر واقعاً ارزشمند و غیرتکراری انتخاب کن. اگر خبر خیلی مهمی نبود، تعداد کمتری انتخاب کن (کیفیت بر کمیت مقدم است).

قوانین نگارش هر پست (دقیقاً ۶ بخش، موجز، دقیق، بدون اغراق و بدون توهم‌زایی):
- short_title: عنوان بسیار کوتاه دوپاره با علامت «؛» — مثل: `VHSeek؛ میزبان ویروس را حدس بزن` یا `Space-map؛ بافت در سه بعد`
- emoji: یک ایموجی مرتبط و جذاب (مثل 🦠، 📊، 🗺️، 🎭، 🛡️، 🔍، 📚، 💻، 📈، 🧬، 🧪، 🤖)
- hook_headline: یک جمله کوتاه و کنجکاوی‌برانگیز درباره هسته خبر (بدون اغراق زرد)
- why_it_matters: بخش 🎯 — ۱ تا ۲ جمله درباره شکاف علمی یا اهمیت واقعی مسئله
- what_happened: بخش 🔬 — گزارش دقیق علمی: در صورت وجود در متن ورودی، نام نویسندگان/تیم/شرکت، نام ژورنال یا رسانه، تاریخ انتشار به فارسی (مثلاً ۲۰ نوامبر ۲۰۲۶)، خلاصه روش و اعداد دقیق و اشاره به دسترسی‌باز (Open Access) بودن. **اکیداً ممنوع:** هرگز نام نویسنده، عدد، درصد، جلد مجله یا ادعایی را که در ورودی نیست حدس نزن یا نساز!
- analysis_and_limitation: بخش 💡 — ۱ تا ۲ جمله شامل معنای کاربردی خبر **به‌همراه نقد صادقانه و بیان محدودیت واقعی** (مثلاً پیش‌بینی محاسباتی است نه اثبات آزمایشگاهی، پیش‌چاپ است و داوری نشده، مطالعه تکی است، یا ارقام اعلامی خود شرکت است)
- closing_question: یک سؤال کوتاه، هوشمندانه و چالشی در پایان بخش 💡 برای ایجاد بحث بین متخصصان و دانشجویان
- source_label: نام دقیق منبع و تاریخ به فارسی (مثلاً: `Nature Communications، ۲۰ نوامبر ۲۰۲۶` یا `bioRxiv (پیش‌چاپ)، ۲۲ نوامبر ۲۰۲۶`)

خروجی باید صرفاً JSON معتبر مطابق اسکیما باشد."""

    user_prompt = (
        f"تاریخ امروز: {today_str}\n"
        f"حداکثر تعداد پست انتخابی در این نوبت: {max_posts}\n\n"
        f"لیست اخبار و مقالات خام ورودی:\n{json.dumps(compact_articles, ensure_ascii=False)}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "selected_posts": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "article_id": {"type": "INTEGER"},
                                "short_title": {"type": "STRING"},
                                "emoji": {"type": "STRING"},
                                "hook_headline": {"type": "STRING"},
                                "why_it_matters": {"type": "STRING"},
                                "what_happened": {"type": "STRING"},
                                "analysis_and_limitation": {"type": "STRING"},
                                "closing_question": {"type": "STRING"},
                                "source_label": {"type": "STRING"},
                            },
                            "required": [
                                "article_id",
                                "short_title",
                                "emoji",
                                "hook_headline",
                                "why_it_matters",
                                "what_happened",
                                "analysis_and_limitation",
                                "closing_question",
                                "source_label",
                            ],
                        },
                    }
                },
                "required": ["selected_posts"],
            },
        },
    }

    last_err = None
    for model in models_to_try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            print(f"[INFO] Calling Gemini model: {model}")
            resp = requests.post(
                endpoint,
                params={"key": api_key},
                json=payload,
                timeout=90,
            )
            if resp.status_code in {404, 429, 503}:
                print(f"[WARN] Model {model} returned {resp.status_code}: {resp.text[:250]}")
                last_err = resp.text
                continue
            resp.raise_for_status()
            data = resp.json()
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text_out)
            posts = parsed.get("selected_posts", [])
            id_map = {idx: item for idx, item in enumerate(candidates)}
            enriched_posts = []
            for p in posts:
                art = id_map.get(p.get("article_id"))
                if not art:
                    continue
                p["url"] = art["url"]
                p["raw_title"] = art["title"]
                enriched_posts.append(p)
            return enriched_posts[:max_posts]
        except Exception as exc:
            print(f"[WARN] Gemini call failed on {model}: {exc}")
            last_err = str(exc)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def format_telegram_html(post: dict[str, Any], post_number: int) -> str:
    num_fa = to_persian_digits(post_number)
    short_title = html.escape(post["short_title"].strip())
    emoji = html.escape(post["emoji"].strip())
    hook = html.escape(post["hook_headline"].strip())
    why = html.escape(post["why_it_matters"].strip())
    what = html.escape(post["what_happened"].strip())
    analysis = html.escape(post["analysis_and_limitation"].strip())
    question = html.escape(post["closing_question"].strip())
    source = html.escape(post["source_label"].strip())
    url = html.escape(post["url"].strip(), quote=True)

    return (
        f"<b>پست {num_fa} | {short_title}</b>\n\n"
        f"{emoji} {hook}\n\n"
        f"🎯 {why}\n\n"
        f"🔬 {what}\n\n"
        f"💡 {analysis} {question}\n\n"
        f"منبع: {source}\n"
        f'🔗 <a href="{url}">مشاهده منبع اصلی</a>'
    )


def send_telegram_message(bot_token: str, chat_id: str, html_text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")


def main() -> None:
    dry_run = os.getenv("DRY_RUN", "false").strip().lower() in {"true", "1", "yes"}
    config = load_sources()
    settings = config.get("settings", {})

    max_posts = int(settings.get("max_posts_per_run", 5))
    max_candidates = int(settings.get("max_candidates_for_ai", 35))
    lookback_hours = int(settings.get("lookback_hours", 36))
    retention_days = int(settings.get("state_retention_days", 21))
    fetch_body = bool(settings.get("fetch_article_body", True))
    priority_keywords = config.get("priority_keywords", {})

    state = load_state(retention_days=retention_days)
    raw_items: list[dict[str, Any]] = []

    enabled_feeds = [f for f in config.get("rss_feeds", []) if f.get("enabled", True)]
    enabled_pubmed = [q for q in config.get("pubmed_queries", []) if q.get("enabled", True)]

    print(f"[INFO] Fetching {len(enabled_feeds)} RSS feeds and {len(enabled_pubmed)} PubMed queries...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch_single_rss, f) for f in enabled_feeds]
        futures.extend(executor.submit(fetch_pubmed_query, q) for q in enabled_pubmed)
        for fut in as_completed(futures):
            raw_items.extend(fut.result())

    print(f"[INFO] Total raw entries fetched: {len(raw_items)}")

    candidates = score_and_deduplicate(
        raw_items=raw_items,
        seen_map=state["seen"],
        lookback_hours=lookback_hours,
        priority_keywords=priority_keywords,
    )
    print(f"[INFO] Fresh deduplicated candidates: {len(candidates)}")

    if not candidates:
        print("[INFO] No fresh news items found in this window. Exiting cleanly.")
        save_state(state)
        return

    top_candidates = candidates[:max_candidates]

    if fetch_body:
        print(f"[INFO] Enriching article excerpts for top {min(20, len(top_candidates))} candidates...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(enrich_article_snippet, top_candidates[:20]))

    if dry_run and not os.getenv("GEMINI_API_KEY"):
        print("[DRY_RUN] GEMINI_API_KEY not set; printing top 5 pre-scored candidates:")
        for i, c in enumerate(top_candidates[:5], 1):
            print(f"  {i}. [{c['pre_score']}] ({c['source']}) {c['title']} -> {c['url']}")
        return

    selected_posts = call_gemini(top_candidates, max_posts=max_posts)
    print(f"[INFO] Gemini selected {len(selected_posts)} posts.")

    if not selected_posts:
        print("[INFO] Gemini found no high-impact stories worth posting right now.")
        now_iso = datetime.now(timezone.utc).isoformat()
        for c in top_candidates[:15]:
            state["seen"][c["url"]] = {"title": c["title"], "ts": now_iso, "status": "skipped"}
        save_state(state)
        return

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not dry_run and (not bot_token or not chat_id):
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

    now_iso = datetime.now(timezone.utc).isoformat()
    sent_urls = set()

    for post in selected_posts:
        state["daily_post_count"] += 1
        msg_html = format_telegram_html(post, state["daily_post_count"])
        if dry_run:
            print("\n" + "=" * 50)
            print(msg_html)
        else:
            send_telegram_message(bot_token, chat_id, msg_html)
            time.sleep(1.2)
        sent_urls.add(post["url"])
        state["seen"][post["url"]] = {
            "title": post.get("raw_title") or post["short_title"],
            "ts": now_iso,
            "status": "sent",
        }

    # Mark remaining top evaluated items as skipped so they are not re-evaluated repeatedly
    for c in top_candidates:
        if c["url"] not in sent_urls and c["url"] not in state["seen"]:
            state["seen"][c["url"]] = {
                "title": c["title"],
                "ts": now_iso,
                "status": "skipped",
            }

    if not dry_run:
        save_state(state)
    print("[INFO] Done successfully.")


if __name__ == "__main__":
    main()
