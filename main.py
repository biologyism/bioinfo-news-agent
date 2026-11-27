import base64
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
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
SOURCES_PATH = BASE_DIR / "sources.json"
STATE_DIR = BASE_DIR / "state"
STATE_PATH = STATE_DIR / "seen_articles.json"
OVERLAY_PATH = BASE_DIR / "assets" / "brand_overlay.png"

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
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]


def get_gemini_keys() -> list[str]:
    raw = os.getenv("GEMINI_API_KEY", "").strip()
    if not raw:
        return []
    keys = [k.strip() for k in re.split(r"[,;\s\n]+", raw) if k.strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]

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
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
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


def fetch_openalex_works(query: str, per_page: int = 12, weight: float = 1.8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not query:
        return items
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": per_page, "sort": "publication_date:desc"},
            headers=REQUEST_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return items
        for w in resp.json().get("results", []):
            title = clean_html_text(w.get("title") or "", max_len=300)
            link = normalize_url(
                (w.get("primary_location") or {}).get("landing_page_url")
                or w.get("doi")
                or w.get("id")
                or ""
            )
            if not title or not link:
                continue
            # Reconstruct abstract from OpenAlex inverted index if present
            inv = w.get("abstract_inverted_index") or {}
            abstract = ""
            if isinstance(inv, dict) and inv:
                max_pos = max((max(pos_list) for pos_list in inv.values() if pos_list), default=-1)
                if 0 <= max_pos < 600:
                    words = [""] * (max_pos + 1)
                    for token, pos_list in inv.items():
                        for p in pos_list:
                            words[p] = token
                    abstract = " ".join(x for x in words if x)[:1500]
            pub_date_str = w.get("publication_date") or ""
            pub_dt = None
            if pub_date_str:
                try:
                    pub_dt = date_parser.parse(pub_date_str).replace(tzinfo=timezone.utc)
                except Exception:
                    pub_dt = None
            src_name = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "OpenAlex Research"
            items.append(
                {
                    "title": title,
                    "url": link,
                    "source": src_name,
                    "category": "custom_topic",
                    "weight": weight,
                    "summary": abstract,
                    "authors": "",
                    "published_utc": pub_dt.isoformat() if pub_dt else "",
                    "published_dt": pub_dt,
                }
            )
    except Exception as exc:
        print(f"[WARN] OpenAlex search failed ({query}): {exc}")
    return items


def _fold_accents(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c)).lower()


def expand_on_demand_topic(custom_topic: str, custom_query: str) -> dict[str, Any]:
    """Turn ANY Persian/English user phrase into precise search queries without broadening niche topics."""
    raw_phrase = (custom_query or custom_topic or "").strip()
    generic_stops = {"syndrome", "disease", "disorder", "infection", "cancer", "virus", "protein", "gene", "cell", "study", "news", "سندرم", "بیماری"}
    raw_tokens = [t.lower() for t in re.split(r"[\s\-_,،]+", _fold_accents(custom_topic or raw_phrase)) if len(t) >= 3]
    distinct_tokens = [t for t in raw_tokens if t not in generic_stops] or raw_tokens
    fallback = {
        "is_specific": bool(custom_topic and custom_topic.lower() not in {"bioinformatics biology ai", "bioinformatics", "artificial intelligence", "biology"}),
        "primary_en": custom_topic or raw_phrase,
        "alt_en": "",
        "primary_fa": raw_phrase,
        "match_tokens": distinct_tokens,
    }
    if not raw_phrase:
        return fallback

    prompt = (
        "Analyze this user news request and return ONLY valid JSON.\n"
        f'User query: "{raw_phrase}"\n'
        f'Initial topic hint: "{custom_topic}"\n'
        "Rules:\n"
        "- is_specific: true ONLY if the user asked about a concrete scientific subject (disease, syndrome, virus, gene, drug, method, company, person, animal — e.g. Brown-Sequard syndrome, parvovirus, CRISPR, Ozempic, neuralink). Set is_specific: false if they asked broadly for daily/viral/controversial/trending/interesting news (e.g. 'خبر جنجالی', 'خبر داغ', 'ریپلای خور', 'خبرهای خفن') without naming a specific disease/gene/topic.\n"
        "- primary_en: exact specific English/scientific search phrase (NEVER broaden 'parvovirus' to 'virus', NEVER broaden 'Brown-Sequard syndrome' to 'neurology').\n"
        "- alt_en: 1 short alternative exact English search query (or empty string).\n"
        "- primary_fa: exact short Persian topic phrase.\n"
        "- match_tokens: 1 to 3 core lowercase keywords (e.g. ['parvovirus'] or ['brown-sequard', 'brown sequard', 'hemisection']) that an article MUST contain to be truly relevant.\n"
        'JSON schema: {"is_specific": bool, "primary_en": str, "alt_en": str, "primary_fa": str, "match_tokens": [str]}'
    )

    gemini_keys = get_gemini_keys()
    if gemini_keys:
        for model in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
            for api_key in gemini_keys:
                try:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        params={"key": api_key},
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                        },
                        timeout=15,
                    )
                    if r.status_code == 200:
                        data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                        if data.get("primary_en"):
                            print(f"[INFO] Expanded custom topic ({model}): {data}")
                            return data
                    elif r.status_code in {400, 403, 429, 503}:
                        continue
                    elif r.status_code == 404:
                        break
                except Exception:
                    continue

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        res = call_openai_compat_json(
            "https://api.groq.com/openai/v1/chat/completions",
            groq_key,
            "llama-3.3-70b-versatile",
            "Return ONLY valid JSON.",
            prompt,
            timeout=12,
        )
        if res and res.get("primary_en"):
            return res

    return fallback


def enrich_article_snippet(item: dict[str, Any]) -> None:
    url = item["url"]
    if "news.google.com" in url or "pubmed.ncbi.nlm.nih.gov" in url:
        return
    if len(item.get("summary", "")) >= 450 and item.get("og_image"):
        return
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            return
        soup = BeautifulSoup(resp.text, "html.parser")
        if not item.get("og_image"):
            og_img = soup.find("meta", {"property": "og:image"}) or soup.find("meta", {"name": "twitter:image"})
            if og_img and og_img.get("content") and og_img["content"].startswith("http"):
                item["og_image"] = og_img["content"].strip()

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
    custom_topic: str = "",
    allow_seen_fallback: bool = False,
) -> list[dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - timedelta(hours=lookback_hours)

    seen_titles = [meta.get("title", "") for meta in seen_map.values() if meta.get("title")]

    fresh: list[dict[str, Any]] = []
    fallback_pool: list[dict[str, Any]] = []
    for item in raw_items:
        url = item["url"]
        pub_dt = item.get("published_dt")
        if pub_dt and pub_dt < cutoff_dt:
            continue
        is_seen = url in seen_map or any(titles_are_similar(item["title"], old_t) for old_t in seen_titles)
        if is_seen:
            if allow_seen_fallback and seen_map.get(url, {}).get("status") != "sent":
                fallback_pool.append(item)
            continue
        fresh.append(item)

    if allow_seen_fallback and len(fresh) < 12 and fallback_pool:
        fresh.extend(fallback_pool)

    topic_tokens = [t.lower() for t in re.split(r"[\s,،]+", custom_topic or "") if len(t) >= 2]

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
        text_blob = f"{item['title']} {item['summary']} {item['category']} {item['source']}".lower()
        kw_score = sum(pts for kw, pts in priority_keywords.items() if kw.lower() in text_blob)

        topic_bonus = 0.0
        if topic_tokens:
            hits = sum(1 for tok in topic_tokens if tok in text_blob)
            topic_bonus = hits * 12.0

        recency_bonus = 2.0
        if item.get("published_dt"):
            age_hours = max(0.0, (now_utc - item["published_dt"]).total_seconds() / 3600.0)
            recency_bonus = max(0.0, 6.0 - (age_hours / 6.0))

        buzz_bonus = (item["buzz_count"] - 1) * 4.0
        cat_bonus = 3.0 if item["category"] in {"bioinformatics", "ai_bio"} else 1.0
        item["pre_score"] = round((kw_score + topic_bonus + recency_bonus + buzz_bonus + cat_bonus) * item["weight"], 2)

    merged.sort(key=lambda x: x["pre_score"], reverse=True)
    return merged


def call_openai_compat_json(
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 25,
    extra_headers: dict[str, str] | None = None,
    use_json_mode: bool = True,
) -> dict[str, Any] | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.25,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 400 and use_json_mode:
            payload.pop("response_format", None)
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            print(f"[WARN] {model} HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            if m:
                return json.loads(m.group(0))
            raise
    except Exception as exc:
        print(f"[WARN] OpenAI-compat call failed ({model}): {exc}")
        return None


def run_scout_agent(
    candidates: list[dict[str, Any]],
    target_keep: int = 15,
    custom_topic: str = "",
) -> list[dict[str, Any]]:
    """Agent 1 (Groq / Cloudflare AI / OpenRouter): Ultra-fast triage of up to 50 headlines."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    cf_acc = os.getenv("CF_ACCOUNT_ID", "").strip()
    cf_tok = os.getenv("CF_AI_TOKEN", "").strip()
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if (not groq_key and not or_key and not (cf_acc and cf_tok)) or len(candidates) <= target_keep:
        return candidates[:target_keep]

    short_list = [
        {
            "id": idx,
            "title": c["title"],
            "source": c["source"],
            "cat": c["category"],
            "snippet": (c.get("summary") or "")[:240],
        }
        for idx, c in enumerate(candidates[:50])
    ]
    topic_hint = f"Target topic requested by user: '{custom_topic}'. " if custom_topic else ""
    sys_p = (
        "You are an elite science news scout for Iran Bioinformatics Academy. "
        "Select the most viral, impactful, story-worthy breakthroughs in bioinformatics, AI in biology/medicine, "
        "genomics, drug discovery, neuroscience, or major AI models. Skip dry incremental benchmark papers unless groundbreaking. "
        'Return ONLY valid JSON: {"top_ids": [int, ...]}.'
    )
    usr_p = f"{topic_hint}Pick the best {target_keep} article IDs from:\n{json.dumps(short_list, ensure_ascii=False)}"

    providers = []
    if groq_key:
        providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "llama-3.3-70b-versatile", True))
        providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "llama-3.1-8b-instant", True))
    if cf_acc and cf_tok:
        cf_ep = f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1/chat/completions"
        providers.append((cf_ep, cf_tok, "@cf/meta/llama-3.3-70b-instruct-fp8-fast", False))
    if or_key:
        providers.append(("https://openrouter.ai/api/v1/chat/completions", or_key, "openrouter/free", False))

    for ep, key, model, jm in providers:
        res = call_openai_compat_json(ep, key, model, sys_p, usr_p, timeout=22, use_json_mode=jm)
        ids = res.get("top_ids") if isinstance(res, dict) else None
        if isinstance(ids, list) and ids:
            picked = []
            seen_idx = set()
            for i in ids:
                if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen_idx:
                    picked.append(candidates[i])
                    seen_idx.add(i)
            if picked:
                print(f"[INFO] Scout Agent ({model}) narrowed {len(short_list)} -> {len(picked)} stories.")
                return picked[:target_keep]

    return candidates[:target_keep]


def _strip_leading_emoji(text: str) -> str:
    # Remove accidental duplicate emoji(s) at the very start of hook_title
    return re.sub(r"^[\W_]*[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\s]+", "", (text or "").strip()).strip()


def validate_and_polish_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent 3 (Deterministic Quality & BiDi Guard): Fixes emojis, RTL BiDi, and formatting without altering Gemini's Persian prose."""
    cleaned: list[dict[str, Any]] = []
    for p in posts:
        item = dict(p)
        t_emoji = (item.get("title_emoji") or "🧬").strip().split()[0]
        hook = (item.get("hook_title") or "").strip()
        if hook.startswith(t_emoji):
            hook = _strip_leading_emoji(hook) or hook
        # Fix Telegram RTL BiDi inversion when hook_title starts with Latin chars (e.g., "AI ...")
        if hook and re.match(r"^[A-Za-z0-9]", hook):
            hook = "\u200f" + hook
        item["title_emoji"] = t_emoji
        item["hook_title"] = hook

        fixed_paras: list[str] = []
        for para in item.get("paragraphs") or []:
            txt = (para or "").strip()
            if not txt:
                continue
            # Collapse accidental doubled leading emoji like "👁️ 👁️ ..."
            txt = re.sub(r"^([\U0001F300-\U0001FAFF\u2600-\u27BF]\uFE0F?)\s+\1", r"\1", txt)
            fixed_paras.append(txt)
        item["paragraphs"] = fixed_paras
        item["quote_box"] = (item.get("quote_box") or "").strip()
        item["closing_question"] = _strip_leading_emoji((item.get("closing_question") or "").strip())
        cleaned.append(item)
    return cleaned


def call_gemini(
    candidates: list[dict[str, Any]],
    max_posts: int,
    custom_topic: str = "",
    custom_query: str = "",
) -> list[dict[str, Any]]:
    gemini_keys = get_gemini_keys()
    cf_acc = os.getenv("CF_ACCOUNT_ID", "").strip()
    cf_tok = os.getenv("CF_AI_TOKEN", "").strip()
    if not gemini_keys and not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GROQ_API_KEY") and not (cf_acc and cf_tok):
        raise RuntimeError("No AI API key configured (GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY / CF_AI_TOKEN).")

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

    system_instruction = """تو سردبیر ارشد رسانه «آکادمی بیوانفورماتیک ایران» (به مدیریت معین شیرین‌کام) هستی.
این خبرها قرار است مستقیم در کانال تلگرام و استوری اینستاگرام منتشر شوند؛ بنابراین باید فوق‌العاده جذاب، قلاب‌دار، داستانی، روان، غیرخشک و کنجکاوکننده باشند — طوری که هم دانشجو و هم علاقه‌مند علم با اشتیاق تا آخر بخواند، ولی هم‌زمان کاملاً دقیق و علمی باشد.

قوانین انتخاب خبر (بسیار مهم):
1. دنبال خبرهای «داستان‌دار و دهان‌بازکن» بگرد: کشف‌های پزشکی و درمانی شگفت‌انگیز، کاربردهای حیرت‌انگیز AI در تشخیص بیماری، کشف دارو، ژنتیک، پروتئین، مغز، سرطان، طول عمر، یا خبرهای بزرگ صنعت بیوتک و AI.
2. از مقالات خشک و صرفاً آماری/تست بنچمارک که هیچ داستان جذابی برای مخاطب عمومی-تخصصی ندارند بپرهیز (مگر اینکه بتوانی زاویه جذاب انسانی/علمی آن‌ها را نشان بدهی).
3. تعداد درخواستی را از بهترین‌ها انتخاب کن و تکراری نباشد.

الگوی دقیق نگارش (موبه‌مو مثل این سبک بنویس):
نمونه ۱:
title_emoji: "💵"
hook_title: "شرط‌بندی ۵ میلیارد دلاری روی آینده داروسازی با هوش مصنوعی!"
paragraphs:
  - "😉 هوش مصنوعی حالا به مرحله‌ای رسیده که می‌تواند <b>ساختار پروتئین‌های مرتبط با بیماری را تحلیل کند و برای ساخت دارو، مولکول‌های جدید طراحی کند.</b>"
  - "🚨 <b>شرکت Isomorphic Labs، زیرمجموعه Alphabet، با جذب ۵ میلیارد دلار سرمایه</b> می‌خواهد موتور <u>هوش مصنوعی خود به نام IsoDDE را توسعه دهد</u>؛ فناوری‌ای که هدفش متحول کردن فرایند کشف و طراحی دارو است."
  - "📣 این فناوری قرار است با شبیه‌سازی برهم‌کنش مولکول‌ها، مسیر پیدا کردن داروهای جدید را <b>سریع‌تر و کم‌هزینه‌تر کند.</b>"
quote_box: "⚠️ نکته حیرت‌انگیز: هنوز هیچ دارویی از این شرکت وارد آزمایش انسانی نشده، اما سرمایه‌گذاران روی آینده آن ۵ میلیارد دلار شرط بسته‌اند!"
closing_question: "آیا نسل بعدی داروها را هوش مصنوعی کشف خواهد کرد؟"

نمونه ۲:
title_emoji: "💊"
hook_title: "<u>ادعای GRAIL</u>: شناسایی بیش از ۵۰ سرطان با یک آزمایش خون!"
paragraphs:
  - "🔬 تصور کنید با یک آزمایش خون، بتوان نشانه‌های ده‌ها نوع سرطان را پیش از بروز علائم پیدا کرد! <b>شرکت GRAIL با توسعه تست Galleri و استفاده از هوش مصنوعی</b>، به دنبال تحقق همین ایده است."
  - "🧪 این تست به جای جست‌وجوی خود تومور، <b>ردپای DNA سرطانی را در خون شناسایی می‌کند؛ <u>حتی می‌تواند محل احتمالی سرطان را پیش‌بینی کند.</u></b>"
  - "🖨 اما در یک <b>کارآزمایی بزرگ با بیش از ۱۴۲ هزار نفر</b> در بریتانیا، این تست توانست برخی سرطان‌ها را زودتر شناسایی کند؛ <b>اما در مجموع فقط حدود یک‌سوم سرطان‌ها را پیدا کرد.</b>"
quote_box: "⚠️ حالا پرسش اصلی این است: آیا چنین آزمایشی می‌تواند آینده غربالگری سرطان را تغییر دهد یا هنوز برای استفاده عمومی زود است؟"
closing_question: ""

قوانین اکید نگارش:
- تیتر (hook_title) باید کوتاه، قلاب‌دار و کنجکاوکننده باشد (مثلاً با کلماتی مثل «ادعای عجیب...»، «کشف حیرت‌انگیز...»، عدد شوکه‌کننده یا پرسش جذاب؛ بدون دروغ).
- هر پست ۲ تا ۳ پاراگراف کوتاه و خوش‌خوان داشته باشد (هر پاراگراف ۱ تا ۲ جمله + یک ایموجی مرتبط در اولش). اصطلاحات فوق‌تخصصی انگلیسی را خیلی روان و قابل‌فهم برای مخاطب ایرانی توضیح بده (مثلاً به جای پپتید ۱۲-مر بگو «مولکولی که فقط ۱۲ واحد سازنده پروتئین دارد»).
- برای تأکید روی مهم‌ترین بخش جمله، داخل متن فقط از تگ‌های <b>...</b> (بولد) و <u>...</u> (زیرخط‌دار، برای نقطه‌ی عطف یا عدد شگفت‌انگیز) استفاده کن؛ زیاده‌روی نکن (در هر پاراگراف نهایتاً ۱ یا ۲ عبارت کوتاه).
- بخش quote_box (باکس نقل‌قول) بسیار مهم است: یک زاویه تامل‌برانگیز، نکته حیرت‌انگیز یا محدودیت واقعی و صادقانه پژوهش را بنویس (مثل: هنوز روی انسان آزمایش نشده / فقط روی موش بوده / پیش‌چاپ است و هنوز داوری نشده / برای استفاده عمومی زود است). با یک ایموجی مناسب مثل ⚠️ یا ⭐️ یا 👁 یا 🎁 شروع شود.
- اگر quote_box خودش سوال نیست و سوال جذابی به ذهن می‌رسد، در closing_question یک سوال کوتاه و چالشی بنویس (بدون ایموجی، خود سیستم ❓ می‌گذارد)؛ وگرنه خالی بگذار "".
- طول کل متن (تیتر + پاراگراف‌ها + نقل‌قول + سوال) ترجیحاً بین ۴۰۰ تا ۶۵۰ کاراکتر باشد تا به عنوان کپشن زیر عکس در تلگرام کامل جا شود.
- image_prompt: یک پرامپت انگلیسی دقیق و سینمایی برای تولید عکس خبری (عکاسی رئال و ژورنالیستی، نورپردازی طبیعی یا دراماتیک، صحنه مرتبط با موضوع خبر مثل پژوهشگر، بیمار، آزمایشگاه یا نمای علمی تمیز؛ ترکیب‌بندی افقی طوری که سوژه در دو سوم بالایی کادر باشد تا پایین تصویر خلوت بماند؛ **اکیداً بدون هیچ متن، نوشته، لوگو یا واترمارک**).
- **اکیداً ممنوع:** هرگز عدد، درصد، نام شرکت یا ادعایی را که در متن ورودی نیست از خودت نساز!

خروجی فقط JSON معتبر مطابق اسکیما باشد."""

    topic_note = ""
    if custom_topic or custom_query:
        topic_note = (
            f"درخواست اختصاصی کاربر: «{custom_query or custom_topic}»\n"
            f"موضوع دقیق هدف (بسیار حیاتی): «{custom_topic}»\n"
            f"- تمام {max_posts} پست انتخابی باید **مستقیماً درباره همین موضوعِ دقیق** باشند؛ اگر کاربر موضوع خاصی مثل «پاروویروس» یا «سندرم براون اسکوارد» خواسته، اکیداً سراغ موضوعات کلی‌تر (مثل ویروس‌های دیگر یا اخبار عمومی) نرو.\n"
            f"- اگر خبرها به صورت پژوهش یا گزارش موردی (Case Report) درباره همین موضوع بودند، آن‌ها را به زبان داستان‌دار، علمی و فوق‌العاده جذاب بنویس.\n"
        )

    user_prompt = (
        f"تاریخ امروز: {today_str}\n"
        f"حداکثر تعداد پست انتخابی در این نوبت: {max_posts}\n"
        f"{topic_note}\n"
        f"لیست اخبار و مقالات خام ورودی:\n{json.dumps(compact_articles, ensure_ascii=False)}"
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.45,
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
                                "title_emoji": {"type": "STRING"},
                                "hook_title": {"type": "STRING"},
                                "paragraphs": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"},
                                },
                                "quote_box": {"type": "STRING"},
                                "closing_question": {"type": "STRING"},
                                "image_prompt": {"type": "STRING"},
                            },
                            "required": [
                                "article_id",
                                "title_emoji",
                                "hook_title",
                                "paragraphs",
                                "quote_box",
                                "closing_question",
                                "image_prompt",
                            ],
                        },
                    }
                },
                "required": ["selected_posts"],
            },
        },
    }

    last_err = None
    if gemini_keys:
        for model in models_to_try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            model_payload = json.loads(json.dumps(payload))
            if any(tag in model for tag in ("3.8", "3.6", "3-flash")):
                model_payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
            for k_idx, api_key in enumerate(gemini_keys, 1):
                try:
                    print(f"[INFO] Calling Writer Agent (Gemini: {model}, key #{k_idx})")
                    resp = requests.post(
                        endpoint,
                        params={"key": api_key},
                        json=model_payload,
                        timeout=30,
                    )
                    if resp.status_code == 400 and "thinkingConfig" in model_payload["generationConfig"]:
                        model_payload["generationConfig"].pop("thinkingConfig", None)
                        resp = requests.post(endpoint, params={"key": api_key}, json=model_payload, timeout=30)
                    if resp.status_code in {404, 503}:
                        print(f"[WARN] Model {model} returned {resp.status_code} (skipping model): {resp.text[:200]}")
                        last_err = resp.text
                        break
                    if resp.status_code in {400, 403, 429}:
                        print(f"[WARN] Model {model} key #{k_idx} returned {resp.status_code}: {resp.text[:200]}")
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
                        p["og_image"] = art.get("og_image", "")
                        enriched_posts.append(p)
                    return enriched_posts[:max_posts]
                except Exception as exc:
                    print(f"[WARN] Gemini call failed on {model} (key #{k_idx}): {exc}")
                    last_err = str(exc)

    # Automatic Multi-Provider Failover: OpenRouter Free -> Groq -> Cloudflare Workers AI
    fallback_providers = []
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if or_key:
        fallback_providers.append(("https://openrouter.ai/api/v1/chat/completions", or_key, "openrouter/free", False))
    if groq_key:
        fallback_providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "openai/gpt-oss-120b", True))
        fallback_providers.append(("https://api.groq.com/openai/v1/chat/completions", groq_key, "llama-3.3-70b-versatile", True))
    if cf_acc and cf_tok:
        cf_ep = f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1/chat/completions"
        fallback_providers.append((cf_ep, cf_tok, "@cf/openai/gpt-oss-120b", False))
        fallback_providers.append((cf_ep, cf_tok, "@cf/meta/llama-3.3-70b-instruct-fp8-fast", False))

    json_hint = (
        '\nReturn ONLY valid JSON: {"selected_posts": [{"article_id": int, "title_emoji": str, '
        '"hook_title": str, "paragraphs": [str], "quote_box": str, "closing_question": str, "image_prompt": str}]}.'
    )
    for ep, key, model, jm in fallback_providers:
        print(f"[INFO] Writer Agent failover -> {model}")
        parsed = call_openai_compat_json(ep, key, model, system_instruction + json_hint, user_prompt, timeout=45, use_json_mode=jm)
        if parsed and isinstance(parsed.get("selected_posts"), list):
            id_map = {idx: item for idx, item in enumerate(candidates)}
            enriched_posts = []
            for p in parsed["selected_posts"]:
                art = id_map.get(p.get("article_id"))
                if not art:
                    continue
                p["url"] = art["url"]
                p["raw_title"] = art["title"]
                p["og_image"] = art.get("og_image", "")
                enriched_posts.append(p)
            if enriched_posts:
                return enriched_posts[:max_posts]

    raise RuntimeError(f"All AI Writer providers failed. Last error: {last_err}")


def sanitize_tg_inline_html(text: str) -> str:
    # Preserve only balanced <b>, </b>, <u>, </u> tags and escape everything else safely
    tokens: list[str] = []
    pattern = re.compile(r"(</?(?:b|u)>)", re.IGNORECASE)
    pos = 0
    open_b = False
    open_u = False

    for match in pattern.finditer(text or ""):
        if match.start() > pos:
            tokens.append(html.escape(text[pos:match.start()]))
        tag = match.group(1).lower()
        if tag == "<b>" and not open_b:
            tokens.append("<b>")
            open_b = True
        elif tag == "</b>" and open_b:
            tokens.append("</b>")
            open_b = False
        elif tag == "<u>" and not open_u:
            tokens.append("<u>")
            open_u = True
        elif tag == "</u>" and open_u:
            tokens.append("</u>")
            open_u = False
        pos = match.end()

    if pos < len(text or ""):
        tokens.append(html.escape(text[pos:]))
    if open_u:
        tokens.append("</u>")
    if open_b:
        tokens.append("</b>")
    return "".join(tokens).strip()


def format_telegram_html(post: dict[str, Any], post_number: int) -> str:
    title_emoji = html.escape((post.get("title_emoji") or "🧬").strip())
    hook_title = sanitize_tg_inline_html(post.get("hook_title") or "")
    paragraphs = [
        sanitize_tg_inline_html(p)
        for p in (post.get("paragraphs") or [])
        if p and p.strip()
    ]
    quote_box = sanitize_tg_inline_html(post.get("quote_box") or "")
    closing_q = sanitize_tg_inline_html(post.get("closing_question") or "")
    url = html.escape((post.get("url") or "").strip(), quote=True)

    parts = [f"{title_emoji} <b>{hook_title}</b>"]
    parts.extend(paragraphs[:3])
    if quote_box:
        parts.append(f"<blockquote>{quote_box}</blockquote>")
    if closing_q:
        parts.append(f"❓ <b>{closing_q}</b>")

    footer = (
        "⭕️ معین شیرین کام\n"
        f'🧷 <a href="{url}">Source</a>\n'
        '<a href="https://www.instagram.com/ir_bioinf_academy?igsh=MTl0ZGgyYzl4OW1uaQ==">Instagram</a> | '
        '<a href="https://www.linkedin.com/in/ir-bioinformatics-academy-888b02377?utm_source=share&amp;utm_campaign=share_via&amp;utm_content=profile&amp;utm_medium=android_app">LinkedIn</a> | '
        '<a href="https://t.me/ir_bioinf_group">Q &amp; A group</a>\n\n'
        "آکادمی بیوانفورماتیک ایران 🖥⭐️\n"
        "@ir_bioinformtic_academy"
    )
    parts.append(footer)
    return "\n\n".join(parts)


def generate_cf_image_bytes(image_prompt: str) -> bytes | None:
    account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
    ai_token = os.getenv("CF_AI_TOKEN", "").strip()
    if not account_id or not ai_token or not image_prompt:
        return None

    clean_prompt = (
        f"{image_prompt.strip()}, photorealistic editorial science photography, "
        "cinematic natural lighting, high detail, 16:9 widescreen composition, "
        "subject framed in upper two thirds, clean empty lower edge, "
        "strictly no text, no letters, no watermark, no logo"
    )[:1800]

    headers = {
        "Authorization": f"Bearer {ai_token}",
        "Content-Type": "application/json",
    }
    models = [
        ("@cf/black-forest-labs/flux-1-schnell", {"prompt": clean_prompt, "steps": 4}),
        ("@cf/stabilityai/stable-diffusion-xl-lightning", {"prompt": clean_prompt}),
    ]
    for model, body in models:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=35)
            if resp.status_code != 200:
                print(f"[WARN] CF AI {model} returned {resp.status_code}: {resp.text[:200]}")
                continue
            ctype = resp.headers.get("Content-Type", "").lower()
            if "application/json" in ctype:
                b64 = resp.json().get("result", {}).get("image")
                if b64:
                    return base64.b64decode(b64)
            elif "image/" in ctype or resp.content[:4] in {b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\x89PNG"}:
                return resp.content
        except Exception as exc:
            print(f"[WARN] CF AI {model} failed: {exc}")
    return None


def download_fallback_image_bytes(og_image_url: str) -> bytes | None:
    if not og_image_url:
        return None
    try:
        resp = requests.get(og_image_url, headers=REQUEST_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        im = Image.open(io.BytesIO(resp.content))
        if im.width < 450 or im.height < 250:
            return None
        return resp.content
    except Exception:
        return None


def apply_brand_overlay(raw_img_bytes: bytes) -> bytes:
    base = Image.open(io.BytesIO(raw_img_bytes)).convert("RGBA")
    fitted = ImageOps.fit(base, (1280, 720), method=Image.Resampling.LANCZOS, centering=(0.5, 0.35))
    if OVERLAY_PATH.exists():
        overlay = Image.open(OVERLAY_PATH).convert("RGBA")
        if overlay.size != (1280, 720):
            overlay = overlay.resize((1280, 720), Image.Resampling.LANCZOS)
        fitted = Image.alpha_composite(fitted, overlay)
    out = io.BytesIO()
    fitted.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def generate_branded_image(post: dict[str, Any]) -> bytes | None:
    raw_bytes = generate_cf_image_bytes(post.get("image_prompt") or "")
    if not raw_bytes:
        raw_bytes = download_fallback_image_bytes(post.get("og_image") or "")
    if not raw_bytes:
        return None
    try:
        return apply_brand_overlay(raw_bytes)
    except Exception as exc:
        print(f"[WARN] Brand overlay failed: {exc}")
        return raw_bytes


def visible_text_length(html_text: str) -> int:
    plain = re.sub(r"<[^>]+>", "", html_text or "")
    return len(html.unescape(plain))


def send_telegram_message(bot_token: str, chat_id: str, html_text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")


def send_telegram_photo(bot_token: str, chat_id: str, image_bytes: bytes, caption_html: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    if visible_text_length(caption_html) <= 1000:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption_html, "parse_mode": "HTML"},
            files={"photo": ("news.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 200:
            return
        print(f"[WARN] sendPhoto with full caption failed ({resp.status_code}), falling back to photo + message: {resp.text[:200]}")

    # Fallback if caption exceeds Telegram's 1024-char photo caption limit
    requests.post(
        url,
        data={"chat_id": chat_id},
        files={"photo": ("news.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    send_telegram_message(bot_token, chat_id, caption_html)


def get_target_chat_ids() -> list[str]:
    reply_chat_id = os.getenv("REPLY_CHAT_ID", "").strip()
    if reply_chat_id:
        return [cid.strip() for cid in reply_chat_id.split(",") if cid.strip()]
    raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return [cid.strip() for cid in raw.split(",") if cid.strip()]


def update_progress(
    bot_token: str,
    chat_ids: list[str],
    status_msg_id: str,
    percent: int,
    label: str,
    done: bool = False,
) -> None:
    if not bot_token or not status_msg_id or not chat_ids:
        return
    filled = max(0, min(10, round(percent / 10)))
    bar = "▓" * filled + "░" * (10 - filled)
    icon = "✅" if done else "⏳"
    text = f"{icon} <b>گزارش وضعیت ربات</b>\n\n<code>[{bar}] {to_persian_digits(percent)}٪</code>\n{html.escape(label)}"
    payload: dict[str, Any] = {
        "chat_id": chat_ids[0],
        "message_id": int(status_msg_id),
        "text": text,
        "parse_mode": "HTML",
    }
    game_url = os.getenv("GAME_URL", "").strip()
    if game_url and not done:
        target_cid = str(chat_ids[0]).strip()
        is_group = target_cid.startswith("-")
        btn = (
            {"text": "🧠 بازی تقویت حافظه تا آماده شدن خبرها", "url": game_url}
            if is_group
            else {"text": "🧠 بازی تقویت حافظه تا آماده شدن خبرها", "web_app": {"url": game_url}}
        )
        payload["reply_markup"] = {"inline_keyboard": [[btn]]}
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/editMessageText",
            json=payload,
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[WARN] update_progress HTTP {r.status_code}: {r.text[:180]}")
            if done and label.startswith("⚠️"):
                requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={"chat_id": chat_ids[0], "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
    except Exception as exc:
        print(f"[WARN] update_progress failed: {exc}")
        if done and label.startswith("⚠️"):
            try:
                requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={"chat_id": chat_ids[0], "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
            except Exception:
                pass


def main() -> None:
    t_start = time.time()
    dry_run = os.getenv("DRY_RUN", "false").strip().lower() in {"true", "1", "yes"}
    on_demand = os.getenv("ON_DEMAND", "false").strip().lower() in {"true", "1", "yes"}
    custom_topic = os.getenv("CUSTOM_TOPIC", "").strip()
    custom_query = os.getenv("CUSTOM_QUERY", "").strip()
    custom_count_raw = os.getenv("CUSTOM_COUNT", "").strip()
    status_msg_id = os.getenv("STATUS_MESSAGE_ID", "").strip()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat_ids = get_target_chat_ids()
    run_mode = os.getenv("MODE", "news").strip().lower()

    if run_mode in {"edu", "edit"}:
        from edu_studio import run_edu_studio
        fa_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
        count_norm = custom_count_raw.translate(fa_map)
        slide_count = int(count_norm) if (count_norm.isdigit() and 1 <= int(count_norm) <= 15) else 5
        edu_images = os.getenv("EDU_IMAGES", "false").strip().lower() in {"true", "1", "yes"}
        reply_context = os.getenv("REPLY_CONTEXT", "").strip()
        aspect_ratio = os.getenv("ASPECT_RATIO", "post").strip().lower() or "post"
        design_spec_raw = os.getenv("DESIGN_SPEC", "").strip()
        try:
            run_edu_studio(
                custom_topic=custom_topic,
                custom_query=custom_query,
                slide_count=slide_count,
                with_images=edu_images,
                reply_context=reply_context,
                edit_mode=(run_mode == "edit"),
                aspect_ratio=aspect_ratio,
                design_spec_raw=design_spec_raw,
                bot_token=bot_token,
                target_chat_ids=target_chat_ids,
                status_msg_id=status_msg_id,
                dry_run=dry_run,
                progress_fn=update_progress,
            )
        except Exception as exc:
            print(f"[ERROR] Edu/Edit Studio failed: {exc}")
            if not dry_run:
                update_progress(
                    bot_token,
                    target_chat_ids,
                    status_msg_id,
                    100,
                    f"⚠️ خطا در پردازش درخواست: {str(exc)[:220]}",
                    done=True,
                )
            raise
        return

    try:
        _run_news_pipeline(
            t_start=t_start,
            dry_run=dry_run,
            on_demand=on_demand,
            custom_topic=custom_topic,
            custom_query=custom_query,
            custom_count_raw=custom_count_raw,
            status_msg_id=status_msg_id,
            bot_token=bot_token,
            target_chat_ids=target_chat_ids,
        )
    except Exception as exc:
        print(f"[ERROR] News pipeline failed: {exc}")
        if on_demand and not dry_run:
            update_progress(
                bot_token,
                target_chat_ids,
                status_msg_id,
                100,
                f"⚠️ خطا در پردازش خبر: {str(exc)[:220]}",
                done=True,
            )
        raise


def _run_news_pipeline(
    *,
    t_start: float,
    dry_run: bool,
    on_demand: bool,
    custom_topic: str,
    custom_query: str,
    custom_count_raw: str,
    status_msg_id: str,
    bot_token: str,
    target_chat_ids: list[str],
) -> None:
    if on_demand and not dry_run:
        update_progress(bot_token, target_chat_ids, status_msg_id, 35, "در حال رصد ۴۰ پایگاه علمی و PubMed...")

    config = load_sources()
    settings = config.get("settings", {})

    max_posts = int(settings.get("max_posts_per_run", 5))
    if custom_count_raw.isdigit() and 1 <= int(custom_count_raw) <= 10:
        max_posts = int(custom_count_raw)

    max_candidates = int(settings.get("max_candidates_for_ai", 35))
    lookback_hours = int(settings.get("lookback_hours", 36))
    if on_demand:
        lookback_hours = max(lookback_hours, 72)
    retention_days = int(settings.get("state_retention_days", 21))
    fetch_body = bool(settings.get("fetch_article_body", True))
    priority_keywords = config.get("priority_keywords", {})

    state = load_state(retention_days=retention_days)
    raw_items: list[dict[str, Any]] = []

    enabled_feeds = [f for f in config.get("rss_feeds", []) if f.get("enabled", True)]
    enabled_pubmed = [q for q in config.get("pubmed_queries", []) if q.get("enabled", True)]

    is_specific_topic = False
    match_tokens: list[str] = []
    if on_demand and (custom_topic or custom_query):
        expanded = expand_on_demand_topic(custom_topic, custom_query)
        is_specific_topic = bool(expanded.get("is_specific"))
        primary_en = (expanded.get("primary_en") or custom_topic or "").strip()
        alt_en = (expanded.get("alt_en") or "").strip()
        primary_fa = (expanded.get("primary_fa") or "").strip()
        match_tokens = [str(t).lower().strip() for t in (expanded.get("match_tokens") or []) if len(str(t).strip()) >= 3]
        if primary_en:
            custom_topic = primary_en

        if is_specific_topic:
            # For specific topics, prioritize targeted searches instead of drowning in 800 unrelated general feeds
            enabled_feeds = []
            enabled_pubmed = []

        for q_str, hl, gl, ceid, w in [
            (primary_en, "en-US", "US", "US:en", 2.4),
            (alt_en, "en-US", "US", "US:en", 2.0),
            (primary_fa, "fa", "IR", "IR:fa", 2.2),
        ]:
            if q_str:
                enc = requests.utils.quote(q_str)
                enabled_feeds.append(
                    {
                        "name": f"Google News ({q_str})",
                        "category": "custom_topic",
                        "url": f"https://news.google.com/rss/search?q={enc}&hl={hl}&gl={gl}&ceid={ceid}",
                        "weight": w,
                        "enabled": True,
                    }
                )
        if primary_en:
            enabled_pubmed.append(
                {
                    "name": f"PubMed ({primary_en})",
                    "category": "custom_topic",
                    "term": primary_en,
                    "retmax": 15,
                    "weight": 2.0,
                    "enabled": True,
                }
            )
            raw_items.extend(fetch_openalex_works(primary_en, per_page=12, weight=1.9))

    print(f"[INFO] Fetching {len(enabled_feeds)} RSS feeds and {len(enabled_pubmed)} PubMed queries...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(fetch_single_rss, f) for f in enabled_feeds]
        for fut in as_completed(futures):
            raw_items.extend(fut.result())

    for q in enabled_pubmed:
        raw_items.extend(fetch_pubmed_query(q))
        time.sleep(0.4)

    print(f"[INFO +{time.time() - t_start:.1f}s] Total raw entries fetched: {len(raw_items)}")

    if is_specific_topic and match_tokens:
        folded_tokens = [_fold_accents(t) for t in match_tokens if len(t) >= 3]
        filtered_items = [
            it for it in raw_items
            if any(tok in _fold_accents(f"{it.get('title', '')} {it.get('summary', '')}") for tok in folded_tokens)
        ]
        if filtered_items:
            print(f"[INFO] Specific topic filter kept {len(filtered_items)}/{len(raw_items)} matching items.")
            raw_items = filtered_items

    # Progressive lookback for specific on-demand topics so rare syndromes/topics always gather rich candidates
    windows = [8760, 200000] if is_specific_topic else [lookback_hours]
    candidates: list[dict[str, Any]] = []
    for win_hours in windows:
        candidates = score_and_deduplicate(
            raw_items=raw_items,
            seen_map=state["seen"],
            lookback_hours=win_hours,
            priority_keywords=priority_keywords,
            custom_topic=custom_topic,
            allow_seen_fallback=on_demand,
        )
        if len(candidates) >= max(max_posts * 3, 10):
            break

    # If still fewer than max_posts for a specific on-demand search, allow previously-seen items too
    if is_specific_topic and len(candidates) < max_posts and raw_items:
        candidates = score_and_deduplicate(
            raw_items=raw_items,
            seen_map={},
            lookback_hours=200000,
            priority_keywords=priority_keywords,
            custom_topic=custom_topic,
            allow_seen_fallback=True,
        )
    print(f"[INFO +{time.time() - t_start:.1f}s] Fresh deduplicated candidates: {len(candidates)}")

    if not candidates:
        print("[INFO] No fresh news items found in this window. Exiting cleanly.")
        if on_demand and not dry_run:
            update_progress(
                bot_token,
                target_chat_ids,
                status_msg_id,
                100,
                "متأسفانه در ۷۲ ساعت گذشته خبر جدیدی مطابق درخواست شما پیدا نشد.",
                done=True,
            )
        save_state(state)
        return

    if on_demand and not dry_run:
        update_progress(
            bot_token,
            target_chat_ids,
            status_msg_id,
            45,
            f"تعداد {to_persian_digits(len(candidates))} خبر تازه یافت شد؛ ایجنت دیده‌بان در حال گلچین داغ‌ترین‌ها...",
        )

    scouted_candidates = run_scout_agent(
        candidates,
        target_keep=max(15, max_posts * 3),
        custom_topic=custom_topic,
    )
    top_candidates = scouted_candidates[:max_candidates]

    if fetch_body:
        enrich_limit = min(15, len(top_candidates))
        print(f"[INFO +{time.time() - t_start:.1f}s] Enriching article excerpts for top {enrich_limit} scouted candidates...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(enrich_article_snippet, top_candidates[:enrich_limit]))

    has_any_ai = any(
        os.getenv(k, "").strip()
        for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CF_AI_TOKEN")
    )
    if dry_run and not has_any_ai:
        print("[DRY_RUN] No AI API key set; printing top 5 pre-scored candidates:")
        for i, c in enumerate(top_candidates[:5], 1):
            print(f"  {i}. [{c['pre_score']}] ({c['source']}) {c['title']} -> {c['url']}")
        return

    if on_demand and not dry_run:
        update_progress(
            bot_token,
            target_chat_ids,
            status_msg_id,
            60,
            "در حال نگارش قلاب‌دار و علمی خبرها توسط ایجنت نویسنده...",
        )

    selected_posts = call_gemini(
        top_candidates,
        max_posts=max_posts,
        custom_topic=custom_topic,
        custom_query=custom_query,
    )
    print(f"[INFO +{time.time() - t_start:.1f}s] Writer Agent produced {len(selected_posts)} posts.")

    if selected_posts:
        selected_posts = validate_and_polish_posts(selected_posts)

    if not selected_posts:
        print("[INFO] Gemini found no high-impact stories worth posting right now.")
        if on_demand and not dry_run:
            update_progress(
                bot_token,
                target_chat_ids,
                status_msg_id,
                100,
                "در حال حاضر خبر شاخص و جدیدی درباره این موضوع پیدا نشد.",
                done=True,
            )
        now_iso = datetime.now(timezone.utc).isoformat()
        for c in top_candidates[:15]:
            state["seen"][c["url"]] = {"title": c["title"], "ts": now_iso, "status": "skipped"}
        save_state(state)
        return

    if not dry_run and (not bot_token or not target_chat_ids):
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

    if on_demand and not dry_run:
        update_progress(
            bot_token,
            target_chat_ids,
            status_msg_id,
            75,
            f"متن {to_persian_digits(len(selected_posts))} خبر آماده شد؛ در حال ساخت هم‌زمان عکس‌های اختصاصی...",
        )

    images_list: list[bytes | None] = [None] * len(selected_posts)
    if not dry_run:
        with ThreadPoolExecutor(max_workers=min(5, len(selected_posts))) as executor:
            images_list = list(executor.map(generate_branded_image, selected_posts))

    if on_demand and not dry_run:
        update_progress(
            bot_token,
            target_chat_ids,
            status_msg_id,
            92,
            "عکس‌ها و لوگو آماده شدند؛ در حال ارسال به تلگرام...",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    sent_urls = set()

    for idx, post in enumerate(selected_posts):
        state["daily_post_count"] += 1
        msg_html = format_telegram_html(post, state["daily_post_count"])
        img_bytes = images_list[idx]
        if dry_run:
            print("\n" + "=" * 50)
            print(msg_html)
            print(f"[IMAGE PROMPT]: {post.get('image_prompt', '')}")
        else:
            for cid in target_chat_ids:
                try:
                    if img_bytes:
                        send_telegram_photo(bot_token, cid, img_bytes, msg_html)
                    else:
                        send_telegram_message(bot_token, cid, msg_html)
                except Exception as exc:
                    print(f"[WARN] Failed sending to {cid}: {exc}")
                time.sleep(0.5)
            time.sleep(0.6)
        sent_urls.add(post["url"])
        state["seen"][post["url"]] = {
            "title": post.get("raw_title") or post.get("hook_title", ""),
            "ts": now_iso,
            "status": "sent",
        }

    if on_demand and not dry_run:
        update_progress(
            bot_token,
            target_chat_ids,
            status_msg_id,
            100,
            f"انجام شد! {to_persian_digits(len(selected_posts))} خبر با عکس اختصاصی ارسال شد.",
            done=True,
        )

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
