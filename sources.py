"""Public, robots-aware gaming-news sources for the Telegram bot.

No website APIs or authenticated pages are used. Source names are stable lowercase
identifiers. Dates are returned as supplied by the publisher, when available.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    published: str | None = None
    summary: str = ""


USER_AGENT = "PixelBot/1.0"
TIMEOUT = (5, 15)  # connect and per-read timeouts, seconds
MAX_FEED_BYTES = 4 * 1024 * 1024
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROBOTS_BYTES = 256 * 1024
MAX_ITEMS = 30
MAX_AGE_DAYS = 30
MAX_ARTICLE_CHARS = 12000

_URLS = {
    "moonmedia": "https://moonmedia.ir/sitemap.xml",
    "gamefa": "https://gamefa.com/game/news/feed/",
    "zoomg": "https://www.zoomg.ir/rss/",
    "vigiato": "https://vigiato.net/c/news-articles/news-reports",
}
_HOSTS = {
    "moonmedia": {"moonmedia.ir", "www.moonmedia.ir"},
    "gamefa": {"gamefa.com", "www.gamefa.com"},
    "zoomg": {"zoomg.ir", "www.zoomg.ir"},
    "vigiato": {"vigiato.net", "www.vigiato.net"},
}
_REDIRECT_CODES = {301, 302, 303, 307, 308}


class _SourceError(Exception):
    """A deliberately non-sensitive, user-displayable source error."""


def _clean(value: str | None, limit: int = 2000) -> str:
    if not value:
        return ""
    # Feed descriptions can be HTML; do not pass markup to downstream models.
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _safe_url(url: str, source: str, base: str = "") -> str | None:
    """Resolve links only within the source's HTTPS hosts; never access APIs/login."""
    try:
        if not url or not url.strip():
            return None
        url = urldefrag(urljoin(base, url.strip()))[0]
        parts = urlsplit(url)
        if (parts.scheme.lower() != "https" or parts.hostname not in _HOSTS[source]
                or parts.username is not None or parts.password is not None
                or parts.port not in (None, 443)):
            return None
        # Decode twice to prevent encoded path separators from hiding a forbidden path.
        path = unquote(unquote(parts.path)).lower()
        if re.search(r"(?:^|/)(?:api|wp-json|wp-admin|wp-login\.php|login|account)(?:/|$)", path):
            return None
        return url
    except (ValueError, KeyError, AttributeError):
        return None


def _robots_rules(content: str) -> list[tuple[bool, str]]:
    """Select RFC 9309 user-agent groups, joining equally specific groups."""
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    seen_rule = False
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (x.strip() for x in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if seen_rule:
                groups.append((agents, rules))
                agents, rules, seen_rule = [], [], False
            agents.append(value.lower())
        elif key in ("allow", "disallow") and agents:
            seen_rule = True
            if value:
                rules.append((key == "allow", value))
    if agents:
        groups.append((agents, rules))
    name = USER_AGENT.split("/", 1)[0].lower()
    matches = []
    for group_agents, group_rules in groups:
        scores = [0 if agent == "*" else len(agent) for agent in group_agents
                  if agent == "*" or (agent and agent in name)]
        if scores:
            matches.append((max(scores), group_rules))
    if not matches:
        return []
    best = max(score for score, _ in matches)
    return [rule for score, group in matches if score == best for rule in group]


def _rule_allows(url: str, rules: list[tuple[bool, str]]) -> bool:
    parts = urlsplit(url)
    target = (parts.path or "/") + ("?" + parts.query if parts.query else "")
    # Check encoded and decoded forms conservatively.
    decisions = []
    for candidate in (target, unquote(target)):
        hits = []
        for allowed, pattern in rules:
            pattern = unquote(pattern)
            end = pattern.endswith("$")
            if end:
                pattern = pattern[:-1]
            expression = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + ("$" if end else "")
            if re.search(expression, candidate):
                hits.append((len(pattern.replace("*", "")), allowed))
        # Longest rule wins; Allow wins an equally long tie.
        decisions.append(max(hits)[1] if hits else True)
    return all(decisions)


class _Client:
    def __init__(self) -> None:
        self._robots: dict[tuple[str, str], list[tuple[bool, str]]] = {}

    @staticmethod
    def _request(url: str, limit: int) -> tuple[int, dict, bytes]:
        response = None
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                                    allow_redirects=False, stream=True)
            headers = response.headers
            status = response.status_code
            if status in _REDIRECT_CODES or status >= 400:
                return status, headers, b""
            declared = headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise _SourceError("response exceeds size limit")
            result = bytearray()
            for chunk in response.iter_content(chunk_size=16384):
                result.extend(chunk)
                if len(result) > limit:
                    raise _SourceError("response exceeds size limit")
            return status, headers, bytes(result)
        except requests.Timeout:
            raise _SourceError("request timed out") from None
        except requests.RequestException:
            raise _SourceError("network request failed") from None
        finally:
            if response is not None:
                response.close()

    def _rules(self, url: str, source: str) -> list[tuple[bool, str]]:
        p = urlsplit(url)
        key = (source, p.netloc.lower())
        if key in self._robots:
            return self._robots[key]
        robots = f"https://{p.netloc}/robots.txt"
        for _ in range(5):
            status, headers, body = self._request(robots, MAX_ROBOTS_BYTES)
            if status in _REDIRECT_CODES:
                target = _safe_url(headers.get("Location", ""), source, robots)
                if not target or urlsplit(target).path != "/robots.txt" or urlsplit(target).query:
                    raise _SourceError("robots.txt redirect not allowed")
                robots = target
                continue
            if status in (404, 410):
                rules = []
            elif status != 200:
                # In particular, 401/403/429 and server failures are not permission.
                raise _SourceError("robots.txt unavailable")
            else:
                rules = _robots_rules(body.decode("utf-8-sig", errors="replace"))
            self._robots[key] = rules
            return rules
        raise _SourceError("too many robots.txt redirects")

    def get(self, url: str, source: str, limit: int) -> tuple[bytes, str]:
        current = _safe_url(url, source)
        if not current:
            raise _SourceError("URL not allowed")
        for _ in range(6):
            if not _rule_allows(current, self._rules(current, source)):
                raise _SourceError("robots.txt disallows URL")
            status, headers, body = self._request(current, limit)
            if status in _REDIRECT_CODES:
                current = _safe_url(headers.get("Location", ""), source, current)
                if not current:
                    raise _SourceError("redirect not allowed")
                continue
            if status != 200:
                raise _SourceError(f"HTTP {status}")
            return body, current
        raise _SourceError("too many redirects")


def _xml(data: bytes) -> ET.Element:
    # Do not accept DTDs or entity declarations in untrusted publisher content.
    scan = data.replace(b"\x00", b"").upper()  # Also detect UTF-16/32 encodings.
    if b"<!DOCTYPE" in scan or b"<!ENTITY" in scan:
        raise _SourceError("invalid XML feed")
    try:
        return ET.fromstring(data)
    except (ET.ParseError, ValueError):
        raise _SourceError("invalid XML feed") from None


def _recent(date: str | None) -> bool:
    # Fail closed: an undated or malformed item cannot be proven fresh.
    if not isinstance(date, str) or not date.strip():
        return False
    try:
        try:
            parsed = datetime.fromisoformat(date.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = parsedate_to_datetime(date.strip())
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    except (ValueError, TypeError, OverflowError):
        # Never let an unparseable publisher date become an implicitly fresh post.
        return False


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    result, seen = [], set()
    for item in items:
        if item.url not in seen and item.title and _recent(item.published):
            result.append(item)
            seen.add(item.url)
            if len(result) >= MAX_ITEMS:
                break
    return result


def _rss(client: _Client, source: str) -> list[NewsItem]:
    data, base = client.get(_URLS[source], source, MAX_FEED_BYTES)
    root = _xml(data)
    if root.tag.lower() != "rss":
        raise _SourceError("invalid RSS feed")
    items = []
    for node in root.findall("./channel/item"):
        url = _safe_url(node.findtext("link") or "", source, base)
        if not url:
            continue
        path = urlsplit(url).path.lower()
        if source == "zoomg" and not path.startswith("/game-news/"):
            continue
        categories = [_clean(c.text, 100) for c in node.findall("category")]
        if source == "gamefa":
            # The feed is already the gaming-news category, but exclude explicit reviews.
            if any(re.search(r"نقد و بررسی|بررسی بازی|game reviews?", c, re.I) for c in categories):
                continue
        title = _clean(node.findtext("title"), 500)
        date = (node.findtext("pubDate") or "").strip() or None
        summary = _clean(node.findtext("description"))
        items.append(NewsItem(source, title, url, date, summary))
    return _dedupe(items)


def _moonmedia(client: _Client) -> list[NewsItem]:
    data, base = client.get(_URLS["moonmedia"], "moonmedia", MAX_FEED_BYTES)
    root = _xml(data)
    sitemap = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    news = "{http://www.google.com/schemas/sitemap-news/0.9}"
    if root.tag != sitemap + "urlset":
        raise _SourceError("invalid news sitemap")
    items = []
    for node in root.findall(sitemap + "url"):
        metadata = node.find(news + "news")
        if metadata is None:
            continue  # The sitemap contains reviews and other non-news posts too.
        url = _safe_url(node.findtext(sitemap + "loc") or "", "moonmedia", base)
        title = _clean(metadata.findtext(news + "title"), 500)
        if not url or not urlsplit(url).path.startswith("/post/"):
            continue
        # The news sitemap mixes films and games but exposes no category. Only
        # exclude explicit movie/TV/review terms; let the final classifier decide
        # ambiguous cases rather than inventing a category.
        if re.search(r"فیلم|سریال|انیمه|انیمیشن|نقد و بررسی|(?:^|[\s:])review(?:[\s:]|$)", title, re.I):
            continue
        date = (metadata.findtext(news + "publication_date") or "").strip() or None
        items.append(NewsItem("moonmedia", title, url, date))
    # The publisher orders the news sitemap newest first.
    return _dedupe(items)


def _vigiato(client: _Client) -> list[NewsItem]:
    data, base = client.get(_URLS["vigiato"], "vigiato", MAX_PAGE_BYTES)
    soup = BeautifulSoup(data, "html.parser")
    if not soup.select_one("main .rowCard, main .rowList, main #todaysNews"):
        raise _SourceError("invalid category page")
    items = []
    # Restrict extraction to the main category list; the navigation/sidebar also
    # contain unrelated articles and reviews.
    for card in soup.select("main .rowCard"):
        link = card.select_one(".rowCard__title[href]")
        if not link:
            continue
        url = _safe_url(link.get("href", ""), "vigiato", base)
        if not url or not re.match(r"^/(?:r|p)/[^/]+", urlsplit(url).path):
            continue
        date_node = card.select_one("time, .rowCard__date")
        date = (date_node.get("datetime") or date_node.get_text(" ", strip=True)) if date_node else None
        desc = card.select_one(".rowCard__description")
        items.append(NewsItem("vigiato", _clean(link.get_text(" ", strip=True), 500), url,
                              date or None, _clean(desc.get_text(" ", strip=True)) if desc else ""))
    return _dedupe(items)


def collect() -> tuple[dict[str, list[NewsItem]], dict[str, str]]:
    """Collect independently; an empty but successful source remains in results.

    No exception details, response bodies, query parameters, or credentials are
    returned in the error mapping. No network request is made to disallowed URLs.
    """
    results: dict[str, list[NewsItem]] = {}
    errors: dict[str, str] = {}
    collectors = {
        "moonmedia": _moonmedia,
        "gamefa": lambda c: _rss(c, "gamefa"),
        "zoomg": lambda c: _rss(c, "zoomg"),
        "vigiato": _vigiato,
    }

    def run_source(entry):
        source, collector = entry
        try:
            return source, collector(_Client()), None
        except _SourceError as exc:
            return source, None, str(exc)
        except Exception:
            # Isolate sources without leaking an exception containing a URL/token.
            return source, None, "unexpected source error"

    with ThreadPoolExecutor(max_workers=len(collectors), thread_name_prefix="news-source") as pool:
        for source, items, error in pool.map(run_source, collectors.items()):
            if error is not None:
                errors[source] = error
            else:
                results[source] = items
    return results, errors


def get_article_text(item: NewsItem) -> str:
    """Return article body text up to 12,000 characters, or the supplied summary.

    Some publishers render the body only using a disallowed/private API. In that
    case the public HTML description is used if no feed summary was provided.
    """
    fallback = _clean(item.summary, MAX_ARTICLE_CHARS)
    if item.source not in _HOSTS or not _safe_url(item.url, item.source):
        return fallback
    try:
        data, _ = _Client().get(item.url, item.source, MAX_PAGE_BYTES)
        soup = BeautifulSoup(data, "html.parser")
        selectors = {
            "moonmedia": ("[itemprop='articleBody']", ".post-body", ".article-body", ".post-content"),
            "gamefa": (".post-content", ".entry-content", "[itemprop='articleBody']"),
            "zoomg": (".long-form", "[itemprop='articleBody']", ".article-body"),
            "vigiato": (".articleContent", ".entry-content", "[itemprop='articleBody']"),
        }
        body = next((node for selector in selectors[item.source]
                     if (node := soup.select_one(selector)) is not None), None)
        if body is not None:
            for node in body.select("script, style, nav, aside, header, footer, form, "
                                    ".comments, .comment, .related, .advertisement, .content-ads, "
                                    "[class*='social'], [class*='share'], [class*='related']"):
                node.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in body.select("p, blockquote")]
            text = "\n\n".join(p for p in paragraphs if p)
            if not text:
                text = body.get_text(" ", strip=True)
            if text:
                return text[:MAX_ARTICLE_CHARS]
        if fallback:
            return fallback
        meta = soup.select_one("meta[property='og:description'], meta[name='description']")
        return _clean(meta.get("content", ""), MAX_ARTICLE_CHARS) if meta else ""
    except Exception:
        # Article extraction is best effort; failures must not break the bot.
        return fallback
