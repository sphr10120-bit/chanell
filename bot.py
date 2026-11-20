"""Single-worker, durable Pixelbot orchestration.

Run ``python bot.py`` (or ``--once`` / ``--dry-run``).  A persistent Railway
volume must be mounted at /data (or set DB_PATH on a persistent volume); an
ephemeral filesystem cannot preserve deduplication across redeploys.  Run one
worker per database/volume.  The local advisory lock prevents a second worker
on the same filesystem; it is not a distributed lock across separate volumes.

Integration contract: sources.collect() returns (source->items, source->error),
sources.NewsItem accepts the five named fields, sources.get_article_text(item)
returns text, and clients.summarize/send_message have the documented signatures.
Summaries may be mappings or objects with named fields.  Only a recognized
clients.TelegramError with uncertain=False and a definite response is retried;
all other send failures are quarantined. New sends pause while any outcome is
unknown. Stop the worker, use --list-unknown and inspect Telegram, then use
--resolve-unknown ID with --delivered or --not-delivered only after confirming
the actual outcome; restart the worker afterward.

--dry-run works on an in-memory snapshot: it can call the AI, but sends no
messages and does not consume the real queue or establish persistent baselines.
No source or network work happens merely by importing this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import fcntl
import hashlib
import importlib
import logging
import os
import queue
import re
import signal
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

LOG = logging.getLogger("pixelbot")
INTERVAL_SECONDS = 300
MAX_TICK_SECONDS = 240
MAX_ITEMS_PER_TICK = 2
COLLECT_TIMEOUT = 75
ARTICLE_TIMEOUT = 20
AI_TIMEOUT = 100
SEND_TIMEOUT = 40
RECENT_EVENT_LIMIT = 30
# Timed-out daemon calls cannot be safely killed in Python.  Limit the number
# that can remain in flight rather than accumulating unbounded worker threads.
_CALL_SLOTS = threading.BoundedSemaphore(4)
_TRACKING_KEYS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid"}


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    model: str
    bot_token: str = field(repr=False)
    chat_id: str = "@pixelmediair"
    db_path: str = "/data/pixelbot.sqlite3"
    dry_run: bool = False

    @classmethod
    def from_env(cls, *, dry_run: bool = False) -> "Config":
        key = os.environ.get("COARENA_API_KEY", "").strip()
        model = os.environ.get("COARENA_MODEL", "").strip()
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat = os.environ.get("TELEGRAM_CHAT_ID", "@pixelmediair").strip()
        path = os.environ.get("DB_PATH", "/data/pixelbot.sqlite3").strip()
        if not key or not model or any(c in key or c in model for c in "\r\n"):
            raise ValueError("COARENA_API_KEY and COARENA_MODEL are required")
        if not dry_run and not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not chat or not path:
            raise ValueError("TELEGRAM_CHAT_ID and DB_PATH must not be empty")
        return cls(key, model, token, chat, path, dry_run)


class CallTimedOut(Exception):
    """A bounded call timed out, was interrupted, or had no worker capacity."""


def bounded_call(function, *args, timeout: float, stop_event: threading.Event):
    """Bound waiting time, including shutdown, without logging exception content.

    A timed-out send may continue in its daemon thread; the caller MUST treat
    its outcome as unknown. No thread in this function touches the database.
    """
    if timeout <= 0 or stop_event.is_set() or not _CALL_SLOTS.acquire(blocking=False):
        raise CallTimedOut()
    result = queue.Queue(maxsize=1)

    def worker():
        try:
            try:
                result.put((True, function(*args)))
            except BaseException as exc:
                result.put((False, exc))
        finally:
            _CALL_SLOTS.release()

    try:
        threading.Thread(target=worker, daemon=True, name="pixelbot-call").start()
    except BaseException:
        _CALL_SLOTS.release()
        raise
    deadline = time.monotonic() + timeout
    while True:
        if stop_event.is_set():
            raise CallTimedOut()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CallTimedOut()
        try:
            ok, value = result.get(timeout=min(remaining, 0.2))
            if ok:
                return value
            if isinstance(value, Exception):
                raise value
            raise CallTimedOut()
        except queue.Empty:
            continue


def canonical_url(value: str) -> str:
    """Canonicalize web URLs for exact dedup, retaining meaningful query keys."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid URL")
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("invalid URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL user info is not supported")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = "[" + host + "]"
    port = parts.port  # also validates malformed port
    if port and not ((parts.scheme.lower() == "http" and port == 80) or
                     (parts.scheme.lower() == "https" and port == 443)):
        host += ":" + str(port)
    # Treat HTTP and HTTPS as the same article on the same host.
    path = parts.path or "/"
    path = re.sub(r"%([0-9a-fA-F]{2})", lambda m: _normalize_escape(m.group(1)), path)
    if path != "/":
        path = path.rstrip("/") or "/"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS]
    query.sort()
    return urlunsplit(("https", host, path, urlencode(query, doseq=True), ""))


def _normalize_escape(digits: str) -> str:
    char = chr(int(digits, 16))
    return char if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~" else "%" + digits.upper()


def _published_value(value):
    if isinstance(value, dt.datetime):
        return value.isoformat(), "datetime"
    if isinstance(value, dt.date):
        return value.isoformat(), "date"
    return (str(value) if value is not None else ""), "str"


def _published_timestamp(value: str):
    if not value:
        return None
    try:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(value)
        if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _restore_published(value: str, kind: str):
    if kind == "datetime":
        return dt.datetime.fromisoformat(value)
    if kind == "date":
        return dt.date.fromisoformat(value)
    return value


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    baselined_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    url_key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published TEXT NOT NULL,
    published_kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('baseline','pending','ready','skipped','duplicate','sending','unknown','sent')),
    discovered_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt REAL NOT NULL DEFAULT 0,
    output_title TEXT,
    paragraph1 TEXT,
    paragraph2 TEXT,
    semantic_event TEXT,
    duplicate_of TEXT,
    checked_revision INTEGER,
    telegram_message_id TEXT
);
CREATE INDEX IF NOT EXISTS queue_idx ON articles(status, next_attempt, attempts, discovered_at);
CREATE TABLE IF NOT EXISTS sent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL UNIQUE REFERENCES articles(id),
    semantic_event TEXT NOT NULL,
    title TEXT NOT NULL,
    sent_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    resolution TEXT NOT NULL,
    resolved_at REAL NOT NULL
);
"""


def open_database(path: str, *, dry_run: bool = False) -> sqlite3.Connection:
    """Open a WAL database, or a read-only snapshot in memory for dry runs."""
    if dry_run:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        if os.path.exists(path):
            # A read-only connection avoids creating or altering the real DB.
            uri = "file:" + quote(os.path.abspath(path), safe="/") + "?mode=ro"
            source = sqlite3.connect(uri, uri=True)
            try:
                source.backup(conn)
            finally:
                source.close()
    else:
        absolute = os.path.abspath(path)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        conn = sqlite3.connect(absolute, timeout=10, isolation_level=None)
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            conn.close()
            raise sqlite3.OperationalError("WAL mode unavailable")
        conn.execute("PRAGMA synchronous=FULL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(SCHEMA)
    return conn


class Transaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.conn.rollback()
            return False
        try:
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return False


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _safe_error_detail(exc):
    """Keep useful exception context without logging URLs or credential-like values."""
    message = str(exc)
    message = re.sub(r"https?://\S+", "[URL redacted]", message, flags=re.IGNORECASE)
    message = re.sub(r"(?i)\b\S*(?:secret|credential|api[_ -]?key)\S*\b", "[credential redacted]", message)
    message = re.sub(r"(?i)\b(bearer|token|api[_ -]?key|secret)\s*[:=]?\s*\S+",
                     r"\1=[redacted]", message)
    message = re.sub(r"\b[A-Za-z0-9_-]{24,}\b", "[value redacted]", message)
    message = " ".join(message.split())[:300]
    return f"{type(exc).__name__}: {message or '[no message]'}"


def recover_interrupted_sends(conn):
    """Under the process lock, quarantine every interrupted POST."""
    with Transaction(conn):
        cursor = conn.execute("UPDATE articles SET status='unknown', updated_at=? WHERE status='sending'", (time.time(),))
        count = cursor.rowcount
    if count:
        LOG.error("ALERT: %d interrupted Telegram send(s) have unknown outcomes; reconcile manually", count)
    return count


def resolve_unknown(conn, article_id, *, delivered):
    """Record an operator-confirmed outcome; never infer it automatically."""
    with Transaction(conn):
        row = conn.execute("SELECT * FROM articles WHERE id=? AND status='unknown'", (article_id,)).fetchone()
        if row is None:
            raise ValueError("Article is not in unknown state")
        now = time.time()
        if delivered:
            if not row["semantic_event"] or not row["output_title"]:
                raise ValueError("Prepared event data is missing")
            conn.execute("UPDATE articles SET status='sent', updated_at=? WHERE id=?", (now, article_id))
            conn.execute("INSERT INTO sent_events(article_id,semantic_event,title,sent_at) VALUES (?,?,?,?)",
                         (article_id, row["semantic_event"], row["output_title"], now))
            resolution = "confirmed_delivered"
        else:
            conn.execute("UPDATE articles SET status='ready', next_attempt=0, updated_at=? WHERE id=?", (now, article_id))
            resolution = "confirmed_not_delivered"
        conn.execute("INSERT INTO delivery_resolutions(article_id,resolution,resolved_at) VALUES (?,?,?)",
                     (article_id, resolution, now))


class Orchestrator:
    def __init__(self, conn, config: Config, source_api=None, client_api=None,
                 stop_event=None, *, max_items=MAX_ITEMS_PER_TICK,
                 max_tick_seconds=MAX_TICK_SECONDS):
        self.db = conn
        self.config = config
        self.sources = source_api if source_api is not None else importlib.import_module("sources")
        self.clients = client_api if client_api is not None else importlib.import_module("clients")
        self.stop = stop_event if stop_event is not None else threading.Event()
        self.max_items = max_items
        self.max_tick_seconds = max_tick_seconds

    def recover_unknown(self):
        return recover_interrupted_sends(self.db)

    def _call(self, fn, *args, timeout, deadline):
        return bounded_call(fn, *args, timeout=min(timeout, deadline - time.monotonic()), stop_event=self.stop)

    def record_source(self, source: str, items):
        """Atomically persist a successful source snapshot and its baseline."""
        if not isinstance(source, str) or not source:
            return
        now = time.time()
        with Transaction(self.db):
            row = self.db.execute("SELECT baselined_at FROM sources WHERE name=?", (source,)).fetchone()
            baseline = row is None
            cutoff = now if baseline else row["baselined_at"]
            for item in items:
                try:
                    url = _field(item, "url")
                    key = hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()
                    item_source = _field(item, "source", source) or source
                    title = _field(item, "title", "") or ""
                    summary = _field(item, "summary", "") or ""
                    if not all(isinstance(v, str) for v in (item_source, title, summary)):
                        raise ValueError("invalid item fields")
                    published, published_kind = _published_value(_field(item, "published", ""))
                    timestamp = _published_timestamp(published)
                    # Known old posts discovered later are also baseline entries.
                    # Without a reliable timestamp, freshness means first seen after baseline.
                    # Feed timestamps commonly have only one-second precision.
                    old = timestamp is not None and timestamp < cutoff - 1
                    status = "baseline" if baseline or old else "pending"
                    self.db.execute("""INSERT OR IGNORE INTO articles
                        (url_key,source,title,url,published,published_kind,summary,status,discovered_at,updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (key, item_source, title, url, published, published_kind, summary, status, now, now))
                except (ValueError, TypeError, UnicodeError, AttributeError):
                    LOG.warning("Skipped a malformed feed item")
            if baseline:
                self.db.execute("INSERT INTO sources(name,baselined_at) VALUES (?,?)", (source, now))
                LOG.info("Initialized one source baseline")

    def _recent(self):
        rows = self.db.execute("SELECT id,semantic_event,title FROM sent_events ORDER BY id DESC LIMIT ?", (RECENT_EVENT_LIMIT,)).fetchall()
        return [{"id": str(row["id"]), "semantic_event": row["semantic_event"], "title": row["title"]} for row in rows]

    def _revision(self):
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM sent_events").fetchone()[0]

    def _retry_pending(self, article_id):
        with Transaction(self.db):
            self.db.execute("UPDATE articles SET attempts=attempts+1, next_attempt=?, updated_at=? WHERE id=? AND status='pending'",
                            (0, time.time(), article_id))

    def _item(self, row):
        return self.sources.NewsItem(source=row["source"], title=row["title"], url=row["url"],
                                     published=_restore_published(row["published"], row["published_kind"]),
                                     summary=row["summary"])

    def _analyze(self, row, deadline):
        article_id = row["id"]
        stage = "preparing article"
        try:
            item = self._item(row)
            stage = "fetching article text"
            text = self._call(self.sources.get_article_text, item, timeout=ARTICLE_TIMEOUT, deadline=deadline)
            if not isinstance(text, str):
                raise ValueError("invalid article text")
            recent = self._recent()
            revision = self._revision()
            stage = "CoArena summarization request"
            result = self._call(self.clients.summarize, item, text, recent,
                                self.config.api_key, self.config.model, timeout=AI_TIMEOUT, deadline=deadline)
            stage = "validating CoArena summary"
            gaming = _field(result, "is_gaming_news")
            duplicate = _field(result, "is_duplicate")
            duplicate_of = _field(result, "duplicate_of")
            if type(gaming) is not bool or type(duplicate) is not bool:
                raise ValueError("invalid decisions")
            if not gaming:
                status = "skipped"
            elif duplicate:
                if str(duplicate_of) not in {e["id"] for e in recent}:
                    raise ValueError("invalid duplicate reference")
                status = "duplicate"
            else:
                if duplicate_of is not None:
                    raise ValueError("inconsistent duplicate reference")
                status = "ready"
            fields = [_field(result, name, "") for name in ("title", "paragraph1", "paragraph2", "semantic_event")]
            if status == "ready" and not all(isinstance(v, str) and v.strip() for v in fields):
                raise ValueError("incomplete summary")
            fields = [v if isinstance(v, str) else "" for v in fields]
        except Exception as exc:
            LOG.warning("Article %d analysis failed during %s (%s); queued for retry",
                        article_id, stage, _safe_error_detail(exc))
            self._retry_pending(article_id)
            return False
        with Transaction(self.db):
            self.db.execute("""UPDATE articles SET status=?, output_title=?, paragraph1=?, paragraph2=?,
                semantic_event=?, duplicate_of=?, checked_revision=?, attempts=attempts+1, next_attempt=0, updated_at=?
                WHERE id=? AND status='pending'""",
                (status, *fields, str(duplicate_of) if duplicate_of is not None else None,
                 revision, time.time(), article_id))
        return status == "ready"

    def _definite_rejection(self, exc):
        error_type = getattr(self.clients, "TelegramError", None)
        status = getattr(exc, "status_code", None)
        return (isinstance(error_type, type) and isinstance(exc, error_type)
                and getattr(exc, "uncertain", True) is False
                and type(status) is int and (400 <= status < 500 or status == 200))

    def _send(self, row, deadline):
        article_id = row["id"]
        if self.config.dry_run:
            LOG.info("Dry run: article %d would be sent", article_id)
            return
        # An unknown delivery may already have covered the same event on a
        # different site. Sent-only semantic comparison cannot detect that;
        # pause all further sends until the operator reconciles the outcome.
        if self.db.execute("SELECT 1 FROM articles WHERE status IN ('unknown','sending') LIMIT 1").fetchone():
            LOG.error("ALERT: Telegram sends paused pending unknown-delivery reconciliation")
            return
        # A ready item can wait across ticks after an explicit rejection; if
        # another article was sent in the meantime it must be checked again.
        if row["checked_revision"] != self._revision():
            with Transaction(self.db):
                self.db.execute("UPDATE articles SET status='pending', next_attempt=0, updated_at=? WHERE id=? AND status='ready'", (time.time(), article_id))
            row = self.db.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
            if not self._analyze(row, deadline):
                return
            row = self.db.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
        # This commit MUST complete before calling Telegram.
        with Transaction(self.db):
            changed = self.db.execute("UPDATE articles SET status='sending', updated_at=? WHERE id=? AND status='ready'", (time.time(), article_id)).rowcount
        if changed != 1:
            return
        try:
            message = self._call(self.clients.send_message, self.config.bot_token, self.config.chat_id,
                                 row["output_title"], row["paragraph1"], row["paragraph2"],
                                 timeout=SEND_TIMEOUT, deadline=deadline)
        except Exception as exc:
            if self._definite_rejection(exc):
                retry_after = getattr(exc, "retry_after", None)
                delay = max(0, retry_after) if type(retry_after) is int else 0
                with Transaction(self.db):
                    self.db.execute("UPDATE articles SET status='ready', next_attempt=?, updated_at=? WHERE id=? AND status='sending'", (time.time() + delay, time.time(), article_id))
                LOG.error("Telegram explicitly rejected article %d; retry is deferred", article_id)
            else:
                with Transaction(self.db):
                    self.db.execute("UPDATE articles SET status='unknown', updated_at=? WHERE id=? AND status='sending'", (time.time(), article_id))
                LOG.error("ALERT: Telegram delivery of article %d is unknown; reconcile manually, do not retry blindly", article_id)
            return
        msg_id = _field(message, "message_id")
        msg_id = str(msg_id)[:100] if isinstance(msg_id, (str, int)) and not isinstance(msg_id, bool) else None
        # Atomic recording of both delivery and event: a failure leaves the
        # previously committed 'sending' state for recovery as unknown.
        with Transaction(self.db):
            self.db.execute("UPDATE articles SET status='sent', telegram_message_id=?, updated_at=? WHERE id=? AND status='sending'", (msg_id, time.time(), article_id))
            self.db.execute("INSERT INTO sent_events(article_id,semantic_event,title,sent_at) VALUES (?,?,?,?)",
                            (article_id, row["semantic_event"], row["output_title"], time.time()))
        LOG.info("Sent article %d", article_id)

    def tick(self):
        deadline = time.monotonic() + self.max_tick_seconds
        try:
            observations = self._call(self.sources.collect, timeout=COLLECT_TIMEOUT, deadline=deadline)
            if not isinstance(observations, (tuple, list)) or len(observations) != 2:
                raise ValueError("invalid collection result")
            feeds, errors = observations
            if not isinstance(feeds, Mapping) or not isinstance(errors, Mapping):
                raise ValueError("invalid collection result")
        except Exception:
            LOG.warning("Source collection failed or timed out; existing queue remains available")
            feeds, errors = {}, {}
        if errors:
            LOG.warning("%d source(s) failed; their baselines were not changed", len(errors))
        for source, items in feeds.items():
            if source in errors:
                continue
            # Commit each source independently, but abort on database errors.
            self.record_source(source, items)
        if self.stop.is_set() or time.monotonic() >= deadline:
            return
        rows = self.db.execute("""SELECT * FROM articles WHERE status IN ('pending','ready') AND next_attempt<=?
            ORDER BY attempts ASC, discovered_at ASC, id ASC LIMIT ?""", (time.time(), self.max_items)).fetchall()
        for row in rows:
            if self.stop.is_set() or time.monotonic() >= deadline:
                break
            if row["status"] == "pending":
                if not self._analyze(row, deadline):
                    continue
                row = self.db.execute("SELECT * FROM articles WHERE id=?", (row["id"],)).fetchone()
            self._send(row, deadline)


class InstanceLock:
    def __init__(self, path):
        self.path = os.path.abspath(path) + ".lock"
        self.handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another worker holds the database lock") from None
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Poll gaming news and post to Telegram")
    parser.add_argument("--once", action="store_true", help="run one polling tick")
    parser.add_argument("--dry-run", action="store_true", help="simulate on an in-memory database snapshot, never post")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--list-unknown", action="store_true", help="list article IDs needing reconciliation (stop worker first)")
    operation.add_argument("--resolve-unknown", type=int, metavar="ID", help="resolve an unknown article after checking Telegram (stop worker first)")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--delivered", action="store_true", help="confirm the unknown post was delivered")
    decision.add_argument("--not-delivered", action="store_true", help="confirm the unknown post was not delivered")
    args = parser.parse_args(argv)
    if args.resolve_unknown is not None:
        if args.resolve_unknown <= 0 or not (args.delivered or args.not_delivered):
            parser.error("--resolve-unknown requires a positive ID and an explicit delivery decision")
    elif args.delivered or args.not_delivered:
        parser.error("delivery decisions require --resolve-unknown")
    operator_action = args.list_unknown or args.resolve_unknown is not None
    if operator_action and (args.once or args.dry_run):
        parser.error("operator actions cannot be combined with --once or --dry-run")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if operator_action:
        db_path = os.environ.get("DB_PATH", "/data/pixelbot.sqlite3").strip()
        if not db_path:
            LOG.error("DB_PATH must not be empty")
            return 2
        try:
            with InstanceLock(db_path):
                conn = open_database(db_path)
                try:
                    recover_interrupted_sends(conn)
                    if args.list_unknown:
                        ids = conn.execute("SELECT id FROM articles WHERE status='unknown' ORDER BY id").fetchall()
                        for row in ids:
                            print(row["id"])
                        LOG.info("%d unknown delivery outcome(s)", len(ids))
                    else:
                        resolve_unknown(conn, args.resolve_unknown, delivered=args.delivered)
                        LOG.info("Recorded operator-confirmed delivery outcome for article %d", args.resolve_unknown)
                finally:
                    conn.close()
        except ValueError:
            LOG.error("Article cannot be reconciled; confirm its unknown state and prepared event")
            return 2
        except (sqlite3.Error, OSError, RuntimeError):
            LOG.error("Storage or lock failure; stop the worker and retry reconciliation")
            return 1
        return 0
    try:
        config = Config.from_env(dry_run=args.dry_run)
    except ValueError as exc:
        LOG.error("Configuration error: %s", exc)  # constant validation messages only
        return 2
    stop = threading.Event()
    def shutdown(_signum, _frame):
        stop.set()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        if not args.once and not args.dry_run:
            from wake import start_wake_server
            if start_wake_server(stop) is not None:
                LOG.info("Optional wake endpoint enabled")
        with InstanceLock(config.db_path):
            conn = open_database(config.db_path, dry_run=config.dry_run)
            try:
                bot = Orchestrator(conn, config, stop_event=stop)
                bot.recover_unknown()
                next_tick = time.monotonic()
                while not stop.is_set():
                    delay = max(0.0, next_tick - time.monotonic())
                    if stop.wait(delay):
                        break
                    bot.tick()
                    if args.once:
                        break
                    next_tick += INTERVAL_SECONDS
                    if next_tick <= time.monotonic():
                        # Skip missed slots; never rush to catch up after downtime.
                        next_tick = time.monotonic() + INTERVAL_SECONDS
            finally:
                conn.close()
    except (sqlite3.Error, OSError, RuntimeError):
        # Exception text may contain secrets/URLs. Stop rather than risking a
        # send without durable state; a restart quarantines interrupted sends.
        LOG.error("Storage, lock, or runtime failure; worker stopped safely")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
