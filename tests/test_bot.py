import dataclasses
import io
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import bot


@dataclasses.dataclass
class NewsItem:
    source: str
    title: str
    url: str
    published: str | None = None
    summary: str = "summary"


class FakeSources:
    NewsItem = NewsItem

    def __init__(self):
        self.observations = ({}, {})
        self.fetches = []
        self.fetch_error = None

    def collect(self):
        return self.observations

    def get_article_text(self, item):
        self.fetches.append(item.url)
        if self.fetch_error:
            raise self.fetch_error
        return "article text"


class TelegramError(Exception):
    def __init__(self, *, uncertain=False, status_code=429, retry_after=None):
        self.uncertain = uncertain
        self.status_code = status_code
        self.retry_after = retry_after


class FakeClients:
    TelegramError = TelegramError

    def __init__(self):
        self.analysis_calls = []
        self.send_calls = []
        self.analysis_error = None
        self.send_error = None

    def summarize(self, item, article_text, recent_events, api_key, model):
        self.analysis_calls.append((item, list(recent_events)))
        if self.analysis_error:
            raise self.analysis_error
        match = next((event for event in recent_events if event["semantic_event"] == item.title), None)
        return {"is_gaming_news": True, "is_duplicate": match is not None,
                "duplicate_of": match["id"] if match else None,
                "title": item.title, "paragraph1": "first", "paragraph2": "second",
                "semantic_event": item.title}

    def send_message(self, token, chat, title, paragraph1, paragraph2):
        self.send_calls.append(title)
        if self.send_error:
            raise self.send_error
        return {"message_id": len(self.send_calls)}


class BotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "state.sqlite3")
        self.config = bot.Config("secret-key", "model", "secret-token", "@test", self.path)
        self.db = bot.open_database(self.path)
        self.sources = FakeSources()
        self.clients = FakeClients()
        self.app = bot.Orchestrator(self.db, self.config, self.sources, self.clients, max_items=10)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def set_feeds(self, feeds, errors=None):
        self.sources.observations = (feeds, errors or {})

    def item(self, site, title, path=None, published=None):
        return NewsItem(site, title, "https://" + site + ".example/" + (path or title), published)

    def rows(self):
        return self.db.execute("SELECT * FROM articles ORDER BY id").fetchall()

    def state(self, title):
        return self.db.execute("SELECT status FROM articles WHERE title=?", (title,)).fetchone()[0]

    def baseline(self, *sites):
        self.set_feeds({s: [self.item(s, "old-" + s)] for s in sites})
        self.app.tick()

    def reopen(self):
        self.db.close()
        self.db = bot.open_database(self.path)
        self.app = bot.Orchestrator(self.db, self.config, self.sources, self.clients, max_items=10)

    def test_independent_baselines_failures_and_freshness(self):
        a0 = self.item("a", "old-a")
        self.set_feeds({"a": [a0], "b": [self.item("b", "unusable")]}, {"b": "secret error"})
        self.app.tick()
        self.assertEqual(self.db.execute("SELECT name FROM sources").fetchall()[0][0], "a")
        self.assertEqual(self.clients.send_calls, [])
        # b's very first successful snapshot is baseline, not a backlog to send.
        self.set_feeds({"a": [a0, self.item("a", "new-a")],
                        "b": [self.item("b", "old-b"), self.item("b", "old-b2")]})
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["new-a"])
        self.assertEqual(self.state("old-b2"), "baseline")
        # Old publications discovered only after baseline are not fresh either.
        old_timestamp = "2000-01-01T00:00:00+00:00"
        self.set_feeds({"b": [self.item("b", "backdated", published=old_timestamp), self.item("b", "new-b")]})
        self.app.tick()
        self.assertEqual(self.state("backdated"), "baseline")
        self.assertEqual(self.clients.send_calls, ["new-a", "new-b"])

    def test_all_discoveries_survive_rollover_and_reopen(self):
        self.baseline("a")
        self.app.max_items = 1
        self.set_feeds({"a": [self.item("a", x) for x in ("one", "two", "three")]})
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["one"])
        self.assertEqual([r["status"] for r in self.rows()][1:], ["sent", "pending", "pending"])
        self.reopen()
        self.set_feeds({"a": []})  # the feed has rolled over
        self.app.tick()
        self.assertCountEqual(self.clients.send_calls, ["one", "two", "three"])
        self.assertTrue(all(self.state(x) == "sent" for x in ("one", "two", "three")))

    def test_url_and_cross_site_semantic_dedup_only_sent_events(self):
        self.baseline("a", "b")
        first = NewsItem("a", "same event", "https://a.example/story/?utm_source=feed#frag")
        url_duplicate = NewsItem("a", "other title", "HTTP://A.EXAMPLE:80/story?fbclid=xyz")
        cross_site = self.item("b", "same event", "different")
        self.set_feeds({"a": [first, url_duplicate], "b": [cross_site]})
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["same event"])
        self.assertEqual(len(self.rows()), 4)  # two baseline and two distinct URLs
        statuses = [r["status"] for r in self.rows()][-2:]
        self.assertEqual(statuses, ["sent", "duplicate"])
        events = self.clients.analysis_calls[-1][1]
        self.assertEqual(events[0]["semantic_event"], "same event")
        self.assertEqual(len(events), 1)
        self.assertNotIn("url", events[0])
        self.app.tick()
        self.assertEqual(len(self.clients.send_calls), 1)

    def test_failed_analysis_retries_and_is_not_marked_processed(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "new")]})
        self.clients.analysis_error = ValueError("secret-key https://x/?token=secret")
        with self.assertLogs("pixelbot", level="WARNING") as logs:
            self.app.tick()
        self.assertNotIn("secret", str(logs.output))
        self.assertEqual(self.state("new"), "pending")
        self.assertEqual(self.clients.send_calls, [])
        self.clients.analysis_error = None
        self.set_feeds({"a": []})
        self.app.tick()
        self.assertEqual(self.state("new"), "sent")
        self.assertEqual(len(self.clients.analysis_calls), 2)

    def test_unknown_timeout_never_retried_even_after_restart(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        self.clients.send_error = TimeoutError("https://api.telegram.org/botsecret-token/sendMessage")
        with self.assertLogs("pixelbot", level="ERROR") as logs:
            self.app.tick()
        self.assertEqual(self.state("news"), "unknown")
        self.assertNotIn("secret", str(logs.output))
        self.assertNotIn("http", str(logs.output))
        self.reopen()
        self.assertEqual(self.app.recover_unknown(), 0)
        self.clients.send_error = None
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["news"])
        self.assertEqual(self.state("news"), "unknown")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM sent_events").fetchone()[0], 0)

    def test_late_success_after_local_send_timeout_is_not_retried(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        release = threading.Event()
        finished = threading.Event()
        def slow_send(*args):
            self.clients.send_calls.append("news")
            release.wait(1)
            finished.set()
            return {"message_id": 123}
        self.clients.send_message = slow_send
        try:
            with mock.patch.object(bot, "SEND_TIMEOUT", 0.02):
                self.app.tick()
            self.assertEqual(self.state("news"), "unknown")
            release.set()
            self.assertTrue(finished.wait(1))
            self.app.tick()
            self.assertEqual(self.state("news"), "unknown")
            self.assertEqual(self.clients.send_calls, ["news"])
        finally:
            release.set()

    def test_definite_rejection_can_retry_but_is_rechecked_after_other_sent(self):
        self.baseline("a", "b")
        self.set_feeds({"a": [self.item("a", "same")]})
        self.clients.send_error = TelegramError(status_code=429, retry_after=600)
        self.app.tick()
        self.assertEqual(self.state("same"), "ready")
        row = self.db.execute("SELECT next_attempt FROM articles WHERE title='same'").fetchone()
        self.assertGreater(row[0], time.time() + 500)
        # While A is waiting, B reports and sends the same event.
        self.clients.send_error = None
        self.set_feeds({"b": [self.item("b", "same", "another-url")]})
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["same", "same"])
        self.db.execute("UPDATE articles SET next_attempt=0 WHERE status='ready'")
        self.app.tick()
        statuses = [r[0] for r in self.db.execute("SELECT status FROM articles WHERE title='same' ORDER BY id")]
        self.assertEqual(statuses, ["duplicate", "sent"])
        self.assertEqual(len(self.clients.send_calls), 2)  # no third send

    def test_unknown_pauses_other_sends_then_confirmed_delivery_dedups(self):
        self.baseline("a", "b")
        self.set_feeds({"a": [self.item("a", "same")], "b": [self.item("b", "same", "other")]})
        self.clients.send_error = TimeoutError("untrusted network error")
        self.app.tick()
        statuses = [r[0] for r in self.db.execute("SELECT status FROM articles WHERE title='same' ORDER BY id")]
        self.assertEqual(statuses, ["unknown", "ready"])
        self.assertEqual(self.clients.send_calls, ["same"])
        unknown_id = self.db.execute("SELECT id FROM articles WHERE status='unknown'").fetchone()[0]
        bot.resolve_unknown(self.db, unknown_id, delivered=True)
        self.clients.send_error = None
        self.app.tick()
        statuses = [r[0] for r in self.db.execute("SELECT status FROM articles WHERE title='same' ORDER BY id")]
        self.assertEqual(statuses, ["sent", "duplicate"])
        self.assertEqual(self.clients.send_calls, ["same"])
        self.assertEqual(self.db.execute("SELECT resolution FROM delivery_resolutions").fetchone()[0], "confirmed_delivered")

    def test_cli_operator_can_confirm_not_delivered_and_requeue(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        self.clients.send_error = TimeoutError("unknown")
        self.app.tick()
        unknown_id = self.db.execute("SELECT id FROM articles WHERE status='unknown'").fetchone()[0]
        self.db.close()
        try:
            with mock.patch.dict(os.environ, {"DB_PATH": self.path}, clear=True):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(bot.main(["--list-unknown"]), 0)
                self.assertEqual(output.getvalue().strip(), str(unknown_id))
                self.assertEqual(bot.main(["--resolve-unknown", str(unknown_id), "--not-delivered"]), 0)
        finally:
            self.db = bot.open_database(self.path)
            self.app = bot.Orchestrator(self.db, self.config, self.sources, self.clients, max_items=10)
        self.assertEqual(self.state("news"), "ready")
        self.clients.send_error = None
        self.app.tick()
        self.assertEqual(self.state("news"), "sent")
        self.assertEqual(len(self.clients.send_calls), 2)
        self.assertEqual(self.db.execute("SELECT resolution FROM delivery_resolutions").fetchone()[0], "confirmed_not_delivered")

    def test_interrupted_sending_recovered_without_repost(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        self.app.tick()
        self.db.execute("DELETE FROM sent_events")
        self.db.execute("UPDATE articles SET status='sending' WHERE title='news'")
        self.reopen()
        with self.assertLogs("pixelbot", level="ERROR"):
            self.assertEqual(self.app.recover_unknown(), 1)
        self.app.tick()
        self.assertEqual(self.state("news"), "unknown")
        self.assertEqual(self.clients.send_calls, ["news"])

    def test_sql_failure_before_post_prevents_send(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        self.db.execute("""CREATE TRIGGER fail_sending BEFORE UPDATE OF status ON articles
                           WHEN NEW.status='sending' BEGIN SELECT RAISE(ABORT,'write failed'); END""")
        with self.assertRaises(sqlite3.Error):
            self.app.tick()
        self.assertEqual(self.clients.send_calls, [])
        self.assertEqual(self.state("news"), "ready")

    def test_sql_failure_after_post_stays_sending_until_unknown_recovery(self):
        self.baseline("a")
        self.set_feeds({"a": [self.item("a", "news")]})
        self.db.execute("""CREATE TRIGGER fail_sent BEFORE UPDATE OF status ON articles
                           WHEN NEW.status='sent' BEGIN SELECT RAISE(ABORT,'write failed'); END""")
        with self.assertRaises(sqlite3.Error):
            self.app.tick()
        self.assertEqual(self.clients.send_calls, ["news"])
        self.assertEqual(self.state("news"), "sending")
        self.reopen()
        with self.assertLogs("pixelbot", level="ERROR"):
            self.assertEqual(self.app.recover_unknown(), 1)
        self.app.tick()
        self.assertEqual(self.state("news"), "unknown")
        self.assertEqual(self.clients.send_calls, ["news"])

    def test_baseline_transaction_rolls_back_on_db_error(self):
        self.db.execute("""CREATE TRIGGER fail_source BEFORE INSERT ON sources
                           BEGIN SELECT RAISE(ABORT,'write failed'); END""")
        self.set_feeds({"a": [self.item("a", "old")]})
        with self.assertRaises(sqlite3.Error):
            self.app.tick()
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 0)
        self.assertEqual(self.clients.send_calls, [])

    def test_dry_run_uses_snapshot_and_never_sends_or_consumes_queue(self):
        self.baseline("a")
        dry = bot.open_database(self.path, dry_run=True)
        cfg = dataclasses.replace(self.config, dry_run=True, bot_token="")
        self.set_feeds({"a": [self.item("a", "new")]})
        try:
            simulation = bot.Orchestrator(dry, cfg, self.sources, self.clients, max_items=10)
            simulation.tick()
            self.assertEqual(dry.execute("SELECT status FROM articles WHERE title='new'").fetchone()[0], "ready")
            self.assertEqual(self.clients.send_calls, [])
            self.assertIsNone(self.db.execute("SELECT 1 FROM articles WHERE title='new'").fetchone())
        finally:
            dry.close()
        self.app.tick()
        self.assertEqual(self.clients.send_calls, ["new"])

    def test_bounded_tick_when_source_call_hangs(self):
        def slow_collect():
            time.sleep(0.15)
            return {}, {}
        self.sources.collect = slow_collect
        self.app.max_tick_seconds = 0.03
        start = time.monotonic()
        self.app.tick()
        self.assertLess(time.monotonic() - start, 0.12)
        self.assertEqual(self.clients.send_calls, [])

    def test_config_fail_closed_and_url_normalization(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                bot.Config.from_env()
        with mock.patch.dict(os.environ, {"COARENA_API_KEY": "k", "COARENA_MODEL": "m"}, clear=True):
            self.assertEqual(bot.Config.from_env(dry_run=True).chat_id, "@pixelmediair")
            with self.assertRaises(ValueError):
                bot.Config.from_env()
        self.assertEqual(bot.canonical_url("HTTP://Example.COM:80/a/%7e/?b=2&utm_x=1&a=1#x"),
                         "https://example.com/a/~?a=1&b=2")
        self.assertNotEqual(bot.canonical_url("https://example.com/a?id=1"),
                            bot.canonical_url("https://example.com/a?id=2"))


if __name__ == "__main__":
    unittest.main()
