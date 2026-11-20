import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources


ROBOTS_OK = b"User-agent: *\nDisallow: /api/\n"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
OLD = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index:index + chunk_size]

    def close(self):
        self.closed = True


def rss(items=""):
    return "<?xml version='1.0'?><rss version='2.0'><channel>" + items + "</channel></rss>"


def item(title, url, date=TODAY, categories="", description=""):
    return ("<item><title>" + title + "</title><link>" + url + "</link>"
            + "<pubDate>" + date + "</pubDate>" + categories
            + "<description>" + description + "</description></item>")


def sitemap(entries=""):
    return ("<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9' "
            "xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>" + entries + "</urlset>")


def moon_entry(title, slug, date=TODAY, news=True):
    n = ("<news:news><news:title>" + title + "</news:title>"
         + "<news:publication_date>" + date + "</news:publication_date></news:news>") if news else ""
    return "<url><loc>https://moonmedia.ir/post/" + slug + "</loc>" + n + "</url>"


def default_routes():
    return {
        "https://moonmedia.ir/robots.txt": Response(ROBOTS_OK),
        "https://gamefa.com/robots.txt": Response(ROBOTS_OK),
        "https://www.zoomg.ir/robots.txt": Response(ROBOTS_OK),
        "https://vigiato.net/robots.txt": Response(ROBOTS_OK),
        sources._URLS["moonmedia"]: Response(sitemap()),
        sources._URLS["gamefa"]: Response(rss()),
        sources._URLS["zoomg"]: Response(rss()),
        sources._URLS["vigiato"]: Response("<html><main><div class='rowList'></div></main></html>"),
    }


class Network:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url not in self.routes:
            raise AssertionError("Unexpected network access: " + url)
        result = self.routes[url]
        if isinstance(result, BaseException):
            raise result
        return result


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.routes = default_routes()
        self.network = Network(self.routes)
        self.patcher = patch.object(sources.requests, "get", self.network)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_empty_sources_are_successful_and_requests_are_bounded(self):
        result, errors = sources.collect()
        self.assertEqual(errors, {})
        self.assertEqual(result, {s: [] for s in sources._URLS})
        for _, options in self.network.calls:
            self.assertEqual(options["timeout"], (5, 15))
            self.assertFalse(options["allow_redirects"])
            self.assertTrue(options["stream"])
            self.assertEqual(options["headers"]["User-Agent"], sources.USER_AGENT)

    def test_collects_and_filters_all_four_sources(self):
        self.routes[sources._URLS["moonmedia"]] = Response(sitemap(
            moon_entry("خبر بازی Alpha", "alpha") + moon_entry("خبر فیلم سینمایی", "film")
            + moon_entry("نقد بازی", "review", news=False) + moon_entry("خبر قدیمی", "old", OLD)))
        self.routes[sources._URLS["gamefa"]] = Response(rss(
            item("خبر بازی", "https://gamefa.com/123/game/", categories="<category>اخبار بازی</category>",
                 description="&lt;p&gt;متن خلاصه&lt;/p&gt;")
            + item("نقد", "https://gamefa.com/124/review/", categories="<category>نقد و بررسی</category>")
            + item("قدیمی", "https://gamefa.com/125/old/", date=OLD)))
        self.routes[sources._URLS["zoomg"]] = Response(rss(
            item("خبر", "https://www.zoomg.ir/game-news/1-gaming/")
            + item("نقد", "https://www.zoomg.ir/game-review/2-review/")
            + item("فیلم", "https://www.zoomg.ir/movie-news/3-movie/")))
        self.routes[sources._URLS["vigiato"]] = Response("""
            <nav><a href='https://vigiato.net/r/review'>Unrelated review</a></nav><main>
            <div class='rowCard'><h2><a class='rowCard__title' href='/r/game-news'>اخبار بازی</a></h2>
            <p class='rowCard__description'>خلاصه</p><time datetime='2026-11-01T00:00:00Z'>recent</time></div>
            <aside><a href='/r/sidebar'>Sidebar</a></aside></main>""")
        # A time-independent recent date for the HTML fixture.
        self.routes[sources._URLS["vigiato"]].body = self.routes[sources._URLS["vigiato"]].body.replace(
            b"2026-11-01T00:00:00Z", TODAY.encode())
        result, errors = sources.collect()
        self.assertEqual(errors, {})
        self.assertEqual({k: len(v) for k, v in result.items()}, {k: 1 for k in sources._URLS})
        self.assertEqual(result["moonmedia"][0].published, TODAY)
        self.assertEqual(result["gamefa"][0].summary, "متن خلاصه")
        self.assertEqual(result["vigiato"][0].url, "https://vigiato.net/r/game-news")
        self.assertEqual(result["vigiato"][0].summary, "خلاصه")
        self.assertEqual(result["zoomg"][0].source, "zoomg")

    def test_failure_is_independent_and_error_never_exposes_secret(self):
        self.routes[sources._URLS["gamefa"]] = requests.Timeout("secret-token in https://example.test/?secret=x")
        result, errors = sources.collect()
        self.assertNotIn("gamefa", result)
        self.assertEqual(errors, {"gamefa": "request timed out"})
        self.assertEqual(set(result), {"moonmedia", "zoomg", "vigiato"})
        self.assertNotIn("secret", str(errors))

    def test_bad_xml_is_a_source_error(self):
        self.routes[sources._URLS["gamefa"]] = Response("<rss><broken>")
        results, errors = sources.collect()
        self.assertEqual(errors["gamefa"], "invalid XML feed")
        self.assertEqual(len(results), 3)

    def test_rejects_external_entities_and_doctype(self):
        self.routes[sources._URLS["gamefa"]] = Response('<!DOCTYPE rss [<!ENTITY x "boom">]><rss/>')
        self.assertEqual(sources.collect()[1]["gamefa"], "invalid XML feed")
        self.routes[sources._URLS["gamefa"]] = Response('<!DOCTYPE rss><rss/>'.encode('utf-16'))
        self.assertEqual(sources.collect()[1]["gamefa"], "invalid XML feed")

    def test_missing_feed_link_is_not_the_feed_itself(self):
        self.routes[sources._URLS["gamefa"]] = Response(rss("<item><title>Missing link</title></item>"))
        results, errors = sources.collect()
        self.assertEqual(errors, {})
        self.assertEqual(results["gamefa"], [])

    def test_challenge_page_is_not_reported_as_an_empty_category(self):
        self.routes[sources._URLS["vigiato"]] = Response("<html><body>Verification required</body></html>")
        results, errors = sources.collect()
        self.assertEqual(errors["vigiato"], "invalid category page")
        self.assertNotIn("vigiato", results)
        self.assertEqual(len(results), 3)

    def test_deduplicates_and_caps_feed(self):
        entries = "".join(item(str(i), f"https://gamefa.com/{i}/story/") for i in range(40))
        self.routes[sources._URLS["gamefa"]] = Response(rss(entries + item("duplicate", "https://gamefa.com/1/story/")))
        result, errors = sources.collect()
        self.assertEqual(errors, {})
        self.assertEqual(len(result["gamefa"]), 30)
        self.assertEqual(len({x.url for x in result["gamefa"]}), 30)

    def test_size_limit_during_stream(self):
        self.routes[sources._URLS["gamefa"]] = Response(b"x" * (sources.MAX_FEED_BYTES + 1))
        result, errors = sources.collect()
        self.assertEqual(errors["gamefa"], "response exceeds size limit")
        self.assertEqual(len(result), 3)

    def test_follows_only_robots_allowed_redirects(self):
        redirect = "https://www.zoomg.ir/feed/"
        self.routes[sources._URLS["zoomg"]] = Response(status=301, headers={"Location": redirect})
        self.routes[redirect] = Response(rss(item("news", "https://www.zoomg.ir/game-news/4-x/")))
        result, errors = sources.collect()
        self.assertEqual(errors, {})
        self.assertEqual(len(result["zoomg"]), 1)
        self.assertIn(redirect, [url for url, _ in self.network.calls])

    def test_redirect_to_disallowed_path_never_requested(self):
        self.routes[sources._URLS["zoomg"]] = Response(status=301, headers={"Location": "/api/private"})
        _, errors = sources.collect()
        self.assertEqual(errors["zoomg"], "redirect not allowed")
        self.assertNotIn("https://www.zoomg.ir/api/private", [x[0] for x in self.network.calls])

    def test_redirect_host_checks_its_own_robots_before_content(self):
        self.routes[sources._URLS["zoomg"]] = Response(status=301, headers={"Location": "https://zoomg.ir/feed/"})
        self.routes["https://zoomg.ir/robots.txt"] = Response("User-agent: *\nDisallow: /feed/")
        _, errors = sources.collect()
        self.assertEqual(errors["zoomg"], "robots.txt disallows URL")
        self.assertNotIn("https://zoomg.ir/feed/", [x[0] for x in self.network.calls])

    def test_robots_unavailable_fails_closed_without_fetching_feed(self):
        self.routes["https://gamefa.com/robots.txt"] = Response(status=403)
        result, errors = sources.collect()
        self.assertEqual(errors["gamefa"], "robots.txt unavailable")
        self.assertNotIn("gamefa", result)
        self.assertNotIn(sources._URLS["gamefa"], [x[0] for x in self.network.calls])

    def test_robot_wildcard_allow_and_specific_user_agent(self):
        self.routes["https://gamefa.com/robots.txt"] = Response("""User-agent: *
Disallow: /game/*
Allow: /game/news/feed/
User-agent: OtherBot
Disallow: /
""")
        self.assertNotIn("gamefa", sources.collect()[1])
        self.routes["https://gamefa.com/robots.txt"] = Response("User-agent: *\nAllow: /\nUser-agent: PixelBot\nDisallow: /game/")
        self.network.calls.clear()
        self.assertEqual(sources.collect()[1]["gamefa"], "robots.txt disallows URL")
        self.assertNotIn(sources._URLS["gamefa"], [x[0] for x in self.network.calls])

    def test_robots_rejects_query_wildcards(self):
        rules = sources._robots_rules("User-agent: *\nDisallow: *preview=true\nDisallow: /*?utm_*\n")
        self.assertFalse(sources._rule_allows("https://vigiato.net/p/1?preview=true", rules))
        self.assertFalse(sources._rule_allows("https://vigiato.net/p/1?utm_source=x", rules))
        self.assertTrue(sources._rule_allows("https://vigiato.net/p/1", rules))


class ArticleTests(unittest.TestCase):
    def setUp(self):
        self.routes = default_routes()
        self.network = Network(self.routes)
        self.patcher = patch.object(sources.requests, "get", self.network)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_real_body_excludes_navigation_related_and_ads(self):
        url = "https://gamefa.com/123/article/"
        self.routes[url] = Response("""<nav>MENU DO NOT INCLUDE</nav><main><div class='post-content'>
            <p>اولین پاراگراف بدنه مقاله.</p><div class='content-ads'><p>AD</p></div>
            <p>دومین پاراگراف واقعی مقاله.</p><div class='related'><p>RELATED</p></div>
            </div></main>""")
        text = sources.get_article_text(sources.NewsItem("gamefa", "title", url, summary="summary"))
        self.assertIn("اولین پاراگراف", text)
        self.assertIn("دومین پاراگراف", text)
        for word in ("MENU", "RELATED", "AD", "summary"):
            self.assertNotIn(word, text)

    def test_source_specific_body_selectors(self):
        examples = (("zoomg", "https://www.zoomg.ir/game-news/123-x/", "long-form"),
                    ("vigiato", "https://vigiato.net/r/x", "articleContent"),
                    ("moonmedia", "https://moonmedia.ir/post/x", "post-body"))
        for name, url, klass in examples:
            with self.subTest(source=name):
                self.routes[url] = Response(f"<nav>BAD</nav><div class='{klass}'><p>BODY {name}</p></div>")
                self.assertEqual(sources.get_article_text(sources.NewsItem(name, "x", url)), f"BODY {name}")

    def test_article_size_capped(self):
        url = "https://vigiato.net/r/long"
        self.routes[url] = Response("<div class='articleContent'><p>" + ("a" * 20000) + "</p></div>")
        self.assertEqual(len(sources.get_article_text(sources.NewsItem("vigiato", "t", url))), 12000)

    def test_missing_body_falls_back_to_summary_then_public_metadata(self):
        url = "https://moonmedia.ir/post/x"
        self.routes[url] = Response("<meta name='description' content='Public description'><nav>Navigation</nav>")
        self.assertEqual(sources.get_article_text(sources.NewsItem("moonmedia", "t", url, summary="Fallback")), "Fallback")
        self.assertEqual(sources.get_article_text(sources.NewsItem("moonmedia", "t", url)), "Public description")

    def test_fallback_on_robots_or_http_errors(self):
        url = "https://gamefa.com/123/a/"
        self.routes["https://gamefa.com/robots.txt"] = Response("User-agent: *\nDisallow: /123/")
        self.assertEqual(sources.get_article_text(sources.NewsItem("gamefa", "t", url, summary="Safe")), "Safe")
        self.assertNotIn(url, [x[0] for x in self.network.calls])

    def test_does_not_fetch_api_login_or_other_hosts(self):
        for url in ("https://moonmedia.ir/api/posts/1", "https://moonmedia.ir/%61pi/x",
                    "https://moonmedia.ir/login", "https://evil.test/article", "http://moonmedia.ir/post/x"):
            with self.subTest(url=url):
                self.assertEqual(sources.get_article_text(sources.NewsItem("moonmedia", "x", url, summary="safe")), "safe")
        self.assertEqual(self.network.calls, [])

    def test_fallback_is_bounded(self):
        item_ = sources.NewsItem("unknown", "t", "https://invalid/", summary="x" * 13000)
        self.assertEqual(len(sources.get_article_text(item_)), 12000)


if __name__ == "__main__":
    unittest.main()
