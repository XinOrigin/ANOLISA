"""Regression tests for HTTP error and empty-content handling in crawl_docs.py.

`crawl_docs.py` refreshes the local Markdown cache used by `anolisa-guide`
(`check_docs.py` runs it and inspects its exit status). Uses only the
Python standard library so `.github/workflows/os-skills-prebuilt-ci.yaml`
can execute it without external packages.
"""

from __future__ import annotations

import contextlib
import html.parser
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "others" / "anolisa-guide" / "scripts" / "crawl_docs.py"
)


class _FakeHTTPError(Exception):
    def __init__(self, message: str = "", response=None) -> None:
        super().__init__(message)
        self.response = response


class _FakeResponse:
    """Minimal requests.Response stand-in matching raise_for_status()."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        reason: str = "OK",
        url: str = "https://help.aliyun.com/zh/alinux/faq",
        http_error_cls: type[Exception] = _FakeHTTPError,
    ) -> None:
        self.status_code, self.text, self.reason, self.url = status_code, text, reason, url
        self.encoding = "utf-8"
        self._http_error_cls = http_error_cls

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise self._http_error_cls(
                f"{self.status_code} Error: {self.reason} for url: {self.url}",
                response=self,
            )


class _SimpleTag:
    """Minimal BeautifulSoup Tag stand-in backed by stdlib html.parser."""

    def __init__(self, name: str, attrs: dict[str, str], parent: _SimpleTag | None = None) -> None:
        self.name, self.attrs, self.parent = name, attrs, parent
        self.children: list[_SimpleTag | str] = []

    def get(self, key: str, default=None):
        return self.attrs.get(key, default)

    def _matches(self, name=None, attrs=None, class_=None) -> bool:
        if name is not None:
            names = name if isinstance(name, (list, tuple, set)) else (name,)
            if self.name not in names:
                return False
        if class_ is not None and class_ not in self.attrs.get("class", "").split():
            return False
        return attrs is None or all(self.attrs.get(k) == v for k, v in attrs.items())

    def find_all(self, name=None, attrs=None, class_=None) -> list[_SimpleTag]:
        found: list[_SimpleTag] = []
        for child in self.children:
            if isinstance(child, _SimpleTag):
                if child._matches(name=name, attrs=attrs, class_=class_):
                    found.append(child)
                found.extend(child.find_all(name=name, attrs=attrs, class_=class_))
        return found

    def find(self, name=None, attrs=None, class_=None) -> _SimpleTag | None:
        matches = self.find_all(name=name, attrs=attrs, class_=class_)
        return matches[0] if matches else None

    def get_text(self, strip: bool = False) -> str:
        text = "".join(c if isinstance(c, str) else c.get_text(False) for c in self.children)
        return text.strip() if strip else text

    def decompose(self) -> None:
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
        self.children.clear()

    def __str__(self) -> str:
        inner = "".join(str(c) for c in self.children)
        if self.name == "[document]":
            return inner
        attrs = "".join(f' {k}="{v}"' for k, v in self.attrs.items())
        return f"<{self.name}{attrs}>{inner}</{self.name}>"


class _SimpleSoupParser(html.parser.HTMLParser):
    VOID_TAGS = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self) -> None:
        super().__init__()
        self.root = _SimpleTag("[document]", {})
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _SimpleTag(tag, {k: (v or "") for k, v in attrs}, parent=self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        for idx in range(len(self._stack) - 1, 0, -1):
            if self._stack[idx].name == tag:
                del self._stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def _simple_soup(markup: str, _parser: str = "html.parser") -> _SimpleTag:
    parser = _SimpleSoupParser()
    parser.feed(markup)
    return parser.root


def _simple_markdownify(html_text: str, **_kwargs) -> str:
    return _simple_soup(html_text).get_text(strip=False)


def _build_stub_modules() -> dict[str, types.ModuleType]:
    stubs: dict[str, types.ModuleType] = {}
    for mod_name in ("requests", "bs4", "markdownify"):
        if mod_name in sys.modules:
            continue
        try:
            __import__(mod_name)
            continue
        except ImportError:
            pass
        if mod_name == "requests":
            req, exc = types.ModuleType("requests"), types.ModuleType("requests.exceptions")
            exc.RequestException, exc.HTTPError = Exception, _FakeHTTPError
            req.exceptions, req.RequestException, req.HTTPError, req.get = exc, Exception, _FakeHTTPError, None
            stubs["requests"], stubs["requests.exceptions"] = req, exc
        elif mod_name == "bs4":
            bs4_mod = types.ModuleType("bs4")
            bs4_mod.BeautifulSoup = _simple_soup
            stubs["bs4"] = bs4_mod
        else:
            md_mod = types.ModuleType("markdownify")
            md_mod.markdownify = _simple_markdownify
            stubs["markdownify"] = md_mod
    return stubs


def _load_crawl_docs():
    with mock.patch.dict(sys.modules, _build_stub_modules()):
        spec = importlib.util.spec_from_file_location("anolisa_crawl_docs_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


VALID_DOC_HTML = (
    '<!DOCTYPE html><html><head><title>FAQ - Alibaba Cloud Linux</title>'
    '<meta name="last-modified" content="2026-05-28T14:44:30+08:00" /></head>'
    '<body><h1>FAQ</h1><div class="markdown-body"><script>x()</script>'
    '<h2>What is ANOLISA?</h2><p>ANOLISA is an Agentic OS.</p></div></body></html>'
)
ERROR_502_HTML = "<html><head><title>502 Bad Gateway</title></head><body><h1>502 Bad Gateway</h1></body></html>"
ERROR_404_HTML = "<html><head><title>404 Not Found</title></head><body><h1>404 Not Found</h1></body></html>"
EMPTY_CONTAINER_HTML = (
    '<html><head><title>Empty</title></head><body><h1>Empty</h1>'
    '<div class="markdown-body"><script>x=1</script><style>a{}</style><nav>n</nav></div></body></html>'
)
CACHED_VALID_MD = (
    "# FAQ\n\n> **爬取时间**: 2026-06-01 12:00:00\n"
    "> **原文链接**: https://help.aliyun.com/zh/alinux/faq\n"
    "> **文档更新**: 2026-05-28T14:44:30+08:00\n\n---\n\nExisting valid cached documentation.\n"
)


class CrawlDocsHttpErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.crawl_docs = _load_crawl_docs()
        cls.http_error_cls = getattr(cls.crawl_docs.requests, "HTTPError", _FakeHTTPError)

    def _response(self, status_code: int, text: str, reason: str = "Error") -> _FakeResponse:
        return _FakeResponse(status_code, text, reason, http_error_cls=self.http_error_cls)

    def _crawl(self, ref_dir: Path, doc_urls: list[tuple[str, str]], responses: dict[str, _FakeResponse]):
        with (
            mock.patch.object(self.crawl_docs, "DOC_URLS", doc_urls),
            mock.patch.object(self.crawl_docs.requests, "get", side_effect=lambda url, **_kw: responses[url]),
            mock.patch.object(self.crawl_docs.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return self.crawl_docs.crawl_all_docs(ref_dir)

    def _run_main(self, ref_dir: Path, doc_urls: list[tuple[str, str]], responses: dict[str, _FakeResponse]):
        stderr = io.StringIO()
        with (
            mock.patch.object(self.crawl_docs, "DOC_URLS", doc_urls),
            mock.patch.object(self.crawl_docs.requests, "get", side_effect=lambda url, **_kw: responses[url]),
            mock.patch.object(self.crawl_docs.time, "sleep"),
            mock.patch.object(sys, "argv", ["crawl_docs.py", "--output-dir", str(ref_dir)]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            rc = self.crawl_docs.main()
        return rc, stderr.getvalue()

    def test_get_page_content_rejects_http_errors(self) -> None:
        for code, body, reason in (
            (404, ERROR_404_HTML, "Not Found"),
            (500, "<html><body>Internal Server Error</body></html>", "Internal Server Error"),
            (502, ERROR_502_HTML, "Bad Gateway"),
            (503, "<html><body>Service Unavailable</body></html>", "Service Unavailable"),
        ):
            with self.subTest(status_code=code):
                resp = self._response(code, body, reason)
                with (
                    mock.patch.object(self.crawl_docs.requests, "get", return_value=resp),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertIsNone(self.crawl_docs.get_page_content("/zh/alinux/faq"))

    def test_get_page_content_rejects_empty_2xx_body(self) -> None:
        for code, body in ((200, ""), (200, "   \n\t  "), (204, "")):
            with self.subTest(status_code=code, body=repr(body)):
                resp = self._response(code, body, "OK")
                with (
                    mock.patch.object(self.crawl_docs.requests, "get", return_value=resp),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertIsNone(self.crawl_docs.get_page_content("/zh/alinux/faq"))

    def test_get_page_content_returns_html_on_200(self) -> None:
        resp = self._response(200, VALID_DOC_HTML, "OK")
        with mock.patch.object(self.crawl_docs.requests, "get", return_value=resp):
            self.assertEqual(self.crawl_docs.get_page_content("/zh/alinux/faq"), VALID_DOC_HTML)

    def test_extract_markdown_fails_closed_on_missing_or_empty_content(self) -> None:
        for label, html_doc in (
            ("missing_container", ERROR_502_HTML),
            ("stripped_only_container", EMPTY_CONTAINER_HTML),
            ("blank_html", "   "),
        ):
            with self.subTest(case=label), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(self.crawl_docs.extract_markdown(html_doc, "/zh/alinux/faq"))

    def test_extract_markdown_supports_fallback_container(self) -> None:
        html_doc = '<html><head><title>Guide</title></head><body><div class="doc-content"><p>Fallback body.</p></div></body></html>'
        with contextlib.redirect_stdout(io.StringIO()):
            data = self.crawl_docs.extract_markdown(html_doc, "/zh/alinux/faq")
        self.assertIsNotNone(data)
        self.assertIn("Fallback body.", data["content"])

    def test_crawl_all_docs_preserves_cached_md_on_http_error(self) -> None:
        for code, body, reason in ((502, ERROR_502_HTML, "Bad Gateway"), (404, ERROR_404_HTML, "Not Found")):
            with self.subTest(status_code=code), tempfile.TemporaryDirectory() as tmp:
                ref_dir = Path(tmp)
                cached = ref_dir / "faq.md"
                cached.write_text(CACHED_VALID_MD, encoding="utf-8")
                summary = self._crawl(
                    ref_dir,
                    [("faq", "/zh/alinux/faq")],
                    {"https://help.aliyun.com/zh/alinux/faq": self._response(code, body, reason)},
                )
                self.assertEqual(cached.read_text(encoding="utf-8"), CACHED_VALID_MD)
                self.assertEqual((summary["success_count"], summary["failed_count"]), (0, 1))
                self.assertEqual(summary["results"][0]["status"], "failed")
                on_disk = json.loads((ref_dir / "crawl_summary.json").read_text(encoding="utf-8"))
                self.assertEqual((on_disk["failed_count"], on_disk["results"][0]["status"]), (1, "failed"))

    def test_crawl_all_docs_preserves_cached_md_and_avoids_stub_on_empty_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            cached = ref_dir / "faq.md"
            cached.write_text(CACHED_VALID_MD, encoding="utf-8")
            summary = self._crawl(
                ref_dir,
                [("faq", "/zh/alinux/faq"), ("releasenotes", "/zh/alinux/releasenotes")],
                {
                    "https://help.aliyun.com/zh/alinux/faq": self._response(200, ERROR_502_HTML, "OK"),
                    "https://help.aliyun.com/zh/alinux/releasenotes": self._response(200, EMPTY_CONTAINER_HTML, "OK"),
                },
            )
            self.assertEqual(cached.read_text(encoding="utf-8"), CACHED_VALID_MD)
            self.assertFalse((ref_dir / "releasenotes.md").exists())
            self.assertEqual((summary["success_count"], summary["failed_count"]), (0, 2))
            self.assertEqual([r["status"] for r in summary["results"]], ["failed", "failed"])

    def test_crawl_all_docs_partial_failure_saves_valid_and_preserves_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            cached_bad = ref_dir / "bad.md"
            cached_bad.write_text(CACHED_VALID_MD, encoding="utf-8")
            summary = self._crawl(
                ref_dir,
                [("good", "/zh/alinux/good"), ("bad", "/zh/alinux/bad")],
                {
                    "https://help.aliyun.com/zh/alinux/good": self._response(200, VALID_DOC_HTML, "OK"),
                    "https://help.aliyun.com/zh/alinux/bad": self._response(502, ERROR_502_HTML, "Bad Gateway"),
                },
            )
            good_text = (ref_dir / "good.md").read_text(encoding="utf-8")
            self.assertIn("# FAQ", good_text)
            self.assertIn("Agentic OS", good_text)
            self.assertEqual(cached_bad.read_text(encoding="utf-8"), CACHED_VALID_MD)
            self.assertEqual(list(ref_dir.glob("*.tmp")), [])
            self.assertEqual((summary["success_count"], summary["failed_count"]), (1, 1))
            self.assertEqual({r["filename"]: r["status"] for r in summary["results"]}, {"good": "success", "bad": "failed"})

    def test_save_markdown_file_rejects_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "faq.md"
            target.write_text(CACHED_VALID_MD, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.crawl_docs.save_markdown_file(
                    {"title": "502", "full_url": "https://help.aliyun.com/zh/alinux/faq", "last_modified": "", "content": " \n "},
                    target,
                )
            self.assertEqual(target.read_text(encoding="utf-8"), CACHED_VALID_MD)

    def test_main_returns_nonzero_on_failure_and_zero_on_full_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            rc_fail, err_fail = self._run_main(
                ref_dir,
                [("faq", "/zh/alinux/faq")],
                {"https://help.aliyun.com/zh/alinux/faq": self._response(502, ERROR_502_HTML, "Bad Gateway")},
            )
            self.assertEqual(rc_fail, 1)
            self.assertTrue(err_fail.strip())

            rc_partial, err_partial = self._run_main(
                ref_dir,
                [("good", "/zh/alinux/good"), ("bad", "/zh/alinux/bad")],
                {
                    "https://help.aliyun.com/zh/alinux/good": self._response(200, VALID_DOC_HTML, "OK"),
                    "https://help.aliyun.com/zh/alinux/bad": self._response(404, ERROR_404_HTML, "Not Found"),
                },
            )
            self.assertEqual(rc_partial, 1)
            self.assertIn("1/2", err_partial)

            rc_ok, err_ok = self._run_main(
                ref_dir,
                [("faq", "/zh/alinux/faq")],
                {"https://help.aliyun.com/zh/alinux/faq": self._response(200, VALID_DOC_HTML, "OK")},
            )
            self.assertEqual(rc_ok, 0)
            self.assertEqual(err_ok, "")


if __name__ == "__main__":
    unittest.main()
