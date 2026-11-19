"""Real-browser tools for RL (Playwright Chromium).

Boots an actual browser process (not a simulated DOM). Three tools share one
session:

* ``browser_goto(url)`` — navigate
* ``browser_read()`` — markdown-ish page text with link URLs + interactive ids
* ``browser_act(element_id, action, text)`` — click or type into a marked id

Install::

    pip install "teapottrainer[browser]"
    playwright install chromium

Usage::

    from teapottrainer import BrowserSettings, bind_tools, build_browser_tools, NQ_OPEN_RL

    tools = build_browser_tools(BrowserSettings())  # launches Chromium
    dataset = bind_tools(NQ_OPEN_RL, tools)
"""

from __future__ import annotations

import os
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .tools import ToolFn

# JS runs in the live page: tag interactive nodes, return markdown + catalog.
_EXTRACT_JS = r"""
() => {
  const SELECTORS = [
    'a[href]',
    'button',
    'input:not([type="hidden"])',
    'textarea',
    'select',
    '[role="button"]',
    '[role="link"]',
    '[contenteditable="true"]',
  ].join(',');

  document.querySelectorAll('[data-teapot-id]').forEach((el) => {
    el.removeAttribute('data-teapot-id');
  });

  const isVisible = (el) => {
    if (!el || !(el instanceof Element)) return false;
    if (typeof el.checkVisibility === 'function') {
      try { return el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }); }
      catch (e) { /* fall through */ }
    }
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const catalog = [];
  let n = 0;
  for (const el of document.querySelectorAll(SELECTORS)) {
    if (!isVisible(el)) continue;
    n += 1;
    const id = 'e' + n;
    el.setAttribute('data-teapot-id', id);
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let kind = tag;
    if (tag === 'a') kind = 'link';
    else if (tag === 'button' || el.getAttribute('role') === 'button') kind = 'button';
    else if (tag === 'input') kind = type === 'submit' || type === 'button' ? 'button' : (type || 'textbox');
    else if (tag === 'textarea') kind = 'textbox';
    else if (tag === 'select') kind = 'select';
    else if (el.getAttribute('contenteditable') === 'true') kind = 'textbox';

    const label = (
      el.getAttribute('aria-label')
      || el.getAttribute('placeholder')
      || el.getAttribute('name')
      || el.getAttribute('value')
      || (el.innerText || '').trim()
      || ''
    ).replace(/\s+/g, ' ').slice(0, 80);

    catalog.push({
      id,
      kind,
      tag,
      type: type || null,
      label,
      href: tag === 'a' ? (el.href || el.getAttribute('href') || null) : null,
    });
  }

  const blockTags = new Set([
    'p', 'div', 'section', 'article', 'main', 'header', 'footer', 'nav',
    'li', 'ul', 'ol', 'table', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'blockquote', 'pre', 'hr', 'br',
  ]);

  function escapeText(s) {
    return String(s || '').replace(/\s+/g, ' ');
  }

  function render(node, parts) {
    if (!node) return;
    if (node.nodeType === Node.TEXT_NODE) {
      const t = escapeText(node.nodeValue);
      if (t.trim()) parts.push(t);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node;
    const tag = el.tagName.toLowerCase();
    if (tag === 'script' || tag === 'style' || tag === 'noscript' || tag === 'svg') return;
    if (!isVisible(el) && tag !== 'body' && tag !== 'html') return;

    const tid = el.getAttribute('data-teapot-id');
    if (tid) {
      if (tag === 'a') {
        const text = escapeText(el.innerText || el.getAttribute('aria-label') || el.href || 'link').trim() || 'link';
        const href = el.href || el.getAttribute('href') || '';
        parts.push(`[${text}](${href}) \`#${tid}\``);
        return;
      }
      if (tag === 'button' || el.getAttribute('role') === 'button'
          || (tag === 'input' && ['button', 'submit', 'reset'].includes((el.type || '').toLowerCase()))) {
        const label = escapeText(
          el.getAttribute('aria-label') || el.value || el.innerText || 'button'
        ).trim() || 'button';
        parts.push(`[button #${tid}: ${label}]`);
        return;
      }
      if (tag === 'input' || tag === 'textarea' || el.getAttribute('contenteditable') === 'true') {
        const kind = tag === 'textarea' ? 'textbox' : ((el.type || 'textbox').toLowerCase());
        const label = escapeText(
          el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name') || kind
        ).trim();
        const value = escapeText(el.value || el.innerText || '').trim();
        const valueBit = value ? ` value=${JSON.stringify(value).slice(0, 60)}` : '';
        parts.push(`[${kind} #${tid}: ${label}${valueBit}]`);
        return;
      }
      if (tag === 'select') {
        const label = escapeText(el.getAttribute('aria-label') || el.getAttribute('name') || 'select').trim();
        parts.push(`[select #${tid}: ${label}]`);
        return;
      }
    }

    if (tag === 'br' || tag === 'hr') {
      parts.push('\n');
      return;
    }
    if (/^h[1-6]$/.test(tag)) {
      const level = Number(tag[1]);
      parts.push('\n' + '#'.repeat(level) + ' ');
      for (const child of el.childNodes) render(child, parts);
      parts.push('\n');
      return;
    }
    if (tag === 'li') {
      parts.push('\n- ');
      for (const child of el.childNodes) render(child, parts);
      return;
    }

    const before = blockTags.has(tag) ? '\n' : '';
    if (before) parts.push(before);
    for (const child of el.childNodes) render(child, parts);
    if (before) parts.push('\n');
  }

  const parts = [];
  const title = escapeText(document.title || '').trim();
  const url = String(location.href || '');
  if (title) parts.push(`# ${title}\n`);
  parts.push(`URL: ${url}\n`);
  render(document.body, parts);

  let markdown = parts.join('').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();

  const interactiveLines = catalog.map((row) => {
    if (row.kind === 'link') {
      return `- link #${row.id}: ${row.label || '(link)'} -> ${row.href || ''}`;
    }
    return `- ${row.kind} #${row.id}: ${row.label || row.tag}`;
  });
  if (interactiveLines.length) {
    markdown += '\n\n## Interactive elements\n' + interactiveLines.join('\n');
  } else {
    markdown += '\n\n## Interactive elements\n(none visible)';
  }

  return { markdown, catalog, title, url };
}
"""


class BrowserSettings(BaseModel):
    """Knobs for the Playwright browser session."""

    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30_000
    max_read_chars: int = 12_000
    # Launch Chromium immediately when tools are built (real process).
    eager: bool = True
    # Deterministic stubs — no Playwright process (unit tests / dry runs).
    mock: bool = False
    # Extra Chromium launch args (e.g. Colab/sandbox tweaks).
    chromium_args: list[str] = Field(
        default_factory=lambda: ["--disable-dev-shm-usage", "--no-sandbox"]
    )

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_env(cls, **overrides: Any) -> "BrowserSettings":
        data: dict[str, Any] = {}
        if os.environ.get("TEAPOT_BROWSER_MOCK", "").strip().lower() in {"1", "true", "yes"}:
            data["mock"] = True
        if os.environ.get("TEAPOT_BROWSER_HEADED", "").strip().lower() in {"1", "true", "yes"}:
            data["headless"] = False
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class BrowserSession:
    """Owns one real Chromium process + page (or a mock stand-in)."""

    def __init__(self, settings: BrowserSettings | None = None):
        self.settings = settings or BrowserSettings()
        self._lock = threading.RLock()
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._started = False
        self._mock_url = "about:blank"
        self._mock_title = ""
        self._mock_body = ""
        self._mock_catalog: list[dict[str, Any]] = []
        self._id_to_selector: dict[str, str] = {}

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self.settings.mock:
                self._started = True
                return
            try:
                from playwright.sync_api import sync_playwright
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "playwright is not installed. Run: pip install playwright && playwright install chromium"
                ) from exc
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=bool(self.settings.headless),
                args=list(self.settings.chromium_args or []),
            )
            self._context = self._browser.new_context(
                viewport={
                    "width": int(self.settings.viewport_width),
                    "height": int(self.settings.viewport_height),
                }
            )
            self._context.set_default_timeout(float(self.settings.timeout_ms))
            self._page = self._context.new_page()
            self._started = True

    def ensure(self) -> None:
        if not self._started:
            self.start()

    def close(self) -> None:
        with self._lock:
            if self.settings.mock:
                self._started = False
                self._id_to_selector.clear()
                return
            for obj in (self._context, self._browser):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:  # noqa: BLE001
                        pass
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:  # noqa: BLE001
                    pass
            self._pw = None
            self._browser = None
            self._context = None
            self._page = None
            self._started = False
            self._id_to_selector.clear()

    def goto(self, url: str) -> str:
        self.ensure()
        target = str(url or "").strip()
        if not target:
            return "ERROR: empty url"
        with self._lock:
            if self.settings.mock:
                self._mock_url = target
                self._mock_title = f"Mock page for {target}"
                self._mock_body = (
                    f"Stub page body for {target}.\n\n"
                    f"[Example]({target}/docs) `#e1`\n"
                    f"[button #e2: OK]\n"
                    f"[textbox #e3: query value=\"\"]\n"
                )
                self._mock_catalog = [
                    {"id": "e1", "kind": "link", "label": "Example", "href": f"{target}/docs"},
                    {"id": "e2", "kind": "button", "label": "OK", "href": None},
                    {"id": "e3", "kind": "textbox", "label": "query", "href": None},
                ]
                self._id_to_selector = {row["id"]: f"[data-teapot-id='{row['id']}']" for row in self._mock_catalog}
                return f"OK navigated to {target}"
            assert self._page is not None
            try:
                self._page.goto(target, wait_until="domcontentloaded")
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: navigation failed: {exc}"
            self._id_to_selector.clear()
            return f"OK navigated to {self._page.url}"

    def read(self, max_chars: int | None = None) -> str:
        self.ensure()
        limit = int(max_chars) if max_chars is not None else int(self.settings.max_read_chars)
        limit = max(500, limit)
        with self._lock:
            if self.settings.mock:
                title = self._mock_title or "(mock)"
                md = f"# {title}\nURL: {self._mock_url}\n\n{self._mock_body}"
                lines = []
                for row in self._mock_catalog:
                    if row["kind"] == "link":
                        lines.append(f"- link #{row['id']}: {row['label']} -> {row['href']}")
                    else:
                        lines.append(f"- {row['kind']} #{row['id']}: {row['label']}")
                md += "\n\n## Interactive elements\n" + ("\n".join(lines) if lines else "(none visible)")
                if len(md) > limit:
                    md = md[: limit - 20] + "\n...[truncated]"
                return md
            assert self._page is not None
            try:
                payload = self._page.evaluate(_EXTRACT_JS)
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: page read failed: {exc}"
            if not isinstance(payload, dict):
                return "ERROR: unexpected extract payload"
            catalog = payload.get("catalog") or []
            self._id_to_selector = {
                str(row["id"]): f"[data-teapot-id='{row['id']}']"
                for row in catalog
                if isinstance(row, dict) and row.get("id")
            }
            md = str(payload.get("markdown") or "").strip() or "(empty page)"
            if len(md) > limit:
                md = md[: limit - 20] + "\n...[truncated]"
            return md

    def act(self, element_id: str, action: str = "click", text: str = "") -> str:
        self.ensure()
        eid = str(element_id or "").strip().lstrip("#")
        act = str(action or "click").strip().lower()
        if not eid:
            return "ERROR: empty element_id"
        if act not in {"click", "type"}:
            return "ERROR: action must be 'click' or 'type'"
        with self._lock:
            if self.settings.mock:
                if eid not in self._id_to_selector and eid not in {r["id"] for r in self._mock_catalog}:
                    known = ", ".join(sorted(self._id_to_selector) or [r["id"] for r in self._mock_catalog]) or "(none)"
                    return f"ERROR: unknown element_id {eid!r}. Call browser_read first. Known: {known}"
                if act == "type":
                    for row in self._mock_catalog:
                        if row["id"] == eid:
                            row["label"] = f"{row.get('label', '')} (typed)"
                    return f"OK typed into #{eid}: {text!r}"
                return f"OK clicked #{eid}"
            assert self._page is not None
            # Refresh map if missing (model skipped read, or page mutated).
            if eid not in self._id_to_selector:
                self.read()
            selector = self._id_to_selector.get(eid)
            if not selector:
                known = ", ".join(sorted(self._id_to_selector)) or "(none)"
                return f"ERROR: unknown element_id {eid!r}. Call browser_read first. Known: {known}"
            try:
                locator = self._page.locator(selector).first
                if act == "click":
                    locator.click()
                    return f"OK clicked #{eid} (url={self._page.url})"
                locator.click()
                locator.fill(str(text))
                return f"OK typed into #{eid}: {text!r}"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: act failed on #{eid}: {exc}"


class BrowserToolkit:
    """Bundle of browser tools sharing one live Chromium session."""

    def __init__(self, session: BrowserSession, tools: list[ToolFn]):
        self.session = session
        self.tools = tools

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "BrowserToolkit":
        self.session.ensure()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def build_browser_toolkit(
    settings: BrowserSettings | None = None,
    **overrides: Any,
) -> BrowserToolkit:
    """Create a real-browser toolkit (goto / read / act) on one Chromium session."""
    cfg = (settings or BrowserSettings.from_env()).model_copy(update=overrides)
    session = BrowserSession(cfg)

    def browser_goto(url: str) -> str:
        """Open a URL in the live browser and wait for the page to load.

        Args:
            url: Absolute URL to visit (http:// or https://).
        """
        return session.goto(url)

    def browser_read(max_chars: int | None = None) -> str:
        """Read the current page as markdown-like text with interactive element ids.

        Links include their URLs. Buttons, text inputs, and other controls are
        tagged as ``#eN`` ids you can pass to ``browser_act``.

        Args:
            max_chars: Optional truncation limit for the returned text.
        """
        return session.read(max_chars=max_chars)

    def browser_act(element_id: str, action: str = "click", text: str = "") -> str:
        """Click or type into an element previously listed by ``browser_read``.

        Args:
            element_id: Id such as ``e3`` or ``#e3`` from ``browser_read``.
            action: ``click`` or ``type``.
            text: Characters to type when ``action`` is ``type``.
        """
        return session.act(element_id, action=action, text=text)

    tools: list[ToolFn] = [browser_goto, browser_read, browser_act]
    toolkit = BrowserToolkit(session, tools)
    if cfg.eager:
        session.start()
    return toolkit


def build_browser_tools(
    settings: BrowserSettings | None = None,
    **overrides: Any,
) -> list[ToolFn]:
    """Return ``[browser_goto, browser_read, browser_act]`` sharing one session.

    Prefer :func:`build_browser_toolkit` when you need ``close()``.
    """
    return build_browser_toolkit(settings, **overrides).tools
