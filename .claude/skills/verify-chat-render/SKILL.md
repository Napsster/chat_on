---
name: verify-chat-render
description: Guardrail check for the web chat's markdown renderer in index.html — run this any time that file's chat-rendering code (escapeHtml/renderInline/renderMarkdown/table helpers) changes, or when a user reports garbled formatting (literal **, |, #, or [text](url) showing up instead of rendering) in a bot reply on the web UI.
---

# Verify chat render

The web chat UI (`index.html`) renders bot replies by escaping
HTML and then converting the markdown constructs the bot's KB-grounded
replies actually produce: bold, italics, inline code, links, bullet lists,
headings, and pipe tables. This has broken silently before — once for links
not being clickable at all, once for pipe tables rendering as a wall of `|`
and `-` characters — because nothing verified the renderer against real
markdown shapes pulled from actual bot replies.

## When to run this

- After editing anything between `function escapeHtml` and
  `function appendChatBubble` in `index.html`.
- When a user reports raw markdown syntax visible in a chat bubble
  (`**text**`, `[label](url)`, a table showing as `| --- | --- |`, etc.).
- Before deploying any change that touches the web chat UI, as a quick
  sanity check alongside the usual test/deploy flow.

## How to run it

```bash
node test_chat_render.js
```

This extracts the **actual shipped renderer source** straight out of
`index.html` (not a reimplementation, so it can't silently drift
out of sync with the real code) and asserts it against a battery of
realistic inputs — including the exact ragged pipe table from the live
"flexi pay" reply that first exposed this gap. Exits non-zero on any
failure.

## If it fails, or a new markdown shape breaks rendering

1. Reproduce the exact bot-reply text that rendered wrong (check
   `data/users/<session>.json` on the server, or the browser's chat log).
2. Add it as a new `check(...)` case in `test_chat_render.js` first — this
   pins the regression before touching the fix, the same way you'd add a
   failing test before a bug fix anywhere else.
3. Fix the renderer in `index.html`.
4. Re-run `node test_chat_render.js` until it's green.
5. Deploy: this file is static HTML/JS served directly — a `git pull` on
   the Hostinger server is enough, no service restart needed for this file
   alone (restart is only required when `knowledge_base.md` or the
   `knowledge_documents` DB content changes, since those trigger the
   embedding rebuild).

## Design constraints worth preserving

- **Escape before converting.** `escapeHtml()` must run before any markdown
  conversion, or a bot reply that happens to contain literal `<script>` /
  `<img onerror=...>` (e.g. echoed back from something a user typed) could
  execute as real HTML.
- **Links are scheme-restricted.** Only `http(s)://` and `mailto:` URLs
  become clickable `<a>` tags — this stops a `javascript:` URL slipped into
  `[label](url)` syntax from ever becoming an executable link.
- **User-typed messages are escaped only, never markdown-rendered** — no
  reason to interpret a user's own raw text as markdown, and it removes a
  class of self-XSS concern for free.
