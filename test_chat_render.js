#!/usr/bin/env node
/**
 * Guardrail for the web chat's markdown renderer (upload_interface.html).
 *
 * Extracts the ACTUAL shipped escapeHtml/renderInline/renderMarkdown
 * function source out of upload_interface.html (not a reimplementation —
 * this always tests the real code, so it can't silently drift out of sync)
 * and runs it against markdown patterns the bot's replies actually produce:
 * bold, links, bullets, headings, inline code, and — the bug this file was
 * written after — pipe tables (KB documents are full of these; an
 * unrendered table shows up as a garbled wall of "|" and "-" characters).
 *
 * Run after ANY change to the renderer in upload_interface.html:
 *   node test_chat_render.js
 *
 * Exits non-zero on any failure, so it's also CI/pre-deploy-friendly.
 */
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const HTML_PATH = path.join(__dirname, "upload_interface.html");
const html = fs.readFileSync(HTML_PATH, "utf-8");

const startMarker = "function escapeHtml(text) {";
const endMarker = "\n        function appendChatBubble(";
const startIdx = html.indexOf(startMarker);
const endIdx = html.indexOf(endMarker);
if (startIdx === -1 || endIdx === -1) {
    console.error("Could not locate renderer functions in upload_interface.html — markers moved?");
    process.exit(1);
}
const rendererSource = html.slice(startIdx, endIdx);

// Evaluate the extracted renderer in an isolated scope, then export what we need.
const exportedFns = new Function(
    rendererSource + "\nreturn { escapeHtml, renderInline, renderMarkdown, isTableRow, isTableSeparator, splitTableRow };"
)();
const { escapeHtml, renderMarkdown } = exportedFns;

let passed = 0;
let failed = 0;

function check(name, fn) {
    try {
        fn();
        passed++;
        console.log(`  ok  - ${name}`);
    } catch (e) {
        failed++;
        console.error(`FAIL - ${name}`);
        console.error(`       ${e.message}`);
    }
}

console.log("Testing renderMarkdown() extracted live from upload_interface.html\n");

check("bold renders as <strong>, no literal ** left", () => {
    const out = renderMarkdown("We work with **650+ brands, 675+ certified recyclers, and 5,000+ aggregators**.");
    assert.ok(out.includes("<strong>650+ brands, 675+ certified recyclers, and 5,000+ aggregators</strong>"));
    assert.ok(!out.includes("**"));
});

check("markdown link becomes a real, safe <a> tag", () => {
    const out = renderMarkdown("[Click here](https://docs.google.com/forms/d/abc/viewform?edit_requested=true)");
    assert.ok(out.includes('<a href="https://docs.google.com/forms/d/abc/viewform?edit_requested=true" target="_blank" rel="noopener noreferrer">Click here</a>'));
});

check("javascript: URLs are never turned into a clickable link", () => {
    const out = renderMarkdown("[bad](javascript:alert(1))");
    assert.ok(!out.includes("<a "));
});

check("real <script> tags in bot text are escaped, not executed", () => {
    const out = renderMarkdown("<script>alert(1)</script>");
    assert.ok(out.includes("&lt;script&gt;"));
    assert.ok(!out.includes("<script>"));
});

check("bullet lines get a bullet marker, not a literal dash", () => {
    const out = renderMarkdown("- Personal Information: fill this in\n- Introduction Form: also this");
    assert.ok(out.includes("• Personal Information"));
    assert.ok(out.includes("• Introduction Form"));
});

check("heading line renders as bold, not literal #", () => {
    const out = renderMarkdown("## Scope\nAll employees are covered.");
    assert.ok(out.includes("<strong>Scope</strong>"));
    assert.ok(!out.includes("##"));
});

check("inline code renders as <code>", () => {
    const out = renderMarkdown("Run `npm install` first.");
    assert.ok(out.includes("<code>npm install</code>"));
});

check("italics render as <em>, without breaking adjacent bold", () => {
    const out = renderMarkdown("This is *important* and **very important**.");
    assert.ok(out.includes("<em>important</em>"));
    assert.ok(out.includes("<strong>very important</strong>"));
});

// The exact table from the live "flexi pay" reply that prompted this fix —
// a real, ragged pipe table (some rows repeat an empty first cell).
const FLEXI_PAY_TABLE = [
    "| **Benefit Component** | **Amount / Limit** | **Eligibility** |",
    "|---|---|---|",
    "| **Food/Meal Vouchers** (via Zaggle) | ₹4,400/month or ₹8,800/month | All Employees |",
    "| **Fuel Allowance** (via Zaggle) | ₹900/month (Two-Wheeler) | All Employees |",
    "| | ₹5,000/month (Car ≤1600 CC) | Sr. Manager & above |",
    "| | ₹7,000/month (Car >1600 CC) | Sr. Manager & above |",
    "| **Mobile / Telephone Allowance** | ₹1,000/month | All Employees |",
].join("\n");

check("pipe table renders as a real <table>, not garbled | and - characters", () => {
    const out = renderMarkdown(FLEXI_PAY_TABLE);
    assert.ok(out.includes("<table"));
    assert.ok(out.includes("<th><strong>Benefit Component</strong></th>"));
    assert.ok(out.includes("<td><strong>Food/Meal Vouchers</strong> (via Zaggle)</td>"));
    // the ragged row with an empty first cell should still produce 3 cells
    assert.ok(out.includes("<td></td><td>₹5,000/month (Car ≤1600 CC)</td><td>Sr. Manager &amp; above</td>"));
    // none of the raw table syntax should leak through as visible text
    assert.ok(!out.includes("|---|"));
});

check("a table followed by more prose keeps both intact", () => {
    const out = renderMarkdown(FLEXI_PAY_TABLE + "\n\nAll full-time employees are eligible.");
    assert.ok(out.includes("<table"));
    assert.ok(out.includes("All full-time employees are eligible."));
});

check("text that merely contains a lone pipe is NOT mistaken for a table", () => {
    const out = renderMarkdown("Use ctrl+shift+p | cmd+shift+p to open the palette.");
    assert.ok(!out.includes("<table"));
});

check("escapeHtml alone still neutralizes HTML for user-typed messages", () => {
    assert.strictEqual(escapeHtml("<b>hi</b> & \"quotes\" 'single'"),
        "&lt;b&gt;hi&lt;/b&gt; &amp; &quot;quotes&quot; &#39;single&#39;");
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
