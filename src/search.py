"""FastAPI + htmx search interface for agenda items."""

import json
import re

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from markdown_it import MarkdownIt

from src.agent import run_agent
from src.retriever import search, get_case, get_meeting

app = FastAPI()
markdown_renderer = MarkdownIt("commonmark", {"breaks": True, "html": False})
default_link_renderer = markdown_renderer.renderer.rules.get("link_open", markdown_renderer.renderer.renderToken)  # type: ignore


def render_link_open(tokens, idx, options, env):
    tokens[idx].attrSet("target", "_blank")
    tokens[idx].attrSet("rel", "noopener")
    return default_link_renderer(tokens, idx, options, env)


markdown_renderer.renderer.rules["link_open"] = render_link_open  # type: ignore


# === Utility functions ===

def format_date(dt: str) -> str:
    """Convert yyyy-mm-dd to dd-mm-yyyy."""
    if dt and len(dt) >= 10:
        parts = dt[:10].split("-")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return dt


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for plain text display."""
    text = re.sub(r'\n#{1,6}\s*', '\n', text)
    text = re.sub(r'^#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def query_terms(query: str) -> list[str]:
    """Extract query words for snippet scoring and highlighting."""
    if not query:
        return []
    # Keep words with letters/digits, allow internal hyphens/apostrophes.
    return re.findall(r"[\w]+(?:[-'][\w]+)*", query)


def create_chunks(text: str, size: int = 250, overlap: int = 50) -> list[str]:
    """Create overlapping chunks at word boundaries."""
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) <= size:
        return [text] if text else []

    words = text.split(' ')
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = []
        chunk_len = 0
        j = i

        while j < len(words):
            word_len = len(words[j])
            new_len = chunk_len + word_len + (1 if chunk_words else 0)
            if new_len > size and chunk_words:
                break
            chunk_words.append(words[j])
            chunk_len = new_len
            j += 1

        if chunk_words:
            chunks.append(' '.join(chunk_words))

        if j >= len(words):
            break

        overlap_chars = 0
        overlap_words = 0
        for k in range(len(chunk_words) - 1, -1, -1):
            new_overlap = overlap_chars + len(chunk_words[k]) + (1 if overlap_words > 0 else 0)
            if new_overlap > overlap:
                break
            overlap_chars = new_overlap
            overlap_words += 1

        i = max(i + 1, j - overlap_words)

    return chunks


def extract_snippet(tokens: list[str], content: str, size: int = 250, overlap: int = 50) -> str:
    """Extract the most relevant snippet using keyword density scoring."""
    if not content:
        return ""

    plain = strip_markdown(content)
    if not plain:
        return ""

    # Short content - return as is
    if len(plain) <= size:
        return plain

    # No tokens - return start of text
    if not tokens:
        return plain[:size] + " ..." if len(plain) > size else plain

    # Pattern: word boundary before token (matches word starts only)
    term_patterns = [f"\\b{re.escape(t)}" for t in tokens]
    pattern = re.compile("|".join(term_patterns), re.IGNORECASE)

    # Sliding window - find chunk with highest keyword density
    best_chunk = plain[:size]
    best_score = len(pattern.findall(best_chunk))
    best_start = 0

    step = size - overlap
    for start in range(step, len(plain) - size // 2, step):
        end = min(start + size, len(plain))
        chunk = plain[start:end]

        # Skip short trailing chunks
        if len(chunk) < size // 2:
            break

        score = len(pattern.findall(chunk))
        if score > best_score:
            best_score = score
            best_chunk = chunk
            best_start = start

    # Format with ellipsis
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if best_start + len(best_chunk) < len(plain) else ""

    return f"{prefix}{best_chunk}{suffix}"


def restore_links(content: str, links: list[str]) -> str:
    """Replace [text](idx) with [text](url) using the links list."""
    def replacer(match: re.Match) -> str:
        text = match.group(1)
        idx_str = match.group(2)
        try:
            idx = int(idx_str)
            if 0 <= idx < len(links):
                return f"[{text}]({links[idx]})"
        except ValueError:
            pass
        return match.group(0)
    return re.sub(r"\[([^\]]+)\]\((\d+)\)", replacer, content)


def linkify_urls(content: str) -> str:
    """Convert bare https:// URLs to markdown links."""
    # Pattern matches https:// followed by non-whitespace chars
    # Negative lookbehind to avoid URLs already in markdown links: ](url) or [url](
    pattern = r'(?<!\]\()(?<!\[)(https://[^\s\[\]<>]+)'

    def replacer(match: re.Match) -> str:
        url = match.group(1)
        # Strip trailing punctuation that's likely not part of URL
        trailing = ""
        while url and url[-1] in ".,;:!?)":
            # Keep ) if there's a matching ( in the URL (Wikipedia-style)
            if url[-1] == ")" and "(" in url:
                break
            trailing = url[-1] + trailing
            url = url[:-1]
        if not url or url == "https://":
            return match.group(0)
        return f"[{url}]({url}){trailing}"

    return re.sub(pattern, replacer, content)


def md_to_html(content: str, links: list[str]) -> str:
    """Simple markdown to HTML conversion."""
    content = restore_links(content, links)
    content = linkify_urls(content)
    if not content.strip():
        return ""
    return markdown_renderer.render(content)


def esc(text: str) -> str:
    """Escape HTML."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def highlight_terms(text: str, tokens: list[str]) -> str:
    """Highlight tokens at word starts, extending to full word."""
    if not text:
        return ""

    if not tokens:
        return esc(text)

    # Dedupe and sort by length (longest first to avoid partial matches)
    unique = []
    seen = set()
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            unique.append(token)

    unique.sort(key=len, reverse=True)
    # Pattern: word boundary before token, then extend to end of word
    # \btoken matches start, \w* extends to full word
    term_patterns = [f"\\b{re.escape(t)}\\w*" for t in unique]
    pattern = re.compile("|".join(term_patterns), re.IGNORECASE)

    parts = []
    last = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > last:
            parts.append(esc(text[last:start]))
        parts.append(f"<strong>{esc(text[start:end])}</strong>")
        last = end
    parts.append(esc(text[last:]))

    return "".join(parts)


# === HTML Templates ===

PAGE_HTML = """<!DOCTYPE html>
<html lang="da">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dagsordener og referater</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/marked@12.0.0/marked.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: arial, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.5; }
        h1 { margin-bottom: 20px; }
        input[type="search"] { width: 100%; padding: 12px; font-size: 16px; border: 2px solid #ccc; border-radius: 4px; }
        input[type="search"]:focus { outline: none; border-color: #0543ab; }
        .results { margin-top: 20px; }
        .result { border: 1px solid transparent; border-radius: 4px; margin-bottom: 10px; position: relative; }
        .result.open { border-color: #ddd; }
        .result-num { position: absolute; left: -32px; top: 12px; font-size: 14px; color: #999; width: 24px; text-align: right; }
        .result-header { padding: 12px; cursor: pointer; }
        .result-header:hover { background: #f8f8f8; }
        .result-meta { font-size: 14px; color: #666; }
        .result-title { font-size: 18px; font-weight: 600; margin: 4px 0; color: #0543ab; }
        .result-snippet { font-size: 14px; color: #444; margin-top: 8px; min-height: 2.8em; line-height: 1.4; }
        .result-content { padding: 12px; display: none; }
        .result.open .result-content { display: block; }
        .result-content p { margin: 0 0 10px 0; }
        .context-nav { margin: 0 0 15px 0; }
        .context-tabs { display: flex; background: #fff; }
        .context-tab { padding: 8px 16px; background: none; border: none; cursor: pointer; font-size: 14px; color: #666; }
        .context-tab:hover { background: #eee; }
        .context-tab.active { color: #000; font-weight: 600; background: #f5f5f5; border: 1px solid #ddd; border-bottom: none; border-radius: 4px 4px 0 0; margin-bottom: -1px; z-index: 1; position: relative; }
        .context-content { padding: 12px; background: #f5f5f5; border: 1px solid #ddd; border-radius: 0 4px 4px 4px; }
        .context-toc { margin: 0; padding-left: 25px; font-size: 14px; }
        .context-toc li { margin: 4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; list-style-position: inside; }
        .context-toc a { color: #0543ab; text-decoration: none; cursor: pointer; }
        .context-toc a:hover { text-decoration: underline; }
        .context-toc a.active { color: #000; font-weight: 700; }
        .context-toc a.active:hover { text-decoration: none; }
        .load-sentinel { height: 1px; }
        .loading { opacity: 0.5; }
        a { color: #0543ab; }
        .result-content h5 { font-size: inherit; font-weight: 600; margin: 10px 0 5px 0; }
        .original-link { font-size: 14px; margin-top: 10px; text-align: right; }
        .status { color: #666; font-size: 14px; margin-bottom: 10px; }
        .ai-answer { background: #f0f7ff; border: 1px solid #c5ddf7; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
        .ai-answer-header { font-weight: 600; margin-bottom: 8px; color: #0543ab; }
        .ai-answer-content { line-height: 1.6; font-size: 14px; }
        .ai-answer-content table { border-collapse: collapse; width: 100%; margin: 8px 0; }
        .ai-answer-content th, .ai-answer-content td { padding: 6px 10px; border: 1px solid rgba(0,0,0,0.1); text-align: left; }
        .ai-answer-content th { background: rgba(0,0,0,0.03); font-weight: 600; }
        .ai-answer-steps { font-size: 13px; color: #666; margin-bottom: 8px; font-style: italic; }
        .ai-answer.loading .ai-answer-content { color: #666; }
    </style>
</head>
<body>
    <h1>Dagsordener og referater</h1>
    <input type="search"
           id="search-input"
           name="q"
           placeholder="Søg i referater..."
           hx-get="/search"
           hx-trigger="keyup[key=='Enter']"
           hx-target="#results"
           hx-indicator="#results">
    <div id="ai-answer" class="ai-answer" style="display:none">
        <div class="ai-answer-header">AI-opsummering (demo)</div>
        <div class="ai-answer-steps"></div>
        <div class="ai-answer-content"></div>
    </div>
    <div id="results" class="results"></div>
    <script>
        // Map: sagsnummer -> Set of "date|udvalg" combinations seen
        const seenSager = new Map();
        let resultCount = 0;

        // Dedupe by sagsnummer, but only if date OR udvalg differs
        // (same sagsnr + same date + same udvalg = not a real case, just reused sagsnr)
        document.body.addEventListener('htmx:beforeSwap', (e) => {
            const isNewSearch = e.detail.target.id === 'results';
            const isLoadMore = e.detail.target.classList.contains('load-sentinel');
            if (!isNewSearch && !isLoadMore) return;

            if (isNewSearch) {
                seenSager.clear();
                resultCount = 0;
            }

            const temp = document.createElement('div');
            temp.innerHTML = e.detail.serverResponse;

            temp.querySelectorAll('.result').forEach(el => {
                const sagsnr = el.dataset.sagsnummer;
                const date = el.dataset.date || '';
                const udvalg = el.dataset.udvalg || '';
                const key = date + '|' + udvalg;

                let isDupe = false;
                if (sagsnr) {
                    if (!seenSager.has(sagsnr)) {
                        seenSager.set(sagsnr, new Set());
                    }
                    const seen = seenSager.get(sagsnr);
                    // Duplicate only if we've seen this sagsnr with a DIFFERENT date/udvalg
                    if (seen.size > 0 && !seen.has(key)) {
                        isDupe = true;
                    }
                    seen.add(key);
                }

                if (isDupe) {
                    el.remove();
                } else {
                    resultCount++;
                    const numEl = document.createElement('span');
                    numEl.className = 'result-num';
                    numEl.textContent = resultCount + '.';
                    el.prepend(numEl);
                }
            });

            e.detail.serverResponse = temp.innerHTML;
        });

        function toggleResult(el) {
            el.closest('.result').classList.toggle('open');
        }

        function switchTab(btn, tabId) {
            const nav = btn.closest('.context-nav');
            nav.querySelectorAll('.context-tab').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            nav.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
            const pane = nav.querySelector('#' + tabId);
            if (pane) pane.style.display = 'block';
        }

        function selectItem(el, punktId, q) {
            const nav = el.closest('.context-nav');
            const content = nav.closest('.result-content');

            // Update active state across all tabs - activate matching punktId, deactivate others
            nav.querySelectorAll('.context-toc a').forEach(a => {
                if (a.dataset.punktId === punktId) {
                    a.classList.add('active');
                } else {
                    a.classList.remove('active');
                }
            });

            // Fetch and replace content
            const mainContent = content.querySelector('.main-content');
            if (mainContent) {
                const currentHeight = mainContent.offsetHeight;
                mainContent.style.minHeight = currentHeight + 'px';
                mainContent.innerHTML = 'Indlæser...';
                fetch('/item-content/' + punktId + '?q=' + encodeURIComponent(q))
                    .then(r => r.text())
                    .then(html => {
                        mainContent.innerHTML = html;
                        mainContent.style.minHeight = '';
                    });
            }
        }

        // AI Agent integration
        let agentController = null;

        document.body.addEventListener('htmx:beforeRequest', (e) => {
            // Only trigger for search requests (not load-more)
            if (e.detail.target.id !== 'results') return;

            // Abort previous agent request
            if (agentController) agentController.abort();

            const q = document.getElementById('search-input').value.trim();
            if (!q) return;

            // Show AI answer box with loading state
            const aiBox = document.getElementById('ai-answer');
            const stepsEl = aiBox.querySelector('.ai-answer-steps');
            const contentEl = aiBox.querySelector('.ai-answer-content');

            aiBox.style.display = 'block';
            aiBox.classList.add('loading');
            stepsEl.textContent = 'Tænker...';
            contentEl.textContent = '';

            // Start agent SSE
            agentController = new AbortController();
            fetch('/agent?q=' + encodeURIComponent(q), { signal: agentController.signal })
                .then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    function read() {
                        reader.read().then(({ done, value }) => {
                            if (done) return;

                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split('\\n');
                            buffer = lines.pop();

                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    try {
                                        const data = JSON.parse(line.slice(6));
                                        if (data.type === 'step') {
                                            stepsEl.textContent = data.content;
                                        } else if (data.type === 'answer_chunk') {
                                            // Stream chunks - accumulate and render
                                            stepsEl.textContent = '';
                                            if (!contentEl.dataset.streaming) {
                                                contentEl.dataset.streaming = '';
                                            }
                                            contentEl.dataset.streaming += data.content;
                                            contentEl.innerHTML = marked.parse(contentEl.dataset.streaming);
                                        } else if (data.type === 'answer') {
                                            aiBox.classList.remove('loading');
                                            stepsEl.textContent = '';
                                            contentEl.innerHTML = marked.parse(data.content);
                                            delete contentEl.dataset.streaming;
                                        } else if (data.type === 'error') {
                                            aiBox.classList.remove('loading');
                                            stepsEl.textContent = '';
                                            contentEl.innerHTML = '<span style="color:red">Fejl: ' + data.content + '</span>';
                                        } else if (data.type === 'done') {
                                            aiBox.classList.remove('loading');
                                            delete contentEl.dataset.streaming;
                                        }
                                    } catch (e) {}
                                }
                            }
                            read();
                        });
                    }
                    read();
                })
                .catch(e => {
                    if (e.name !== 'AbortError') {
                        aiBox.classList.remove('loading');
                        contentEl.textContent = 'Kunne ikke hente AI-svar';
                    }
                });
        });
    </script>
</body>
</html>"""


def result_html(item: dict, query: str, tokens: list[str]) -> str:
    """Render a single search result."""
    payload = item["payload"]
    punkt_id = item["punkt_id"]
    title = highlight_terms(payload.get("title", "Uden titel"), tokens)
    udvalg = esc(payload.get("udvalg", ""))
    dt = format_date(payload.get("datetime", "")[:10]) if payload.get("datetime") else ""
    sagsnummer = payload.get("sagsnummer")
    content = payload.get("content_md", "")

    meta = f"{dt} · {udvalg}"
    if sagsnummer:
        meta += f" · sagsnr.: {esc(sagsnummer)}"

    snippet = extract_snippet(tokens, content)
    snippet_html = highlight_terms(snippet, tokens) if snippet else ""

    # For dedupe: same sagsnummer only counts as duplicate if date OR udvalg differs
    raw_date = payload.get("datetime", "")[:10] if payload.get("datetime") else ""
    data_attrs = ""
    if sagsnummer:
        data_attrs = f' data-sagsnummer="{esc(sagsnummer)}" data-date="{esc(raw_date)}" data-udvalg="{udvalg}"'

    return f"""
    <div class="result"{data_attrs}>
        <div class="result-header" onclick="toggleResult(this)"
             hx-get="/content/{punkt_id}?q={esc(query)}"
             hx-target="next .result-content"
             hx-trigger="click once"
             hx-swap="innerHTML">
            <div class="result-meta">{highlight_terms(meta, tokens)}</div>
            <div class="result-title">{title}</div>
            <div class="result-snippet">{snippet_html}</div>
        </div>
        <div class="result-content">Indlæser...</div>
    </div>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


@app.get("/search", response_class=HTMLResponse)
async def search_results(q: str = Query(""), offset: int = Query(0)):
    if not q.strip():
        return "<p class='status'>Indtast en søgetekst for at finde dagsordenpunkter.</p>"

    try:
        # Use query words for snippet scoring/highlighting
        tokens = query_terms(q)

        result = search(
            query=q,
            limit=16,
            prefetch_limit=1024,
            offset=offset,
        )
        items = result["items"]
        has_more = result["has_more"]
    except Exception as e:
        return f"<p class='status'>Fejl: {esc(str(e))}</p>"

    if not items:
        return "<p class='status'>Ingen resultater fundet.</p>"

    html_parts = []

    for item in items:
        html_parts.append(result_html(item, q, tokens))

    if has_more:
        next_offset = offset + 10
        html_parts.append(f"""
            <div class="load-sentinel"
                 hx-get="/search?q={esc(q)}&offset={next_offset}"
                 hx-trigger="revealed"
                 hx-swap="outerHTML">
            </div>""")

    return "\n".join(html_parts)


@app.get("/agent")
async def agent_stream(q: str = Query("")):
    """SSE endpoint for agent responses with step streaming."""
    if not q.strip():
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'content': 'Tom query'})}\n\n"]),
            media_type="text/event-stream",
        )

    def generate():
        try:
            # Agent searches itself - no history in search.py context
            agent_query = f"Du er en søgeagent, undersøg følgende emne: {q}"
            answer_content = ""
            for update in run_agent(agent_query, stream=True):
                if update["type"] == "tool_call":
                    # Convert tool_call to step format for frontend with full details
                    args_str = json.dumps(update["args"], ensure_ascii=False)
                    step_msg = f"{update['name']}({args_str})"
                    yield f"data: {json.dumps({'type': 'step', 'content': step_msg}, ensure_ascii=False)}\n\n"
                elif update["type"] == "answer_chunk":
                    # Accumulate streamed answer and send chunks
                    answer_content += update["content"]
                    yield f"data: {json.dumps({'type': 'answer_chunk', 'content': update['content']}, ensure_ascii=False)}\n\n"
                elif update["type"] == "answer_done":
                    # Send complete answer for clients that need full content
                    yield f"data: {json.dumps({'type': 'answer', 'content': answer_content}, ensure_ascii=False)}\n\n"
                elif update["type"] == "answer":
                    # Fallback for non-streamed answer
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/content/{punkt_id}", response_class=HTMLResponse)
async def get_content(punkt_id: str, q: str = Query(""), with_case: bool = Query(True)):
    """Lazy load full content and optionally case navigation."""
    from src.retriever import get_point

    point = get_point(punkt_id)
    if not point:
        return "<p>Ikke fundet</p>"

    payload = point["payload"]
    content = payload.get("content_md", "")
    links = payload.get("links", [])
    url = payload.get("url", "")

    html_parts = []

    # Context navigation (lazy loaded) - only on top level, not recursive
    if with_case:
        html_parts.append(f"""
            <div hx-get="/context/{punkt_id}?q={esc(q)}"
                 hx-trigger="load"
                 hx-swap="outerHTML">
            </div>""")

    # Main content (including original link so it gets swapped together)
    main_html = md_to_html(content, links)
    if url:
        main_html += f'<p class="original-link"><a href="{esc(url)}" target="_blank">Vis på dagsordener.aarhus.dk</a></p>'
    html_parts.append(f'<div class="main-content">{main_html}</div>')

    return "\n".join(html_parts)


@app.get("/context/{punkt_id}", response_class=HTMLResponse)
async def get_context_nav(punkt_id: str, q: str = Query("")):
    """Lazy load context navigation with tabs for case and meeting."""
    from src.retriever import get_point

    point = get_point(punkt_id)
    if not point:
        return ""

    payload = point["payload"]
    sagsnummer = payload.get("sagsnummer")
    meeting_id = payload.get("meeting_id")

    # Get case points if sagsnummer exists
    case_points = []
    if sagsnummer:
        try:
            case_points = get_case(sagsnummer)
        except Exception:
            pass

    # Get meeting points if meeting_id exists
    meeting_points = []
    if meeting_id:
        try:
            meeting_points = get_meeting(meeting_id)
        except Exception:
            pass

    # For case: only count as "same case" if date OR udvalg differs
    # (same sagsnr + same date + same udvalg = not a real case, just reused sagsnr)
    current_date = payload.get("datetime", "")[:10] if payload.get("datetime") else ""
    current_udvalg = payload.get("udvalg", "")

    # Filter case_points to only include those with different date OR different udvalg
    real_case_points = [
        p for p in case_points
        if p["payload"].get("datetime", "")[:10] != current_date
        or p["payload"].get("udvalg", "") != current_udvalg
        or p["punkt_id"] == punkt_id  # always include current
    ]

    has_case = len(real_case_points) > 1
    has_meeting = len(meeting_points) > 1

    if not has_case and not has_meeting:
        return ""

    # Build case TOC
    case_toc = ""
    if has_case:
        toc_items = []
        for p in real_case_points:
            pp = p["payload"]
            dt = format_date(pp.get("datetime", "")[:10]) if pp.get("datetime") else ""
            udvalg = esc(pp.get("udvalg", ""))
            title = esc(pp.get("title", "Uden titel"))
            is_current = p["punkt_id"] == punkt_id
            active_class = ' class="active"' if is_current else ''
            toc_items.append(
                f'<li>behandling d. {dt} i {udvalg}: <a{active_class} data-punkt-id="{p["punkt_id"]}" onclick="selectItem(this, \'{p["punkt_id"]}\', \'{esc(q)}\')">{title}</a></li>'
            )
        case_toc = f'<ol class="context-toc" reversed>{"".join(toc_items)}</ol>'

    # Build meeting TOC
    meeting_toc = ""
    if has_meeting:
        udvalg = esc(meeting_points[0]["payload"].get("udvalg", "")) if meeting_points else ""
        dt = format_date(meeting_points[0]["payload"].get("datetime", "")[:10]) if meeting_points and meeting_points[0]["payload"].get("datetime") else ""
        toc_items = []
        for p in meeting_points:
            pp = p["payload"]
            title = esc(pp.get("title", "Uden titel"))
            is_current = p["punkt_id"] == punkt_id
            active_class = ' class="active"' if is_current else ''
            toc_items.append(
                f'<li><a{active_class} data-punkt-id="{p["punkt_id"]}" onclick="selectItem(this, \'{p["punkt_id"]}\', \'{esc(q)}\')">{title}</a></li>'
            )
        meeting_toc = f'<ol class="context-toc">{"".join(toc_items)}</ol>'

    # Build tabs (meeting first, case second, but case is active by default)
    tabs = []
    panes = []

    if has_meeting:
        meeting_active = " active" if not has_case else ""
        meeting_display = " style=\"display:none\"" if has_case else ""
        tabs.append(f'<button class="context-tab{meeting_active}" onclick="switchTab(this, \'meeting-pane\')">Møde: {dt}, {udvalg}</button>')
        panes.append(f'<div class="tab-pane" id="meeting-pane"{meeting_display}>{meeting_toc}</div>')

    if has_case:
        case_active = " active" if has_case else ""
        case_display = ""
        tabs.append(f'<button class="context-tab{case_active}" onclick="switchTab(this, \'case-pane\')">Sag: {esc(sagsnummer)}</button>')
        panes.append(f'<div class="tab-pane" id="case-pane"{case_display}>{case_toc}</div>')

    return f'''<div class="context-nav">
<div class="context-tabs">{"".join(tabs)}</div>
<div class="context-content">{"".join(panes)}</div>
</div>'''


@app.get("/item-content/{punkt_id}", response_class=HTMLResponse)
async def get_item_content(punkt_id: str, q: str = Query("")):
    """Get content for a case item (without case nav, just content + link)."""
    from src.retriever import get_point

    point = get_point(punkt_id)
    if not point:
        return "<p>Ikke fundet</p>"

    payload = point["payload"]
    content = payload.get("content_md", "")
    links = payload.get("links", [])
    url = payload.get("url", "")

    html_parts = []
    html_parts.append(md_to_html(content, links))

    if url:
        html_parts.append(f'<p class="original-link"><a href="{esc(url)}" target="_blank">Se på dagsordener.aarhus.dk</a></p>')

    return "\n".join(html_parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
