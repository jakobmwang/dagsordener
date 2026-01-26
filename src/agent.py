"""Agent for answering questions about agenda items using OpenAI function calling."""

import json
import os
import re
from pathlib import Path
from typing import Generator

from openai import AzureOpenAI

from src.retriever import get_case, get_meeting, get_point, search

# Load knowledge base
KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge.yaml"
KNOWLEDGE = KNOWLEDGE_PATH.read_text() if KNOWLEDGE_PATH.exists() else ""

# Azure OpenAI client (lazy init)
_client: AzureOpenAI | None = None

MODEL = "gpt-oss-120b"


def get_client() -> AzureOpenAI:
    """Get or create Azure OpenAI client."""
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
    return _client


SYSTEM_PROMPT = f"""Du er ByrådsGPT, en assistent der hjælper med at besvare spørgsmål om dagsordener og referater fra Aarhus Kommune.

## Grundviden om Aarhus Byråd
{KNOWLEDGE}

## Arbejdsgang
VIGTIGT: Du SKAL ALTID kalde search() funktionen FØRST for at finde information i databasen.

### Søgestrategi - vær GRUNDIG
1. **Søg bredt først**: Start med brede søgetermer for at få overblik over feltet
2. **Følg sagsnumre**: Når du finder et relevant sagsnummer, brug ALTID get_case() til at se ALLE behandlinger af sagen på tværs af udvalg og tid - sager behandles ofte flere gange
3. **Snævr derefter ind**: Lav evt. mere specifikke søgninger baseret på det du har lært
4. **Læs fuldt indhold**: Brug get_point() for at læse hele indholdet af vigtige punkter

### Hvorfor dette er vigtigt
- En sag kan være behandlet i flere udvalg over tid (fx først i fagudvalg, så i Byrådet)
- Beslutninger udvikler sig - det nyeste punkt har ikke nødvendigvis hele historien
- Kontekst fra tidligere behandlinger giver bedre svar

## Output
- Svar på dansk
- Kort og præcist, men fyldestgørende
- Inkluder kilder (udvalg, dato, sagsnummer)
- Beskriv gerne sagsforløbet hvis relevant
- Sig ærligt hvis du ikke finder noget relevant"""

# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Søg i dagsordener og referater med semantisk søgning. Brug denne til at finde relevante punkter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Søgetekst, fx 'cykelstier i Aarhus' eller 'budget 2024'",
                    },
                    "udvalg": {
                        "type": "string",
                        "description": "Filtrer efter udvalg, fx 'Byrådet' eller 'Teknisk Udvalg'",
                    },
                    "date_after": {
                        "type": "string",
                        "description": "Kun punkter efter denne dato (YYYY-MM-DD)",
                    },
                    "date_before": {
                        "type": "string",
                        "description": "Kun punkter før denne dato (YYYY-MM-DD)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Antal resultater (default 8)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case",
            "description": "Hent alle dagsordenpunkter for en sag på tværs af møder. Brug dette til at se hele historikken.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sagsnummer": {
                        "type": "string",
                        "description": "Sagsnummer, fx 'SAG-2024-12345'",
                    },
                },
                "required": ["sagsnummer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meeting",
            "description": "Hent alle dagsordenpunkter fra et møde. Brug dette til at se hele dagsordenen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "Møde-ID",
                    },
                },
                "required": ["meeting_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_point",
            "description": "Hent det fulde indhold af et specifikt dagsordenpunkt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "punkt_id": {
                        "type": "string",
                        "description": "Punkt-ID fra søgeresultater",
                    },
                },
                "required": ["punkt_id"],
            },
        },
    },
]


# === Snippet extraction ===


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
    """Extract query words for snippet scoring."""
    if not query:
        return []
    return re.findall(r"[\w]+(?:[-'][\w]+)*", query)


def extract_snippet(tokens: list[str], content: str, size: int = 600, overlap: int = 100) -> str:
    """Extract the most relevant snippet using keyword density scoring."""
    if not content:
        return ""

    plain = strip_markdown(content)
    if not plain:
        return ""

    if len(plain) <= size:
        return plain

    if not tokens:
        return plain[:size] + " ..."

    term_patterns = [f"\\b{re.escape(t)}" for t in tokens]
    pattern = re.compile("|".join(term_patterns), re.IGNORECASE)

    best_chunk = plain[:size]
    best_score = len(pattern.findall(best_chunk))
    best_start = 0

    step = size - overlap
    for start in range(step, len(plain) - size // 2, step):
        end = min(start + size, len(plain))
        chunk = plain[start:end]

        if len(chunk) < size // 2:
            break

        score = len(pattern.findall(chunk))
        if score > best_score:
            best_score = score
            best_chunk = chunk
            best_start = start

    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if best_start + len(best_chunk) < len(plain) else ""

    return f"{prefix}{best_chunk}{suffix}"


# === Result formatting ===


def format_search_results(result: dict, query: str) -> str:
    """Format search results for the model with focused snippets."""
    items = result.get("items", [])
    if not items:
        return "Ingen resultater fundet."

    tokens = query_terms(query)
    parts = []
    for item in items[:10]:
        payload = item["payload"]
        dt = payload.get("datetime", "")[:10] if payload.get("datetime") else ""
        content = payload.get("content_md", "")
        snippet = extract_snippet(tokens, content, size=600)
        parts.append(
            f"- **{payload.get('title', 'Uden titel')}** ({payload.get('udvalg', '')}, {dt})\n"
            f"  Sagsnr: {payload.get('sagsnummer', 'Ingen')} | punkt_id: {item['punkt_id']}\n"
            f"  {snippet}"
        )
    return f"Fandt {result.get('total', len(items))} resultater:\n\n" + "\n\n".join(parts)


def format_case_results(points: list[dict]) -> str:
    """Format case results for the model."""
    if not points:
        return "Ingen punkter fundet for denne sag."

    parts = []
    for p in points:
        payload = p["payload"]
        dt = payload.get("datetime", "")[:10] if payload.get("datetime") else ""
        parts.append(
            f"## {payload.get('title', 'Uden titel')}\n"
            f"Udvalg: {payload.get('udvalg', '')}, Dato: {dt}\n\n"
            f"{payload.get('content_md', '')[:2000]}"
        )
    return "\n\n---\n\n".join(parts)


def format_meeting_results(points: list[dict]) -> str:
    """Format meeting results for the model."""
    if not points:
        return "Ingen punkter fundet for dette møde."

    first = points[0]["payload"]
    header = f"Møde: {first.get('udvalg', '')} - {first.get('datetime', '')[:10]}\n\n"

    parts = []
    for p in points:
        payload = p["payload"]
        parts.append(
            f"## {payload.get('index', '?')}. {payload.get('title', 'Uden titel')}\n\n"
            f"{payload.get('content_md', '')[:1500]}"
        )
    return header + "\n\n---\n\n".join(parts)


def format_point_result(point: dict | None) -> str:
    """Format a single point for the model."""
    if not point:
        return "Punkt ikke fundet."

    payload = point["payload"]
    dt = payload.get("datetime", "")[:10] if payload.get("datetime") else ""
    return (
        f"# {payload.get('title', 'Uden titel')}\n"
        f"Udvalg: {payload.get('udvalg', '')}, Dato: {dt}\n"
        f"Sagsnummer: {payload.get('sagsnummer', 'Ingen')}\n\n"
        f"{payload.get('content_md', '')}"
    )


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return formatted result."""
    try:
        if name == "search":
            udvalg = args.get("udvalg") or None
            date_after = args.get("date_after") or None
            date_before = args.get("date_before") or None
            limit = args.get("limit") or 8

            query = args["query"]
            result = search(
                query=query,
                udvalg=udvalg,
                date_after=date_after,
                date_before=date_before,
                limit=limit,
            )
            return format_search_results(result, query)
        elif name == "get_case":
            points = get_case(args["sagsnummer"])
            return format_case_results(points)
        elif name == "get_meeting":
            points = get_meeting(args["meeting_id"])
            return format_meeting_results(points)
        elif name == "get_point":
            point = get_point(args["punkt_id"])
            return format_point_result(point)
        else:
            return f"Ukendt tool: {name}"
    except Exception as e:
        return f"Fejl ved {name}: {e}"


# === Agent runner ===


def _stream_final_answer(client: AzureOpenAI, messages: list[dict]) -> Generator[dict, None, None]:
    """Stream the final answer token by token."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )

    has_content = False
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            has_content = True
            yield {"type": "answer_chunk", "content": chunk.choices[0].delta.content}

    if not has_content:
        yield {"type": "answer_chunk", "content": "Jeg kunne ikke finde relevante oplysninger."}

    yield {"type": "answer_done"}


def run_agent(
    query: str,
    history: list[dict] | None = None,
    max_iterations: int = 6,
    stream: bool = False,
) -> Generator[dict, None, None]:
    """
    Run the agent with a query.

    Args:
        query: User question
        history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
        max_iterations: Max tool call iterations
        stream: If True, stream the final answer token by token

    Yields:
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "answer", "content": "..."}  (full answer if stream=False)
        {"type": "answer_chunk", "content": "..."}  (token chunks if stream=True)
        {"type": "answer_done"}  (signals end of streaming)
        {"type": "error", "content": "..."}
    """
    client = get_client()

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    for iteration in range(max_iterations):
        is_last = iteration >= max_iterations - 1

        if is_last:
            # Force conclusion without tools
            if stream:
                yield from _stream_final_answer(
                    client,
                    messages + [{"role": "user", "content": "Du har nu søgt nok. Giv venligst dit bedste svar baseret på det du har fundet."}],
                )
            else:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages + [{"role": "user", "content": "Du har nu søgt nok. Giv venligst dit bedste svar baseret på det du har fundet."}],
                )
                yield {"type": "answer", "content": response.choices[0].message.content or "Jeg kunne ikke finde relevante oplysninger."}
            return

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Debug logging
        print(f"[AGENT DEBUG] iteration={iteration}, tool_calls={bool(assistant_message.tool_calls)}, content_len={len(assistant_message.content or '')}, content_preview={repr((assistant_message.content or '')[:200])}")

        if assistant_message.tool_calls:
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                yield {"type": "tool_call", "name": func_name, "args": func_args}

                result = execute_tool(func_name, func_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue

        # Final answer (no tool calls)
        final_content = assistant_message.content

        # If empty response after tool calls, force a conclusion
        if not final_content and len(messages) > 2:
            if stream:
                yield from _stream_final_answer(
                    client,
                    messages + [{"role": "user", "content": "Opsummer venligst hvad du har fundet baseret på søgeresultaterne ovenfor."}],
                )
            else:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages + [{"role": "user", "content": "Opsummer venligst hvad du har fundet baseret på søgeresultaterne ovenfor."}],
                )
                final_content = response.choices[0].message.content
                yield {"type": "answer", "content": final_content or "Jeg kunne ikke finde relevante oplysninger."}
            return

        # Stream or return the final answer
        if stream:
            # We already have the content, yield it as chunks (simulated streaming)
            yield {"type": "answer_chunk", "content": final_content or "Jeg kunne ikke finde relevante oplysninger."}
            yield {"type": "answer_done"}
        else:
            yield {"type": "answer", "content": final_content or "Jeg kunne ikke finde relevante oplysninger."}
        return

    # Max iterations reached
    if stream:
        yield {"type": "answer_chunk", "content": "Jeg nåede ikke frem til et svar inden for det tilladte antal forsøg."}
        yield {"type": "answer_done"}
    else:
        yield {"type": "answer", "content": "Jeg nåede ikke frem til et svar inden for det tilladte antal forsøg."}


def ask(query: str, history: list[dict] | None = None) -> str:
    """
    Simple interface - run agent and return final answer.

    Args:
        query: User question
        history: Optional conversation history

    Returns:
        Final answer string
    """
    answer = ""
    for update in run_agent(query, history):
        if update["type"] == "answer":
            answer = update["content"]
        elif update["type"] == "error":
            answer = f"Fejl: {update['content']}"
    return answer


if __name__ == "__main__":
    # Test
    print("Testing agent...")
    for update in run_agent("Hvad er der besluttet om cykelstier?"):
        if update["type"] == "tool_call":
            print(f"  -> {update['name']}({update['args']})")
        elif update["type"] == "answer":
            print(f"\nSvar: {update['content']}")
