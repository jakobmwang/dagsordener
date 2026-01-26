"""Chainlit chat app for agenda items - uses shared agent module."""

import json
import os

# Set default URLs for local development (outside Docker)
if not os.getenv("QDRANT_URL"):
    os.environ["QDRANT_URL"] = "http://localhost:6333"
if not os.getenv("FLAGSERVE_URL"):
    os.environ["FLAGSERVE_URL"] = "http://localhost:8273"

import chainlit as cl

from src.agent import run_agent


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    history = cl.user_session.get("history", [])

    # Create response message for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Run agent with history and streaming
    final_content = ""
    tool_text = ""
    for update in run_agent(message.content, history, stream=True):
        if update["type"] == "tool_call":
            name = update["name"]
            args = json.dumps(update["args"], ensure_ascii=False)
            tool_text += f"*Søger: {name}({args})...*\n\n"
            await msg.stream_token(f"*Søger: {name}({args})...*\n\n")
        elif update["type"] == "answer_chunk":
            # Clear tool text on first answer chunk and stream answer
            if tool_text and not final_content:
                msg.content = ""
                await msg.update()
            final_content += update["content"]
            await msg.stream_token(update["content"])
        elif update["type"] == "answer":
            # Fallback for non-streamed answer
            final_content = update["content"]

    # Final update
    if not final_content:
        msg.content = "Jeg kunne ikke finde relevante oplysninger."
        await msg.update()

    # Save to history
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_content})
    cl.user_session.set("history", history)
