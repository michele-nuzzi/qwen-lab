"""Render Qwen3.5's chat template with jinja2 (the reference for src/lib/chat.ts)."""

from __future__ import annotations

import jinja2


def render_chat(template: str, messages: list[dict], thinking: bool = False) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=True, enable_thinking=thinking)
