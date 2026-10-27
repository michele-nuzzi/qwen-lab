// The part of Qwen3.5's chat template (chat_template.jinja at the pinned revision) that a text-only chat
// uses: no system prompt, tools or images. Checked against the jinja original by scripts/prompt.py.

export interface ChatMessage {
  role: 'user' | 'assistant';
  /** For assistant turns, the answer only; earlier reasoning is dropped from the context, as Qwen's template does. */
  content: string;
}

export const THINK_END = '</think>';

export function renderPrompt(messages: ChatMessage[], thinking: boolean): string {
  if (!messages.length || messages[messages.length - 1].role !== 'user') {
    throw new Error('The conversation must end with a user message.');
  }
  let out = '';
  for (const m of messages) {
    let content = m.content.trim();
    if (m.role === 'assistant' && content.includes(THINK_END)) {
      content = content.split(THINK_END).pop()!.replace(/^\n+/, '');
    }
    out += `<|im_start|>${m.role}\n${content}<|im_end|>\n`;
  }
  return out + '<|im_start|>assistant\n' + (thinking ? '<think>\n' : '<think>\n\n</think>\n\n');
}
