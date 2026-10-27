import './styles.css';
import type { ChatMessage } from './lib/chat';
import { describeApplied } from './lib/interventions';
import type { GenerationState, LoadProgress } from './lib/model';
import type { FromWorker, GenSettings } from './protocol';
import * as client from './ui/client';
import { $, el, errorText } from './ui/dom';
import { initFinder, setFinderEnabled } from './ui/finder';
import { initLab, lab } from './ui/lab';
import { renderMarkdown } from './ui/markdown';
import { initNeuronDialog, showNeuron } from './ui/neuron';
import { initPresets } from './ui/presets';
import { currentView, setHashParams, type View } from './ui/route';
import { DEFAULTS, loadSettings, parseStrengths, saveSettings, type Settings } from './ui/settings';

type Mode = 'modified' | 'baseline' | 'compare';

interface Reply {
  /** 0 = the unmodified model */
  strength: number;
  root: HTMLElement;
  body: HTMLElement;
  thinking: HTMLDetailsElement;
  thinkingBody: HTMLElement;
  foot: HTMLElement;
  /** The answer so far (without reasoning); what goes into later turns' context. */
  answer: string;
  sawAnswer: boolean;
}

/** One user message and its replies: one in Modified / Baseline mode, one per strength in Compare mode. */
interface Turn {
  user: string;
  replies: Reply[];
}

const SUGGESTIONS = [
  'What do you think about humans?',
  'Give me three tips for a long-lasting marriage.',
  'Write a haiku about the ocean.',
  'What makes a good leader?',
  "What's your honest opinion of me?",
];
const PLACEHOLDERS: Record<Mode, string> = {
  modified: 'Message the modified model…',
  baseline: 'Message the unmodified model…',
  compare: 'Ask every column…',
};
const LINUX_CHROME_HINT =
  'In Chrome on Linux, turn on chrome://flags/#enable-unsafe-webgpu and chrome://flags/#ignore-gpu-blocklist, ' +
  'relaunch, and check that chrome://gpu says "WebGPU: Hardware accelerated".';

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(2)} GB`;
const duration = (s: number) => (s < 90 ? `${Math.ceil(s)} s` : `${Math.round(s / 60)} min`);
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const strengthLabel = (s: number) => `×${Number(s.toFixed(2))}`;

const ui = {
  modelStatus: $('model-status'),
  loader: $('loader'),
  gpuStatus: $('gpu-status'),
  cacheStatus: $('cache-status'),
  gpuNote: $('gpu-note'),
  loadButton: $<HTMLButtonElement>('load-button'),
  loadError: $('load-error'),
  progress: $('progress'),
  progressPhase: $('progress-phase'),
  progressDetail: $('progress-detail'),
  progressBar: $('progress-bar'),
  chat: $('chat'),
  messages: $('messages'),
  empty: $('empty'),
  suggestions: $('suggestions'),
  composer: $<HTMLFormElement>('composer'),
  prompt: $<HTMLTextAreaElement>('prompt'),
  send: $<HTMLButtonElement>('send'),
  stop: $<HTMLButtonElement>('stop'),
  newChat: $<HTMLButtonElement>('new-chat'),
  summary: $('settings-summary'),
  settings: $<HTMLDialogElement>('settings'),
  temperature: $<HTMLInputElement>('temperature'),
  maxTokens: $<HTMLInputElement>('max-tokens'),
  thinking: $<HTMLInputElement>('thinking'),
  compare: $<HTMLInputElement>('compare'),
  compareError: $('compare-error'),
};

let settings: Settings = loadSettings();
let mode: Mode = 'modified';
let turns: Turn[] = [];
let modelReady = false;
/** generating or tracing: both need the GPU session */
let busy = false;
let stopAll = false;

// ---------------------------------------------------------------- pages

function applyView() {
  const view = currentView();
  document.body.dataset.view = view; // each page sets its own width
  for (const v of document.querySelectorAll<HTMLElement>('.view')) v.hidden = v.dataset.view !== view;
  for (const t of document.querySelectorAll<HTMLElement>('.tab')) {
    if (t.dataset.view === view) t.setAttribute('aria-current', 'page');
    else t.removeAttribute('aria-current');
  }
  ui.loader.hidden = modelReady || view === 'about';
}

function showView(view: View) {
  if (view === currentView()) return;
  setHashParams({ view: view === 'chat' ? null : view }, true);
  applyView();
  window.scrollTo({ top: 0 });
}

// tabs, the brand and in-text links all switch pages
for (const link of document.querySelectorAll<HTMLElement>('button[data-view], a[data-view]')) {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    showView(link.dataset.view as View);
  });
}

function setModelStatus(text: string, state: 'idle' | 'busy' | 'ok' | 'bad') {
  ui.modelStatus.textContent = text;
  ui.modelStatus.dataset.state = state;
}
window.addEventListener('popstate', applyView);
window.addEventListener('hashchange', applyView);

// ---------------------------------------------------------------- loading

let cachedAtProbe = false;
let downloadSize = 0;
/** Adapter + model build, the key for remembering how long kernel compilation took on this device. */
let gpuKey = '';
let softwareGpu = false;
let speed: { t: number; bytes: number }[] = [];
let tick = 0;
let warmupStart = 0;
const WARMUP_KEY = 'evil-qwen:warmup-seconds';

function warmupHistory(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(WARMUP_KEY) ?? '{}');
  } catch {
    return {};
  }
}

function rememberWarmup(seconds: number) {
  try {
    localStorage.setItem(WARMUP_KEY, JSON.stringify({ ...warmupHistory(), [gpuKey]: Math.round(seconds) }));
  } catch {
    // no storage: the next load just shows no estimate
  }
}

function showNote(message: string) {
  ui.gpuNote.textContent = message;
  ui.gpuNote.hidden = false;
}

function showLoadError(message: string) {
  ui.loadError.textContent = message;
  ui.loadError.hidden = false;
}

function onProbe(m: Extract<FromWorker, { type: 'probe' }>) {
  if (!m.gpu) setModelStatus('no WebGPU', 'bad');
  cachedAtProbe = m.cached;
  downloadSize = m.downloadBytes;
  gpuKey = m.gpu ? `${m.gpu.name}|${m.gpu.variant}` : '';
  softwareGpu = !!m.gpu?.fallback;
  $('download-size').textContent = gb(downloadSize);
  ui.gpuNote.hidden = true;
  if (!m.gpu) {
    ui.gpuStatus.textContent = m.gpuError ?? 'unavailable';
    ui.gpuStatus.className = 'bad';
    showLoadError(`This page needs WebGPU. Use a recent desktop Chrome, Edge or Safari. ${LINUX_CHROME_HINT}`);
  } else if (m.gpu.fallback) {
    ui.gpuStatus.textContent = `${m.gpu.name} (software emulation, not your GPU)`;
    ui.gpuStatus.className = 'bad';
    showNote(
      'Your browser gave this page a software WebGPU adapter that emulates a GPU on the CPU, so the model would ' +
        `be far too slow to use (minutes to start, then about a minute per word). ${LINUX_CHROME_HINT}`,
    );
  } else {
    ui.gpuStatus.textContent = m.gpu.variant === 'q4f16' ? m.gpu.name : `${m.gpu.name} (no 16-bit shaders)`;
    ui.gpuStatus.className = m.gpu.variant === 'q4f16' ? 'ok' : 'warn';
    if (m.gpu.variant === 'q4') {
      showNote(
        "This GPU or browser can't use 16-bit floats in WebGPU shaders, so the page uses the 32-bit build of the " +
          'model: a bigger download and slower generation.',
      );
    }
  }
  ui.cacheStatus.textContent = m.cached ? 'downloaded, in this browser' : `not downloaded yet (${gb(downloadSize)})`;
  ui.cacheStatus.className = m.cached ? 'ok' : '';
  ui.loadButton.textContent =
    (m.gpu?.fallback ? 'Try anyway: ' : '') + (m.cached ? 'Load model' : `Download and load (${gb(downloadSize)})`);
  ui.loadButton.disabled = !m.gpu;
}

function setProgress(label: string, detail: string, fraction: number | null) {
  ui.progress.hidden = false;
  ui.progressPhase.textContent = label;
  ui.progressDetail.textContent = detail;
  ui.progress.classList.toggle('indeterminate', fraction === null);
  ui.progressBar.style.width = fraction === null ? '' : `${(fraction * 100).toFixed(1)}%`;
}

/** Phases without a byte count re-render every second, so the elapsed time visibly counts up. */
function ticking(render: () => void) {
  clearInterval(tick);
  render();
  tick = window.setInterval(render, 1000);
}

function onProgress(p: LoadProgress) {
  if (p.phase === 'tokenizer' || p.phase === 'session') {
    const label = p.phase === 'tokenizer' ? 'Loading the tokenizer…' : 'Uploading the weights to the GPU…';
    const t0 = performance.now();
    return ticking(() => setProgress(label, clock((performance.now() - t0) / 1000), null));
  }
  if (p.phase === 'warmup') {
    // Shader compilation reports no progress and its speed depends on the GPU, driver and browser, so the
    // estimate is how long it took last time on this device.
    warmupStart ||= performance.now();
    const label = `Compiling GPU kernels, step ${p.step} of ${p.steps}…`;
    const last = warmupHistory()[gpuKey];
    return ticking(() => {
      const elapsed = (performance.now() - warmupStart) / 1000;
      if (!last) {
        const usual = softwareGpu ? 'software emulation: several minutes' : 'usually well under a minute';
        return setProgress(label, `${clock(elapsed)} · ${usual}`, null);
      }
      const eta =
        elapsed < last
          ? `about ${duration(last - elapsed)} left (took ${duration(last)} last time)`
          : `longer than last time (${duration(last)})`;
      setProgress(label, `${clock(elapsed)} · ${eta}`, Math.min(elapsed / last, 0.97));
    });
  }
  clearInterval(tick);
  const loaded = p.loaded ?? 0;
  const total = p.total ?? 1;
  const now = performance.now();
  speed.push({ t: now, bytes: loaded });
  while (speed.length > 2 && now - speed[0].t > 5000) speed.shift();
  const dt = (now - speed[0].t) / 1000;
  const rate = dt > 1 ? (loaded - speed[0].bytes) / dt : 0;
  let detail = `${gb(loaded)} of ${gb(total)}`;
  if (rate > 0 && !cachedAtProbe) detail += ` · ${(rate / 1e6).toFixed(0)} MB/s · ${duration((total - loaded) / rate)} left`;
  setProgress(cachedAtProbe ? 'Reading the model from browser storage…' : 'Downloading the model…', detail, loaded / total);
}

ui.loadButton.addEventListener('click', async () => {
  setModelStatus('loading…', 'busy');
  ui.loadButton.hidden = true;
  ui.loadError.hidden = true;
  speed = [];
  warmupStart = 0;
  setProgress('Starting…', '', null);
  // ask the browser not to evict the multi-GB cache under storage pressure (best effort)
  await navigator.storage?.persist?.().catch(() => false);
  client.post({ type: 'load', baseUrl: document.baseURI });
});

function onReady(m: Extract<FromWorker, { type: 'ready' }>) {
  clearInterval(tick);
  if (warmupStart) rememberWarmup((performance.now() - warmupStart) / 1000);
  modelReady = true;
  setModelStatus('model ready', 'ok');
  ui.loader.hidden = true;
  ui.chat.hidden = false;
  ui.composer.hidden = false;
  setFinderEnabled(true);
  ui.modelStatus.title = `Loaded ${m.cached ? 'from browser storage' : 'from Hugging Face'} in ${duration(m.seconds)}`;
  ui.prompt.focus();
}

function onLoadFailed(message: string) {
  clearInterval(tick);
  if (modelReady) return alert(message);
  setModelStatus('not loaded', 'bad');
  ui.progress.hidden = true;
  showLoadError(`Loading failed: ${message}`);
  ui.loadButton.hidden = false;
  ui.loadButton.disabled = false;
  ui.loadButton.textContent = 'Try again';
}

client.listen((m) => {
  switch (m.type) {
    case 'probe':
      return onProbe(m);
    case 'progress':
      return onProgress(m.progress);
    case 'ready':
      return onReady(m);
    case 'error':
      return onLoadFailed(m.message);
    case 'cache-cleared':
      cachedAtProbe = false;
      ui.cacheStatus.textContent = `not downloaded yet (${gb(downloadSize)})`;
      ui.cacheStatus.className = '';
      if (!modelReady) ui.loadButton.textContent = `Download and load (${gb(downloadSize)})`;
      alert(
        modelReady
          ? 'The downloaded model was deleted from this browser. It stays loaded until you close the page.'
          : 'The downloaded model was deleted from this browser.',
      );
      return;
  }
});

// ---------------------------------------------------------------- chat rendering

const nearBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
const scrollToBottom = () => window.scrollTo({ top: document.documentElement.scrollHeight });

function makeReply(strength: number, specs: string | null, showStrength: boolean): Reply {
  const modified = strength > 0;
  const root = el('article', `reply ${modified ? 'steered' : 'baseline'}`);
  const head = el('header', 'reply-head');
  head.append(el('span', 'badge', modified ? `🧪 Modified${showStrength ? ` ${strengthLabel(strength)}` : ''}` : '🙂 Baseline'));
  if (modified && specs) head.append(el('span', 'reply-specs', specs));
  const thinking = el('details', 'thinking');
  thinking.hidden = true;
  const thinkingBody = el('div', 'thinking-body md');
  thinking.append(el('summary', '', 'Thinking…'), thinkingBody);
  const body = el('div', 'reply-body md');
  body.append(el('span', 'typing', ''));
  const foot = el('footer', 'reply-foot', 'waiting…');
  root.append(head, thinking, body, foot);
  return { strength, root, body, thinking, thinkingBody, foot, answer: '', sawAnswer: false };
}

function paint(r: Reply, s: GenerationState, final: boolean) {
  if (s.reasoning !== null) {
    r.thinking.hidden = false;
    r.thinking.querySelector('summary')!.textContent = s.answering ? 'Reasoning' : 'Thinking…';
    renderMarkdown(r.thinkingBody, s.reasoning || '…');
    if (!s.answering) r.thinking.open = true;
    else if (!r.sawAnswer) r.thinking.open = false; // collapse once, when the answer starts
  }
  r.sawAnswer ||= s.answering;
  r.answer = s.answer;
  renderMarkdown(r.body, s.answer);
  r.body.classList.toggle('streaming', !final);
  if (!final && !s.answer) r.body.append(el('span', 'typing', ''));

  const parts = [`${s.tokens} tokens`];
  if (s.tokensPerSecond >= 0.5) parts.push(`${s.tokensPerSecond.toFixed(1)} tok/s`);
  else if (s.tokensPerSecond) parts.push(`${(1 / s.tokensPerSecond).toFixed(0)} s/token`);
  if (s.stopReason === 'length') parts.push('max tokens');
  if (s.stopReason === 'stopped') parts.push('stopped');
  r.foot.replaceChildren(el('span', '', parts.join(' · ')));
  if (final) {
    const more = el('details', 'reply-more');
    more.append(el('summary', '', 'details'), el('div', '', `prompt ${s.promptTokens} tokens, read in ${s.prefillSeconds.toFixed(1)} s`));
    if (s.applied.length) more.append(el('div', 'mono', `edits applied: ${describeApplied(s.applied)}`));
    r.foot.append(more);
  }
}

// coalesce streamed updates to one paint per frame
const dirty = new Map<Reply, GenerationState>();
let frame = 0;
function schedulePaint(r: Reply, s: GenerationState) {
  dirty.set(r, s);
  frame ||= requestAnimationFrame(() => {
    frame = 0;
    const stick = nearBottom();
    for (const [reply, state] of dirty) paint(reply, state, false);
    dirty.clear();
    if (stick) scrollToBottom();
  });
}

function replyFailed(r: Reply, message: string) {
  r.body.classList.remove('streaming');
  r.body.querySelector('.typing')?.remove();
  r.body.append(el('p', 'error-text', message));
  r.foot.textContent = 'failed';
}

// ---------------------------------------------------------------- sending

/** Earlier turns as context. Compare turns pair column k with column k, so each column keeps its own history. */
function contextFor(history: Turn[], column: number, text: string): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  for (const t of history) {
    const r = t.replies[Math.min(column, t.replies.length - 1)];
    if (!r.answer.trim()) continue; // failed or empty replies are left out
    msgs.push({ role: 'user', content: t.user }, { role: 'assistant', content: r.answer });
  }
  msgs.push({ role: 'user', content: text });
  return msgs;
}

function setBusy(on: boolean, generating = false) {
  busy = on;
  ui.send.disabled = on && !generating;
  ui.send.hidden = generating;
  ui.stop.hidden = !generating;
  ui.newChat.disabled = generating;
  setFinderEnabled(modelReady && !on);
}

/** Compare puts the modified columns first and the unmodified model (0) last. */
const compareOrder = (strengths: number[]) => [...strengths.filter((x) => x > 0), ...strengths.filter((x) => x === 0)];

/** Strengths to answer with, one reply each. */
function columns(): number[] | null {
  if (mode === 'baseline') return [0];
  if (mode === 'modified') return [lab.strength()];
  try {
    return compareOrder(parseStrengths(settings.compare));
  } catch (e) {
    openSettings();
    showCompareError(errorText(e));
    return null;
  }
}

async function submit(raw: string) {
  const text = raw.trim();
  if (!text || busy || !modelReady) return;
  const strengths = columns();
  if (!strengths) return;
  if (strengths.some((s) => s > 0) && lab.error()) {
    showView('neurons');
    $<HTMLInputElement>('iv-text').focus();
    return;
  }
  const interventions = lab.interventions();
  const specs = lab.describe();
  ui.prompt.value = '';
  autosize();
  ui.empty.hidden = true;

  const turn: Turn = { user: text, replies: [] };
  const turnEl = el('section', 'turn');
  const row = el('div', strengths.length > 1 ? 'replies compare' : 'replies');
  row.style.setProperty('--cols', String(strengths.length));
  turnEl.append(el('div', 'user', text), row);
  for (const s of strengths) {
    const r = makeReply(s, specs, mode === 'compare');
    turn.replies.push(r);
    row.append(r.root);
  }
  const history = turns;
  turns = [...turns, turn];
  ui.messages.append(turnEl);
  turnEl.scrollIntoView({ block: 'start', behavior: 'smooth' });

  setBusy(true, true);
  stopAll = false;
  try {
    for (const [k, reply] of turn.replies.entries()) {
      if (stopAll) {
        replyFailed(reply, 'Skipped.');
        continue;
      }
      reply.foot.textContent = 'reading the prompt…';
      const gen: GenSettings = {
        interventions,
        strength: reply.strength,
        thinking: settings.thinking,
        maxNewTokens: settings.maxNewTokens,
        temperature: settings.temperature,
      };
      try {
        const final = await client.generate(contextFor(history, k, text), gen, (s) => schedulePaint(reply, s));
        dirty.delete(reply);
        paint(reply, final, true);
      } catch (e) {
        replyFailed(reply, errorText(e));
      }
    }
  } finally {
    setBusy(false);
  }
}

function autosize() {
  ui.prompt.style.height = 'auto';
  ui.prompt.style.height = `${Math.min(ui.prompt.scrollHeight, 220)}px`;
}

ui.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  void submit(ui.prompt.value);
});
ui.prompt.addEventListener('input', autosize);
ui.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    ui.composer.requestSubmit();
  }
});
ui.stop.addEventListener('click', () => {
  stopAll = true;
  client.post({ type: 'stop' });
});

function clearChat() {
  turns = [];
  ui.messages.replaceChildren();
  ui.empty.hidden = false;
}
ui.newChat.addEventListener('click', () => {
  if (!busy) clearChat();
  ui.prompt.focus();
});

for (const s of SUGGESTIONS) {
  const b = el('button', 'chip', s);
  b.type = 'button';
  b.addEventListener('click', () => void submit(s));
  ui.suggestions.append(b);
}

for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
  radio.addEventListener('change', () => {
    const next = radio.value as Mode;
    if ((mode === 'compare') !== (next === 'compare') && turns.length) {
      if (busy || !confirm('Switching to or from Compare starts a new chat. Clear this one?')) {
        document.querySelector<HTMLInputElement>(`input[name="mode"][value="${mode}"]`)!.checked = true;
        return;
      }
      clearChat();
    }
    mode = next;
    ui.prompt.placeholder = PLACEHOLDERS[mode];
    updateSummary();
  });
}

// ---------------------------------------------------------------- settings

function updateSummary() {
  let what: string;
  if (mode === 'baseline') what = 'unmodified model';
  else if (lab.error()) what = 'interventions need fixing';
  else if (mode === 'modified') what = `strength ${strengthLabel(lab.strength())}`;
  else {
    let order = settings.compare;
    try {
      order = compareOrder(parseStrengths(settings.compare)).map(strengthLabel).join(', ');
    } catch {
      // shown as typed; sending opens Settings with the error
    }
    what = `strengths ${order}`;
  }
  const parts = [what, `temperature ${settings.temperature}`, `up to ${settings.maxNewTokens} tokens`];
  if (settings.thinking) parts.push('thinking on');
  ui.summary.textContent = parts.join(' · ');
}

function showCompareError(message: string | null) {
  ui.compareError.textContent = message ?? '';
  ui.compareError.hidden = !message;
  ui.compare.setAttribute('aria-invalid', String(!!message));
}

function fillSettings() {
  ui.temperature.value = String(settings.temperature);
  ui.maxTokens.value = String(settings.maxNewTokens);
  ui.thinking.checked = settings.thinking;
  ui.compare.value = settings.compare;
  syncOutputs();
}

function syncOutputs() {
  $('temperature-value').textContent = settings.temperature.toFixed(2);
  $('max-tokens-value').textContent = String(settings.maxNewTokens);
}

function readSettings() {
  settings = {
    temperature: Number(ui.temperature.value),
    maxNewTokens: Number(ui.maxTokens.value),
    thinking: ui.thinking.checked,
    compare: ui.compare.value,
  };
  try {
    parseStrengths(settings.compare);
    showCompareError(null);
  } catch (e) {
    showCompareError(errorText(e));
  }
  syncOutputs();
  saveSettings(settings);
  updateSummary();
}

function openSettings() {
  fillSettings();
  ui.settings.showModal();
}

for (const input of [ui.temperature, ui.maxTokens, ui.thinking, ui.compare]) {
  input.addEventListener('input', readSettings);
}
$('open-settings').addEventListener('click', openSettings);
$('reset-settings').addEventListener('click', () => {
  settings = { ...DEFAULTS };
  fillSettings();
  readSettings();
});
$('clear-cache').addEventListener('click', () => {
  if (confirm(`Delete the downloaded model (${gb(downloadSize)}) from this browser?`)) client.post({ type: 'clear-cache' });
});

for (const d of [ui.settings]) {
  d.addEventListener('click', (e) => {
    if (e.target === d) d.close(); // backdrop click
  });
}

// ---------------------------------------------------------------- start

applyView();
initLab();
lab.onChange(updateSummary);
lab.onInspect(showNeuron);
initPresets(() => showView('neurons'));
initNeuronDialog(() => modelReady);
initFinder({ canRun: () => modelReady && !busy, setBusy: (on) => setBusy(on), inspect: showNeuron });
fillSettings();
updateSummary();
ui.prompt.placeholder = PLACEHOLDERS[mode];
client.post({ type: 'probe' });
