// The URL hash holds the page (view=neurons); a shared link also carries interventions (iv=…&s=…), which the editor
// applies once on arrival. Browsers never send the hash to the server.

export type View = 'chat' | 'neurons' | 'about';
const VIEWS: View[] = ['chat', 'neurons', 'about'];

export function hashParams(): Map<string, string> {
  const params = new Map<string, string>();
  for (const part of location.hash.slice(1).split('&')) {
    const i = part.indexOf('=');
    if (i > 0) params.set(part.slice(0, i), decodeURIComponent(part.slice(i + 1)));
  }
  return params;
}

/** Change some hash parameters (null removes one): a new history entry for page changes, otherwise in place. */
export function setHashParams(update: Record<string, string | null>, push = false): void {
  const params = hashParams();
  for (const [k, v] of Object.entries(update)) {
    if (v === null) params.delete(k);
    else params.set(k, v);
  }
  // readable links: keep ':' ',' '+' '!' as they are, escape only what would break the hash
  const encoded = [...params].map(([k, v]) => `${k}=${encodeURI(v).replace(/[#&]/g, encodeURIComponent)}`);
  const url = encoded.length ? `#${encoded.join('&')}` : location.pathname + location.search;
  if (url !== location.hash) history[push ? 'pushState' : 'replaceState'](null, '', url);
}

export const currentView = (): View => {
  const v = hashParams().get('view') as View | undefined;
  return v && VIEWS.includes(v) ? v : 'chat';
};
