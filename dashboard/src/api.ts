const KEY = import.meta.env.VITE_AGENTMOAT_API_KEY as string | undefined;

export function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (KEY) headers.set("X-API-Key", KEY);
  return fetch(path, { ...init, headers });
}
