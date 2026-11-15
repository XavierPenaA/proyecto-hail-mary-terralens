/**
 * localStorage wrapper for per-viewer conveniences (theme, language, expert mode).
 * Storage can be unavailable (private windows, blocked site data), so every call is guarded.
 */
export const storage = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore: the app works without persistence */
    }
  },
};
