/**
 * Internationalisation. Every visible string lives in /static/i18n/<lang>.json under a stable key.
 * Static markup uses data-i18n, data-i18n-title, data-i18n-aria-label; scripts call t(key, params).
 * Long prose (guide, about) uses data-lang-block="en|es" and is switched by CSS on <html lang>.
 */
import { storage } from './storage.js';

export const LANGUAGES = ['en', 'es'];
const STORAGE_KEY = 'terralens.lang';
const dictionaries = {};
let current = 'en';

export async function loadLocales() {
  await Promise.all(
    LANGUAGES.map(async (lang) => {
      const res = await fetch(`/static/i18n/${lang}.json`);
      if (!res.ok) throw new Error(`Missing locale file: ${lang}`);
      dictionaries[lang] = await res.json();
    }),
  );
}

export function initialLanguage() {
  const param = new URLSearchParams(location.search).get('lang');
  if (LANGUAGES.includes(param)) return param;
  const saved = storage.get(STORAGE_KEY);
  if (LANGUAGES.includes(saved)) return saved;
  return 'en';
}

export const getLang = () => current;

export function setLang(lang) {
  current = LANGUAGES.includes(lang) ? lang : 'en';
  document.documentElement.lang = current;
  storage.set(STORAGE_KEY, current);
  applyStaticTranslations();
}

/** Translated string for key, with {placeholders} filled from params. Falls back to English, then the key. */
export function t(key, params = {}) {
  const template = dictionaries[current]?.[key] ?? dictionaries.en?.[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

/** Picks the current language from a bilingual object such as {en: '...', es: '...'}. */
export function pick(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return value[current] ?? value.en ?? '';
}

const ATTRIBUTE_BINDINGS = [
  ['data-i18n-title', 'title'],
  ['data-i18n-aria-label', 'aria-label'],
];

export function applyStaticTranslations(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  ATTRIBUTE_BINDINGS.forEach(([dataAttr, attr]) => {
    root.querySelectorAll(`[${dataAttr}]`).forEach((el) => el.setAttribute(attr, t(el.getAttribute(dataAttr))));
  });
}
