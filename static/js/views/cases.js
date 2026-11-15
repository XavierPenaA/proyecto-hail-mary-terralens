/** Guided cases: the sidebar list and the step-by-step walkthrough panel. */
import { escapeHtml } from '../format.js';
import { pick, t } from '../i18n.js';
import { regionName, variableShort } from '../labels.js';

const $ = (id) => document.getElementById(id);

export function renderCaseList(cases, activeId, onSelect) {
  if (activeId) $('caseListGroup').open = true;
  $('caseList').replaceChildren(
    ...cases.map((c) => {
      const btn = document.createElement('button');
      btn.className = `story-btn${c.id === activeId ? ' active' : ''}`;
      btn.setAttribute('aria-pressed', String(c.id === activeId));
      btn.innerHTML = `
        <div class="story-btn-title"><span class="story-btn-number">${c.number}</span><span>${escapeHtml(pick(c.title))}</span></div>
        <div class="story-btn-meta">${escapeHtml(regionName(c.setup.region))} · ${escapeHtml(variableShort(c.setup.variable))}</div>`;
      btn.addEventListener('click', () => onSelect(c.id));
      return btn;
    }),
  );
}

export function renderCasePanel(caseObj, stepIndex) {
  const panel = $('casePanel');
  panel.hidden = !caseObj;
  $('caseDock').hidden = !caseObj;
  if (!caseObj) return;
  const step = caseObj.steps[stepIndex];
  $('caseKicker').textContent = t('case.kicker', { n: caseObj.number });
  $('caseTitle').textContent = pick(caseObj.title);
  $('caseQuestion').textContent = pick(caseObj.question);
  $('caseFinding').textContent = pick(caseObj.finding);
  $('caseFinding').hidden = stepIndex > 0; // the finding is read once; later steps stay compact
  $('caseStepText').textContent = pick(step.text);
  $('caseStepCounter').textContent = t('case.counter', { i: stepIndex + 1, n: caseObj.steps.length });
  $('btnCasePrev').disabled = stepIndex === 0;
  $('btnCaseNext').textContent = t(stepIndex === caseObj.steps.length - 1 ? 'case.finish' : 'case.next');
}
