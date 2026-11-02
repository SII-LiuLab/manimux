const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const hero = $('#hero-video');
const stage = $('#hero-stage');
const motionToggle = $('#motion-toggle');
const dialog = $('#film-dialog');
const film = $('#film-video');
let paused = reducedMotion.matches || Boolean(navigator.connection?.saveData);
let heroVisible = false;
let typingTimer;

function syncPlayback() {
  document.body.classList.toggle('motion-paused', paused);
  motionToggle.setAttribute('aria-pressed', String(paused));
  motionToggle.textContent = paused ? 'Play motion ▷' : 'Pause motion Ⅱ';
  if (paused || !heroVisible || document.hidden || dialog.open) { hero.pause(); return; }
  const source = $('source', hero);
  if (!source.src) { source.src = source.dataset.src; hero.load(); }
  hero.play().catch(() => { paused = true; syncPlayback(); });
}
motionToggle.addEventListener('click', () => { paused = !paused; syncPlayback(); });
new IntersectionObserver(([entry]) => { heroVisible = entry.isIntersecting; syncPlayback(); }, {threshold: .1}).observe(stage);
document.addEventListener('visibilitychange', () => { if (document.hidden) film.pause(); syncPlayback(); });
reducedMotion.addEventListener('change', () => { paused = reducedMotion.matches; syncPlayback(); });

const descriptions = {
  embodiment: {
    yam: 'Dual-arm integration with deployment recipes and RoboGUI visualization.',
    tianji: 'Tianji–TacCap deployment recipes with embodiment-specific observation and action adapters.',
    arx: 'Experimental ARX X5 SDK integration and station templates. Physical deployment remains unvalidated.',
    piper: 'Experimental PiPER SDK integration, plus offline visualization assets. Physical deployment remains unvalidated.'
  },
  policy: {
    pi05: 'Pi05 deployment recipes, with model serving separate from robot execution.',
    xr1: 'Xiaomi XR-1 recipes include embodiment-specific action conversion. Check the recipe for station requirements.',
    lingbot: 'LingBot-VLA2 recipes and sampling hooks. RTC support is currently withdrawn; inspect the support matrix.',
    dp: 'Diffusion Policy recipes with model-specific observation and action contracts.',
    openwam: 'OpenWAM recipes and sampling integrations. Specialized modes require compatible backend settings.',
    abc: 'ABC-DiT deployment recipes with model-specific chunk conditioning and action adaptation.'
  },
  inference: {
    async: 'Overlap inference and execution with asynchronous action chunking.',
    rtc: 'Condition compatible model sampling on a committed action prefix for chunk handoff continuity.',
    paint: 'Adjust sampling noise using the preceding prediction. Requires a compatible model-side hook.',
    aac: 'Adapt the execution horizon using candidate uncertainty and motion statistics.',
    autohorizon: 'Select an execution horizon using action attention from compatible model backends.',
    bid: 'Use backward-only candidate selection to encourage consistency between action chunks.'
  }
};
$$('.chips').forEach(group => $$('button', group).forEach((button, index) => {
  button.style.setProperty('--chip-index', index);
  button.addEventListener('click', () => {
    $$('button', group).forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    $(`#${button.dataset.axis}-detail`).textContent = descriptions[button.dataset.axis][button.dataset.value];
  });
}));
const axesObserver = new IntersectionObserver(([entry]) => {
  if (entry.isIntersecting) { $('#axes').classList.add('arrive'); axesObserver.disconnect(); }
}, {threshold:.15});
axesObserver.observe($('#axes'));

const steps = {
  deploy: {label:'AGENT-ASSISTED DEPLOYMENT', html:'<p class="panel-kicker">A FRESH SESSION. A CLEAR INTENTION.</p><p class="prompt-line"><span id="typed-prompt"></span><span class="cursor" aria-hidden="true"></span></p><div class="terminal-checks"><span>Resolve the local deployment recipe</span><span>Check station and model prerequisites</span><span>Open RoboGUI for the operator</span></div>'},
  run: {label:'ROBOGUI / OPERATOR IN CONTROL', html:'<img class="workflow-still" src="assets/execution-poster.jpg" alt="Archived RoboGUI interface with camera and robot views"><p class="panel-copy">Prepare the run, activate warmup when configured, then start execution. The operator controls the physical robot.</p>'},
  review: {label:'REVIEW / RECORDED EVIDENCE', html:'<p class="panel-kicker">THE RUN FINISHES. THE EVIDENCE REMAINS.</p><h3 class="panel-title">Look back at<br>what actually happened.</h3><div class="review-track" aria-hidden="true">' + Array.from({length:24}, (_,i) => `<i style="--bar-height:${12 + (i * 13 % 39)}px"></i>`).join('') + '</div><p class="panel-copy">Inspect recorded trajectories and camera observations in hardware-free replay. The graphic above illustrates a timeline; it is not measured telemetry.</p>'},
  score: {label:'EVALUATE / KEEP THE OUTCOME', html:'<p class="panel-kicker">TRY THE REVIEW INTERACTION</p><h3 class="panel-title">How did the task go?</h3><p class="panel-copy">Actual score options come from the task’s evaluation rules. This example stays in this browser session.</p><div class="score-options" role="group" aria-label="Illustrative task score"><button aria-pressed="false" data-score="Success">Success</button><button aria-pressed="false" data-score="Partial">Partial</button><button aria-pressed="false" data-score="Failure">Failure</button></div><button class="save-score" disabled>Save example</button><p class="save-status" role="status"></p>'}
};
let savedExample = null;
function selectStep(name, animate = true) {
  clearTimeout(typingTimer);
  $$('.step-list button').forEach(button => { const active = button.dataset.step === name; button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1; });
  $('#workflow-panel').setAttribute('aria-labelledby', `tab-${name}`);
  $('#panel-label').textContent = steps[name].label;
  $('#step-content').innerHTML = steps[name].html;
  if (name === 'deploy') {
    const message = 'Deploy my policy on this station\nand open RoboGUI.';
    const target = $('#typed-prompt');
    let position = 0;
    function type() { target.textContent = message.slice(0, ++position); if (position < message.length) typingTimer = setTimeout(type, 20); }
    if (reducedMotion.matches || paused || !animate) target.textContent = message;
    else type();
  }
  if (name === 'score') {
    let selected = savedExample?.score ?? null;
    const updateScore = () => {
      $$('[data-score]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.score === selected)));
      $('.save-score').disabled = !selected;
    };
    $$('[data-score]').forEach(button => button.addEventListener('click', () => { selected = button.dataset.score; updateScore(); $('.save-status').textContent = ''; }));
    $('.save-score').addEventListener('click', () => {
      savedExample = {score:selected, savedAt:new Date().toISOString(), illustrative:true};
      $('.save-status').textContent = `${selected} saved for this page session. No robot experiment was changed.`;
    });
    updateScore();
    if (savedExample) $('.save-status').textContent = `Saved example: ${savedExample.score}. Retained until this page is reloaded.`;
  }
}
const stepButtons = $$('.step-list button');
stepButtons.forEach((button,index) => {
  button.addEventListener('click', () => selectStep(button.dataset.step));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowDown') next = (index + 1) % stepButtons.length;
    if (event.key === 'ArrowUp') next = (index + stepButtons.length - 1) % stepButtons.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = stepButtons.length - 1;
    if (next !== undefined) { event.preventDefault(); stepButtons[next].focus(); selectStep(stepButtons[next].dataset.step); }
  });
});
selectStep('deploy', false);
const workflowObserver = new IntersectionObserver(([entry]) => {
  if (entry.isIntersecting) { if ($('#tab-deploy').getAttribute('aria-selected') === 'true') selectStep('deploy'); workflowObserver.disconnect(); }
}, {threshold:.4});
workflowObserver.observe($('#workflow-panel'));

const films = {
  product: {src:'assets/product-demo-hook.mp4',title:'ManiMux — The product film',caption:'92 seconds · 2560 × 1440 · 30 fps. Includes synthetic RoboGUI footage, repeated wall tiles, and an illustrative human-evaluation workflow. Not a recording of a complete physical experiment.'},
  archive: {src:'assets/execution.mp4',title:'RoboGUI — Archived real-robot demo',caption:'12-second excerpt at original speed from the repository’s real-robot demo. The archive does not include a verified model/checkpoint and inference manifest.'}
};
$$('[data-film]').forEach(button => button.addEventListener('click', () => {
  const selected = films[button.dataset.film];
  $('#film-title').textContent = selected.title;
  $('#film-caption').textContent = selected.caption;
  hero.pause();
  film.src = selected.src;
  dialog.showModal();
  document.body.classList.add('modal-open');
  film.play().catch(() => {});
}));
$('#close-film').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
dialog.addEventListener('close', () => { film.pause(); film.removeAttribute('src'); film.load(); document.body.classList.remove('modal-open'); syncPlayback(); });
syncPlayback();
