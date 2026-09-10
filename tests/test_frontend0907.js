'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const { MiniDocument, Element } = require('./minidom');

const ROOT = path.resolve(__dirname, '..');
const DIR0907 = path.join(ROOT, 'static', 'newvis0907');
const PROJECT_DIR = '/tmp/proj_0907_demo';

const PASS = [];
const FAIL = [];

function check(name, cond, detail) {
  if (cond) {
    PASS.push(name);
    console.log('  ok   ' + name);
    return;
  }
  FAIL.push(name + (detail ? ' — ' + detail : ''));
  console.log('  FAIL ' + name + (detail ? '  (' + detail + ')' : ''));
}

function section(title) {
  console.log('\n=== ' + title + ' ===');
}

function read(file) {
  return fs.readFileSync(file, 'utf8');
}

function settle(rounds) {
  let p = Promise.resolve();
  for (let i = 0; i < (rounds || 120); i++) p = p.then(() => {});
  return p;
}

function modelReady() {
  return {
    model: {
      loaded: true,
      base_model: 'demo',
      adapter: 'demo_adapter',
      backend: 'local',
    },
  };
}

function projectPayload() {
  return {
    project_id: 'proj_0907_demo',
    project_dir: PROJECT_DIR,
    run_id: 'run_demo_0907',
    final_video_path: '/tmp/proj_0907_demo/final.mp4',
    dirty: { S02: ['render'] },
    quality: {
      status: 'needs_attention',
      score: 0.78,
      issues: [
        { code: 'subtitle_tight', message: '第二镜字幕略密', severity: 'warn', field: 'S02' },
        { code: 'cut_jump', message: '转场略跳', severity: 'warn', field: 'S02::subclip::2' },
      ],
    },
    hierarchy: {
      total_duration: 14.5,
      chapters: [
        {
          segment_id: 'seg_a',
          title: '开场<script>alert(1)</script>',
          summary: '提出问题',
          display_index: 1,
          model_shot_ids: ['S01'],
          start: 0,
          end: 6,
        },
        {
          segment_id: 'seg_b',
          title: '论证阶段',
          summary: '展开论证',
          display_index: 2,
          model_shot_ids: ['S02'],
          start: 6,
          end: 14.5,
        },
      ],
      model_shots: [
        {
          shot_id: 'S01',
          segment_id: 'seg_a',
          narration: '第一镜口播',
          display_number: '01',
          display_subtitle: '第一镜字幕<script>alert(2)</script>',
          start: 0,
          end: 6,
          duration: 6,
          is_spotlight: false,
          animation_type: 'pan_left',
          quality_status: 'ok',
          subclips: [
            {
              subclip_id: 'S01::subclip::1',
              display_number: '1.1',
              clip_id: 'clip_1_1',
              path: '/mnt/assets/shot1_a.png',
              source_duration: 4.5,
              source_in: 0,
              source_out: 4.5,
              final_duration: 3,
              playback_rate: 1.5,
              score: 0.98,
              query: '城市清晨',
              start: 0,
              end: 3,
              replaced: false,
              user_override: false,
              fingerprint: 'fp11',
            },
            {
              subclip_id: 'S01::subclip::2',
              display_number: '1.2',
              clip_id: 'clip_1_2',
              path: '/mnt/assets/shot1_b.png',
              source_duration: 5,
              source_in: 1,
              source_out: 4,
              final_duration: 3,
              playback_rate: 1,
              score: 0.92,
              query: '街道近景',
              start: 3,
              end: 6,
              replaced: false,
              user_override: false,
              fingerprint: 'fp12',
            },
          ],
        },
        {
          shot_id: 'S02',
          segment_id: 'seg_b',
          narration: '第二镜口播',
          display_number: '02',
          display_subtitle: '第二镜字幕<img src=x onerror=alert(3)>',
          start: 6,
          end: 14.5,
          duration: 8.5,
          is_spotlight: true,
          animation_type: 'zoom_in',
          quality_status: 'warn',
          subclips: [
            {
              subclip_id: 'S02::subclip::1',
              display_number: '2.1',
              clip_id: 'clip_2_1',
              path: '/mnt/assets/shot2_a.png',
              source_duration: 8,
              source_in: 0,
              source_out: 5,
              final_duration: 4,
              playback_rate: 1.25,
              score: 0.89,
              query: '实验室近景<script>alert(4)</script>',
              start: 0,
              end: 4,
              replaced: false,
              user_override: false,
              fingerprint: 'fp21',
            },
            {
              subclip_id: 'S02::subclip::2',
              display_number: '2.2',
              clip_id: 'clip_2_2',
              path: '/mnt/assets/shot2_b.png',
              source_duration: 6.5,
              source_in: 1,
              source_out: 5.5,
              final_duration: 4.5,
              playback_rate: 1,
              score: 0.74,
              query: '显微镜镜头',
              start: 4,
              end: 8.5,
              replaced: true,
              user_override: true,
              fingerprint: 'fp22',
            },
          ],
        },
      ],
    },
    progress: {
      total_duration: 14.5,
      overall_pct: 100,
      track_order: ['shots', 'render'],
      labels: { shots: '镜头', render: '渲染' },
      clips: {
        shots: [
          { shot_id: 'S01', title: '第一镜', start: 0, end: 6, duration: 6 },
          { shot_id: 'S02', title: '第二镜', start: 6, end: 14.5, duration: 8.5 },
        ],
        render: [
          { shot_id: 'S01', start: 0, end: 6, duration: 6, ready: true },
          { shot_id: 'S02', start: 6, end: 14.5, duration: 8.5, ready: false },
        ],
      },
      stats: {
        shots: { done: 2, total: 2, pct: 100 },
        render: { done: 1, total: 2, pct: 50 },
      },
    },
    active_transactions: [],
  };
}

function txView(state, extra) {
  return Object.assign({
    transaction_id: `tx_${state}`,
    state: state,
    ops: [{ op: 'replace_subclip', subclip_id: 'S02::subclip::1', shot_id: 'S02', query: '实验室夜景' }],
    needs_confirm: [],
    reply: '模型原话：已经修改成功。',
    applied_results: [],
    verification: null,
    render_result: null,
    error: state.includes('failed') ? '后处理失败' : '',
    created_at: '2026-09-07T12:00:00',
    updated_at: '2026-09-07T12:00:10',
    history: [],
  }, extra || {});
}

function installDomPolyfills(document) {
  if (typeof Element.prototype.append !== 'function') {
    Element.prototype.append = function (...nodes) {
      nodes.forEach((node) => {
        if (node === null || node === undefined) return;
        if (typeof node === 'string') this.appendChild(this.ownerDocument.createTextNode(node));
        else this.appendChild(node);
      });
    };
  }
  if (typeof Element.prototype.prepend !== 'function') {
    Element.prototype.prepend = function (...nodes) {
      const first = this.childNodes[0] || null;
      nodes.slice().reverse().forEach((node) => {
        let next = node;
        if (typeof next === 'string') next = this.ownerDocument.createTextNode(next);
        if (first) this.insertBefore(next, first);
        else this.appendChild(next);
      });
    };
  }
  if (typeof Element.prototype.insertBefore !== 'function') {
    Element.prototype.insertBefore = function (node, before) {
      if (!before) return this.appendChild(node);
      if (node.parentNode) node.parentNode.removeChild(node);
      const idx = this.childNodes.indexOf(before);
      if (idx === -1) return this.appendChild(node);
      node.parentNode = this;
      this.childNodes.splice(idx, 0, node);
      return node;
    };
  }
  if (typeof Element.prototype.after !== 'function') {
    Element.prototype.after = function (...nodes) {
      if (!this.parentNode) return;
      let idx = this.parentNode.childNodes.indexOf(this) + 1;
      nodes.forEach((node) => {
        let next = node;
        if (typeof next === 'string') next = this.ownerDocument.createTextNode(next);
        if (next.parentNode) next.parentNode.removeChild(next);
        next.parentNode = this.parentNode;
        this.parentNode.childNodes.splice(idx, 0, next);
        idx += 1;
      });
    };
  }
  if (typeof Element.prototype.replaceWith !== 'function') {
    Element.prototype.replaceWith = function (node) {
      if (!this.parentNode) return;
      const parent = this.parentNode;
      const idx = parent.childNodes.indexOf(this);
      if (idx === -1) return;
      let next = node;
      if (typeof next === 'string') next = parent.ownerDocument.createTextNode(next);
      if (next.parentNode) next.parentNode.removeChild(next);
      next.parentNode = parent;
      parent.childNodes[idx] = next;
      this.parentNode = null;
    };
  }
  if (typeof Element.prototype.setAttributeNS !== 'function') {
    Element.prototype.setAttributeNS = function (_ns, name, value) {
      this.setAttribute(name, value);
    };
  }
  if (typeof Element.prototype.contains !== 'function') {
    Element.prototype.contains = function (target) {
      if (!target) return false;
      let cur = target;
      while (cur) {
        if (cur === this) return true;
        cur = cur.parentNode;
      }
      return false;
    };
  }
  if (typeof Element.prototype.focus !== 'function') {
    Element.prototype.focus = function () {};
  }
  if (typeof Element.prototype.blur !== 'function') {
    Element.prototype.blur = function () {};
  }
  if (document.body && document.body.classList && typeof document.body.classList.replace !== 'function') {
    const proto = Object.getPrototypeOf(document.body.classList);
    proto.replace = function (oldName, newName) {
      if (this.contains(oldName)) this.remove(oldName);
      this.add(newName);
      return true;
    };
  }
}

function bootstrap(options) {
  options = options || {};
  const html = read(path.join(DIR0907, 'index.html'));
  const document = new MiniDocument(html);
  installDomPolyfills(document);

  const fetchLog = [];
  const promptLog = [];
  const promptReplies = (options.promptReplies || []).slice();
  const localStore = Object.assign({}, options.localStore || {});
  const payload = options.projectPayload || projectPayload();

  const responses = Object.assign({
    '/api/config': {
      tracks: [
        { key: 'shots', label: '镜头' },
        { key: 'render', label: '渲染' },
      ],
      features: { editor: true, chat_edit: true, auth: false },
      editor: { op_labels: { replace_subclip: '替换子镜', set_subclip_timing: '调整时长' } },
      user: null,
    },
    '/api/status': modelReady(),
    '/api/runs': { runs: [] },
    '/api/0907/projects': { projects: [{ project_dir: PROJECT_DIR, project_id: 'proj_0907_demo' }] },
    '/api/0907/project': payload,
    '/api/0907/project/chat/history': { messages: [] },
  }, options.responses || {});

  const player = document.getElementById('player');
  player.load = () => {};
  player.play = () => Promise.resolve();
  player.pause = () => {};
  player.paused = true;
  player.ended = false;

  const fetchImpl = async (url, init) => {
    const raw = String(url);
    const method = (init && init.method) || 'GET';
    const body = init && init.body ? JSON.parse(init.body) : null;
    fetchLog.push({ url: raw, method: method, body: body });

    const override = options.onFetch && options.onFetch(raw, method, body, responses);
    if (override) {
      return {
        ok: (override.status || 200) < 400,
        status: override.status || 200,
        statusText: '',
        text: async () => JSON.stringify(override.json === undefined ? {} : override.json),
      };
    }

    const key = raw.split('?')[0].replace(/^https?:\/\/[^/]+/, '');
    const value = Object.prototype.hasOwnProperty.call(responses, key) ? responses[key] : {};
    return {
      ok: true,
      status: 200,
      statusText: '',
      text: async () => JSON.stringify(value),
    };
  };

  function FakeTimeline(opts) {
    this.order = [];
    this.selectedShots = [];
    this.selectedClips = [];
    this.lastUpdate = null;
    this.onSelect = opts.onSelect;
    this.setTracks = (tracks) => { this.order = tracks.map((t) => t.key); };
    this.update = (progress) => { this.lastUpdate = progress; };
    this.fit = () => {};
    this.setDirty = () => {};
    this.select = (shotId) => { this.selectedShots.push(shotId); };
    this.selectClip = (track, clip) => { this.selectedClips.push({ track, clip }); };
  }

  function FakeRail() {
    this.el = document.getElementById('trackRail');
    this.built = '';
    this.update = () => {};
  }

  class FakeWebSocket {
    constructor(url) { this.url = url; this.readyState = 1; }
    send() {}
    close() { this.readyState = 3; if (this.onclose) this.onclose(); }
  }
  FakeWebSocket.OPEN = 1;

  const sandbox = {
    console: console,
    document: document,
    fetch: fetchImpl,
    __fetchImpl: fetchImpl,
    WebSocket: FakeWebSocket,
    URL: URL,
    JSON: JSON,
    Math: Math,
    Date: Date,
    Number: Number,
    String: String,
    Object: Object,
    Array: Array,
    Set: Set,
    Map: Map,
    Promise: Promise,
    Error: Error,
    requestAnimationFrame: (fn) => { fn(); return 0; },
    setTimeout: (fn) => { fn(); return 1; },
    clearTimeout: () => {},
    setInterval: () => 0,
    clearInterval: () => {},
    alert: () => {},
    confirm: () => true,
    prompt: (label, value) => {
      promptLog.push({ label: label, value: value });
      return promptReplies.length ? promptReplies.shift() : null;
    },
    encodeURIComponent: encodeURIComponent,
    localStorage: {
      getItem: (key) => (Object.prototype.hasOwnProperty.call(localStore, key) ? localStore[key] : null),
      setItem: (key, value) => { localStore[key] = String(value); },
      removeItem: (key) => { delete localStore[key]; },
    },
    location: { origin: 'http://localhost:8097', host: 'localhost:8097', protocol: 'http:', pathname: '/0907', search: '' },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.window.addEventListener = () => {};
  sandbox.window.AICutTimeline = FakeTimeline;
  sandbox.window.AICutRail = FakeRail;

  vm.createContext(sandbox);
  vm.runInContext('globalThis.fetch = (...args) => globalThis.__fetchImpl(...args);', sandbox);
  [
    path.join(DIR0907, 'shot_navigator.js'),
    path.join(DIR0907, 'newvis.js'),
  ].forEach((file) => vm.runInContext(read(file), sandbox, { filename: path.basename(file) }));

  return { document, sandbox, fetchLog, promptLog, responses };
}

async function bootReady(options) {
  const env = bootstrap(options);
  env.document.dispatch('DOMContentLoaded');
  await settle();
  return env;
}

function testFilesExist() {
  section('files');
  ['index.html', 'newvis.css', 'newvis.js', 'navigator.css', 'shot_navigator.js']
    .forEach((name) => check(`${name} exists`, fs.existsSync(path.join(DIR0907, name))));
}

function testSourceIsolationContract() {
  section('source isolation contract');
  const js = read(path.join(DIR0907, 'newvis.js'));
  const css = read(path.join(DIR0907, 'newvis.css'));
  const nav = read(path.join(DIR0907, 'shot_navigator.js'));
  check('dedicated 0907 storage keys remain',
    js.includes('aicut.newvis0907.inputMode') && js.includes('aicut.newvis0907.lastSession'));
  check('0907 keeps dedicated project api prefix',
    js.includes('/api/0907/project') && js.includes('/api/0907/projects'));
  check('0907 launch uses dedicated /api/0907/runs endpoint',
    js.includes("api.post('/api/0907/runs'"), 'newvis0907 must launch via /api/0907/runs');
  check('legacy 0901 frontend still launches via /api/runs',
    (function () {
      const legacy = path.join(ROOT, 'static', 'newvis', 'newvis.js');
      if (!fs.existsSync(legacy)) return true; // no legacy file => nothing to protect
      const src = read(legacy);
      return src.includes("'/api/runs'") && !src.includes('/api/0907/runs');
    })(),
    'legacy newvis must keep posting /api/runs');
  check('0907 stylesheet stays isolated',
    !css.includes('/static/newvis/newvis.css') && !css.includes('@import'));
  check('shot navigator contract methods are present in source',
    nav.includes('class AICutShotNavigator')
      && nav.includes('constructor(')
      && nav.includes('update(')
      && nav.includes('selectShot(')
      && nav.includes('selectSubclip(')
      && nav.includes('syncTime(')
      && nav.includes('clear('));
}

function testMarkupContract() {
  section('markup contract');
  const html = read(path.join(DIR0907, 'index.html'));
  const document = new MiniDocument(html);
  const ids = new Set();
  [path.join(DIR0907, 'newvis.js'), path.join(DIR0907, 'shot_navigator.js')].forEach((file) => {
    const code = read(file);
    const re = /getElementById\('([^']+)'\)/g;
    let m;
    while ((m = re.exec(code)) !== null) ids.add(m[1]);
  });
  const missing = [...ids].filter((id) => !document.getElementById(id));
  check('all referenced ids exist in index', missing.length === 0, missing.join(', '));

  const stage = document.querySelector('.stage');
  const stageIds = stage.children.map((node) => node.id || node.className || node.tagName.toLowerCase());
  check('index includes navigator mount',
    html.includes('shotNavigatorMount') || html.includes('镜头导航'),
    'missing shotNavigatorMount');
  check('navigator mount sits between viewer and progress fold',
    stageIds.join('|').includes('viewer|foldNavigator|foldRail'),
    stageIds.join(' -> '));
  const foldNavigator = document.getElementById('foldNavigator');
  check('0907 shot navigator is hidden from the page',
    !!foldNavigator && foldNavigator.hidden === true,
    foldNavigator && String(foldNavigator.hidden));
}

function testLoginBranding() {
  section('login branding');
  const html = read(path.join(ROOT, 'static', 'newvis', 'login.html'));
  check('login title uses WevisionCut',
    html.includes('<title>WevisionCut · 登录</title>'),
    html.match(/<title>[^<]+<\/title>/)?.[0]);
  check('login brand text uses WevisionCut',
    html.includes('<b>WevisionCut</b>') && !html.includes('<b>WEDirector</b>'),
    html);
  check('login logo uses public WevisionCut mark',
    html.includes('logo_aicut.png') && html.includes('/static/newvis0907/assets/logo_aicut.png') && !html.includes('/api/media?path='),
    html);
}

function testScriptSyntaxLoads() {
  section('script parse');
  [path.join(DIR0907, 'newvis.js'), path.join(DIR0907, 'shot_navigator.js')].forEach((file) => {
    try {
      new vm.Script(read(file), { filename: path.basename(file) });
      check(path.basename(file) + ' parses', true);
    } catch (err) {
      check(path.basename(file) + ' parses', false, err.message);
    }
  });
}

async function testBootAndNavigatorRender() {
  section('boot + navigator render');
  const env = await bootReady({
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  check('app instance created', !!sandbox.window.aicut, typeof sandbox.window.aicut);
  check('boot requests /api/config', env.fetchLog.some((f) => f.url.includes('/api/config')));
  check('restores last project through 0907 endpoint',
    env.fetchLog.some((f) => f.url.includes('/api/0907/project?dir=')),
    JSON.stringify(env.fetchLog.map((f) => f.url)));

  const mount = document.getElementById('shotNavigatorMount');
  check('navigator mount exists in DOM', !!mount);
  check('navigator mounted visibly after project load', !!mount && mount.hidden === false);
  check('navigator summary uses accessible title',
    !!mount && mount.getAttribute('aria-label') === '镜头导航',
    mount && mount.getAttribute('aria-label'));

  const tabs = mount ? mount.querySelectorAll('[data-nav-tab]') : [];
  check('renders chapter tabs', tabs.length === 2, String(tabs.length));
  check('renders model shots with 01/02 numbers',
    !!mount && mount.textContent.includes('01') && mount.textContent.includes('02'),
    mount && mount.textContent);
  const previews = mount ? mount.querySelectorAll('.shot-card-preview') : [];
  const materials = mount ? mount.querySelectorAll('[data-shot-material]') : [];
  check('every shot card has unified preview area',
    !!mount && previews.length >= 2,
    String(previews.length));
  check('shot cards always show material name',
    !!mount && materials.length >= 2 && materials[0].textContent.trim().length > 0,
    materials[0] && materials[0].textContent);
  check('navigator does not render quality score or auto-fix',
    !!mount && !mount.textContent.includes('一键修复') && !mount.textContent.includes('质量 fail'),
    mount && mount.textContent);

  const detail = mount && mount.querySelector('[data-selected-shot-detail]');
  check('selected shot detail is rendered',
    !!detail && detail.textContent.includes('口播') && detail.textContent.includes('animation'),
    detail && detail.textContent);

  check('unsafe chapter text rendered as text, not html nodes',
    !!mount && mount.textContent.includes('开场<script>alert(1)</script>'),
    mount && mount.textContent);
  check('query text rendered literally instead of image node injection',
    !!detail && detail.textContent.includes('<script>alert(4)</script>'),
    detail && detail.textContent);

  const covers = mount ? mount.querySelectorAll('.shot-card-preview') : [];
  const withMedia = [].filter.call(covers, (node) =>
    node.querySelectorAll('.shot-card-preview-img').length > 0);
  check('every shot tile renders an actual frame',
    covers.length > 0 && withMedia.length === covers.length,
    withMedia.length + '/' + covers.length);

  const thumb = mount && mount.querySelector('.subclip-thumb');
  check('subclip thumbnail uses media endpoint',
    !!thumb && String(thumb.getAttribute('src') || '').includes('/api/media?path='),
    thumb && thumb.getAttribute('src'));

  return env;
}

async function testSelectionAndSync() {
  section('selection + player sync');
  const env = await bootReady({
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const mount = document.getElementById('shotNavigatorMount');
  const player = document.getElementById('player');
  if (!mount) {
    check('renders shot cards', false, 'missing shotNavigatorMount');
    check('clicking shot selects navigator state', false, 'missing shotNavigatorMount');
    check('clicking shot seeks player to absolute shot start', false, 'missing shotNavigatorMount');
    check('clicking shot asks timeline to highlight shot', false, 'missing shotNavigatorMount');
    check('only selected model shot expands subclips', false, 'missing shotNavigatorMount');
    check('selected shot exposes hierarchical subclip numbers', false, 'missing shotNavigatorMount');
    check('clicking subclip seeks to absolute subclip start', false, 'missing shotNavigatorMount');
    check('timeline onClipSelected drives navigator.selectShot', false, 'missing shotNavigatorMount');
    check('player timeupdate syncs active shot', false, 'missing shotNavigatorMount');
    check('player timeupdate syncs active subclip', false, 'missing shotNavigatorMount');
    return;
  }

  const shotCards = mount.querySelectorAll('[data-shot-id] .shot-card-main');
  check('renders shot cards', shotCards.length >= 2, String(shotCards.length));
  shotCards[1].click();
  await settle();
  check('clicking shot selects navigator state', app.navigator && app.navigator.selectedShotId === 'S02',
    app.navigator && app.navigator.selectedShotId);
  check('clicking shot seeks player to absolute shot start',
    Math.abs(player.currentTime - 6) < 1e-6, String(player.currentTime));
  check('clicking shot asks timeline to highlight shot',
    app.timeline.selectedShots.includes('S02'),
    JSON.stringify(app.timeline.selectedShots));

  const expanded = mount.querySelectorAll('[data-subclip-list-open="true"]');
  check('only selected model shot expands subclips', expanded.length === 1, String(expanded.length));
  check('subclips expand alongside the strip, not stacked inside the card',
    !!expanded[0] && expanded[0].parentNode
      && !String(expanded[0].parentNode.className || '').includes('shot-card'),
    expanded[0] && expanded[0].parentNode && expanded[0].parentNode.className);
  check('selected shot exposes hierarchical subclip numbers',
    expanded[0] && expanded[0].textContent.includes('2.1') && expanded[0].textContent.includes('2.2'),
    expanded[0] && expanded[0].textContent);

  const subclip = mount.querySelector('[data-subclip-id="S02::subclip::2"] .subclip-main');
  subclip.click();
  await settle();
  check('clicking subclip seeks to absolute subclip start',
    Math.abs(player.currentTime - 10) < 1e-6, String(player.currentTime));

  app.onClipSelected('shots', { shot_id: 'S01', start: 0, end: 6 });
  await settle();
  check('timeline onClipSelected drives navigator.selectShot',
    app.navigator && app.navigator.selectedShotId === 'S01',
    app.navigator && app.navigator.selectedShotId);

  player.currentTime = 11.2;
  player.dispatch('timeupdate');
  await settle();
  check('player timeupdate syncs active shot',
    app.navigator && app.navigator.selectedShotId === 'S02',
    app.navigator && app.navigator.selectedShotId);
  check('player timeupdate syncs active subclip',
    app.navigator && app.navigator.selectedSubclipId === 'S02::subclip::2',
    app.navigator && app.navigator.selectedSubclipId);
}

async function testTransactionsUxAndActions() {
  section('transactions ux');
  const payload = projectPayload();
  payload.active_transactions = [
    txView('located'),
    txView('planned'),
    txView('confirmation_required', {
      transaction_id: 'tx_confirm',
      needs_confirm: ['replace_subclip'],
      ops: [{ op: 'replace_subclip', subclip_id: 'S02::subclip::1', shot_id: 'S02', query: '夜景城市' }],
    }),
    txView('persisted', { transaction_id: 'tx_persisted' }),
    txView('rendering', { transaction_id: 'tx_rendering' }),
    txView('verifying', { transaction_id: 'tx_verifying' }),
    txView('succeeded', { transaction_id: 'tx_succeeded' }),
    txView('render_failed', { transaction_id: 'tx_retryable' }),
    txView('verification_failed', { transaction_id: 'tx_retryable_2' }),
    txView('rolled_back', { transaction_id: 'tx_rolled' }),
  ];

  const env = await bootReady({
    projectPayload: payload,
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
    onFetch: (url, method) => {
      if (url.includes('/confirm') && method === 'POST') {
        const next = projectPayload();
        next.active_transactions = [txView('persisted', { transaction_id: 'tx_confirm' })];
        return { json: { transaction: txView('persisted', { transaction_id: 'tx_confirm' }), project: next } };
      }
      if (url.includes('/retry') && method === 'POST') {
        const next = projectPayload();
        next.active_transactions = [txView('rendering', { transaction_id: 'tx_retryable' })];
        return { json: { transaction: txView('rendering', { transaction_id: 'tx_retryable' }), project: next } };
      }
      if (url.includes('/rollback') && method === 'POST') {
        const next = projectPayload();
        next.active_transactions = [txView('rolled_back', { transaction_id: 'tx_retryable_2' })];
        return { json: { transaction: txView('rolled_back', { transaction_id: 'tx_retryable_2' }), project: next } };
      }
      return null;
    },
  });
  const { document, sandbox } = env;
  const log = document.getElementById('stream');
  const app = sandbox.window.aicut;

  const expectedLabels = ['已定位', '已规划', '等待确认', '已保存', '渲染中', '校验中', '修改成功', '渲染失败', '校验失败', '已回滚'];
  expectedLabels.forEach((label) => {
    check(`transaction status label ${label}`, log.textContent.includes(label), log.textContent);
  });
  const persistedCard = log.querySelector('.state-persisted .tx-card-body');
  check('non-succeeded states do not surface generic success copy',
    !!persistedCard && !persistedCard.textContent.includes('修改成功'),
    persistedCard && persistedCard.textContent);

  const target = log.querySelector('[data-transaction-target="S02"]');
  check('transaction target card exists', !!target);
  if (!target) {
    check('clicking transaction target selects shot', false, 'missing transaction target');
    check('confirmation_required exposes confirm button', false, 'missing transaction ui');
    check('failed transaction exposes retry button', false, 'missing transaction ui');
    check('failed transaction exposes rollback button', false, 'missing transaction ui');
    check('confirm posts to 0907 confirm endpoint', false, 'missing transaction ui');
    check('retry posts to 0907 retry endpoint', false, 'missing transaction ui');
    check('rollback posts to 0907 rollback endpoint', false, 'missing transaction ui');
    return;
  }
  target.click();
  await settle();
  check('clicking transaction target selects shot',
    app.navigator && app.navigator.selectedShotId === 'S02',
    app.navigator && app.navigator.selectedShotId);

  const confirmBtn = log.querySelector('[data-transaction-action="confirm"]');
  const retryBtn = log.querySelector('[data-transaction-action="retry"]');
  const rollbackBtn = log.querySelector('[data-transaction-action="rollback"]');
  check('confirmation_required exposes confirm button', !!confirmBtn);
  check('failed transaction exposes retry button', !!retryBtn);
  check('failed transaction exposes rollback button', !!rollbackBtn);

  confirmBtn.click();
  await settle();
  retryBtn.click();
  await settle();
  rollbackBtn.click();
  await settle();

  const postUrls = env.fetchLog.filter((f) => f.method === 'POST').map((f) => f.url);
  check('confirm posts to 0907 confirm endpoint',
    postUrls.some((u) => u.includes('/api/0907/project/transactions/tx_confirm/confirm')),
    postUrls.join(', '));
  check('retry posts to 0907 retry endpoint',
    postUrls.some((u) => u.includes('/api/0907/project/transactions/tx_retryable/retry')),
    postUrls.join(', '));
  check('rollback posts to 0907 rollback endpoint',
    postUrls.some((u) => u.includes('/api/0907/project/transactions/') && u.includes('/rollback')),
    postUrls.join(', '));
}

async function testSubclipOpsAndClear() {
  section('subclip ops + clear');
  const sentOps = [];
  const env = await bootReady({
    promptReplies: ['夜景实验室', '5.5'],
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
    onFetch: (url, method, body) => {
      if (url.includes('/api/0907/project/ops') && method === 'POST') {
        sentOps.push(body);
        return { json: { applied: true, transaction: txView('persisted'), project: projectPayload() } };
      }
      return null;
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const mount = document.getElementById('shotNavigatorMount');
  if (!mount) {
    check('replace_subclip op posted', false, 'missing shotNavigatorMount');
    check('replace_subclip carries project_dir and idempotency_key', false, 'missing shotNavigatorMount');
    check('replace_subclip carries prompted query', false, 'missing shotNavigatorMount');
    check('duration op uses set_subclip_timing', false, 'missing shotNavigatorMount');
    check('duration op carries numeric duration_sec', false, 'missing shotNavigatorMount');
    check('restore op uses mode auto', false, 'missing shotNavigatorMount');
    check('clear hides navigator on exit', false, 'missing shotNavigatorMount');
    return;
  }

  const shotCard = mount.querySelector('[data-shot-id="S02"]');
  shotCard.click();
  await settle();

  mount.querySelector('[data-subclip-op="replace"][data-subclip-id="S02::subclip::1"]').click();
  await settle();
  mount.querySelector('[data-subclip-op="duration"][data-subclip-id="S02::subclip::1"]').click();
  await settle();
  mount.querySelector('[data-subclip-op="restore"][data-subclip-id="S02::subclip::1"]').click();
  await settle();

  check('replace_subclip op posted', sentOps[0] && sentOps[0].ops[0].op === 'replace_subclip',
    JSON.stringify(sentOps[0]));
  check('replace_subclip carries project_dir and idempotency_key',
    sentOps[0] && sentOps[0].project_dir === PROJECT_DIR && !!sentOps[0].idempotency_key,
    JSON.stringify(sentOps[0]));
  check('replace_subclip carries prompted query',
    sentOps[0] && sentOps[0].ops[0].query === '夜景实验室',
    JSON.stringify(sentOps[0] && sentOps[0].ops[0]));

  check('duration op uses set_subclip_timing',
    sentOps[1] && sentOps[1].ops[0].op === 'set_subclip_timing',
    JSON.stringify(sentOps[1]));
  check('duration op carries numeric duration_sec',
    sentOps[1] && sentOps[1].ops[0].duration_sec === 5.5,
    JSON.stringify(sentOps[1] && sentOps[1].ops[0]));

  check('restore op uses mode auto',
    sentOps[2] && sentOps[2].ops[0].op === 'set_subclip_timing' && sentOps[2].ops[0].mode === 'auto',
    JSON.stringify(sentOps[2]));

  app.newSession();
  await settle();
  check('clear hides navigator on exit', mount.hidden === true, String(mount.hidden));
}

function qualityPayload() {
  const payload = projectPayload();
  payload.quality = {
    status: 'fail',
    score: 62,
    summary: { fail: 1, warning: 1, info: 0 },
    issues: [
      {
        code: 'subtitle_line_overflow',
        severity: 'fail',
        message: '第二镜字幕超过两行',
        shot_id: 'S02',
        field: 'S02.subtitle',
        auto_fixable: true,
        user_override: false,
      },
      {
        code: 'footage_playback_rate_low',
        severity: 'warning',
        message: '第一镜播放速率过低',
        shot_id: 'S01',
        subclip_id: 'S01::subclip::1',
        field: 'S01::subclip::1.playback_rate',
        auto_fixable: false,
        user_override: false,
      },
    ],
  };
  return payload;
}

function threeShotPlaylistPayload() {
  const payload = JSON.parse(JSON.stringify(projectPayload()));
  payload.final_video_path = '';
  payload.stitch_pending = true;
  payload.playlist = [
    { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6, start: 0, end: 6 },
    { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5, start: 6, end: 14.5 },
    { shot_id: 'S03', path: '/tmp/s03.mp4', duration_sec: 5, start: 14.5, end: 19.5 },
  ];
  payload.preview_bgm = { volume: 0.35, tracks: [{ path: '/mnt/music/bgm.mp3' }] };
  payload.hierarchy.chapters[1].model_shot_ids.push('S03');
  payload.hierarchy.model_shots.push({
    shot_id: 'S03',
    segment_id: 'seg_b',
    narration: '第三镜口播',
    display_number: '03',
    display_subtitle: '第三镜字幕',
    start: 14.5,
    end: 19.5,
    duration: 5,
    is_spotlight: false,
    animation_type: 'none',
    quality_status: 'ok',
    subclips: [
      {
        subclip_id: 'S03::subclip::1',
        display_number: '3.1',
        clip_id: 'clip_3_1',
        path: '/mnt/assets/shot3.mp4',
        source_duration: 5,
        source_in: 0,
        source_out: 5,
        final_duration: 5,
        playback_rate: 1,
        score: 0.91,
        query: '第三镜素材',
        start: 0,
        end: 5,
      },
    ],
  });
  return payload;
}

async function testQualityPanel() {
  section('quality panel removed');
  const applied = [];
  const env = await bootReady({
    projectPayload: qualityPayload(),
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
    onFetch: (url, method, body) => {
      if (url.includes('/api/0907/project/quality-check') && method === 'POST') {
        applied.push(body);
        const next = qualityPayload();
        next.quality = { status: 'pass', score: 100, summary: { fail: 0, warning: 0, info: 0 }, issues: [] };
        return { json: { status: 'pass', score: 100, transaction: txView('succeeded'), project: next, proposals: [{ op: 'rebuild_subtitles', shot_id: 'S02' }] } };
      }
      return null;
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const mount = document.getElementById('shotNavigatorMount');
  check('quality panel is hidden from navigator',
    !!mount && !mount.querySelector('[data-quality-fix]') && !mount.querySelector('[data-quality-issue]'),
    mount && mount.textContent);
  check('no quality-check request from plain navigator render',
    applied.length === 0,
    JSON.stringify(applied));
}

async function testPreviewBgmAndTransitions() {
  section('preview bgm + transitions');
  const env = await bootReady({
    projectPayload: Object.assign(projectPayload(), {
      stitch_pending: true,
      preview_bgm: { volume: 0.35, tracks: [{ path: '/mnt/music/bgm.mp3' }] },
      playlist: [
        { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6, start: 0, end: 6 },
        { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5, start: 6, end: 14.5 },
      ],
    }),
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const player = document.getElementById('player');
  const app = sandbox.window.aicut;
  check('playlist preview attaches bgm audio before export',
    !!app._bgmAudio && String(app._bgmAudio.getAttribute('src') || '').includes('/api/media?path='),
    app._bgmAudio && app._bgmAudio.getAttribute('src'));
  if (app._bgmAudio) app._bgmAudio.duration = 20;
  app._playIndex = 1;
  player.currentTime = 1.7;
  player.dispatch('seeked');
  await settle();
  check('playlist seek keeps bgm aligned with current absolute time',
    !!app._bgmAudio && Math.abs((Number(app._bgmAudio.currentTime) || 0) - 7.7) < 1e-6,
    app._bgmAudio && String(app._bgmAudio.currentTime));
  app._playIndex = 0;
  app._onNavigatorSeek({ shot: { shot_id: 'S02' }, seconds: 6 });
  await settle();
  check('shot switching applies natural transition class',
    player.classList.contains('shot-transitioning'),
    player.className);
}

async function testPreviewBgmFallbackAndReadyRetry() {
  section('preview bgm fallback + ready retry');
  const payload = Object.assign(projectPayload(), {
    stitch_pending: true,
    final_video_path: '',
    bgm_tracks: [{ audio_path: '/mnt/music/from_audio_path.mp3' }],
    playlist: [
      { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6, start: 0, end: 6 },
      { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5, start: 6, end: 14.5 },
    ],
  });
  delete payload.preview_bgm;
  const env = await bootReady({
    projectPayload: payload,
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const player = document.getElementById('player');
  check('bgm fallback accepts audio_path tracks',
    !!app._bgmAudio && String(app._bgmAudio.getAttribute('src') || '').includes('from_audio_path.mp3'),
    app._bgmAudio && app._bgmAudio.getAttribute('src'));

  let plays = 0;
  if (app._bgmAudio) {
    app._bgmAudio.play = () => { plays += 1; return Promise.resolve(); };
    player.paused = false;
    app._bgmAudio.dispatch('canplay');
  }
  await settle();
  check('bgm retries playback when audio becomes ready',
    plays > 0,
    String(plays));
}

async function testThumbnailFallsBackToRenderedShot() {
  section('thumbnail fallback');
  const payload = JSON.parse(JSON.stringify(projectPayload()));
  payload.hierarchy.model_shots[1].subclips.forEach((clip) => { clip.path = ''; });
  payload.playlist = [
    { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6 },
    { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5 },
  ];
  const env = await bootReady({
    projectPayload: payload,
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const mount = env.document.getElementById('shotNavigatorMount');
  const card = mount && mount.querySelector('[data-shot-id="S02"] .shot-card-preview');
  const frame = card && card.querySelector('.shot-card-preview-img');
  check('shot without subclip media still renders a frame', !!frame,
    card && card.className);
  check('fallback frame points at the rendered shot clip',
    !!frame && String(frame.getAttribute('src') || '').includes(encodeURIComponent('/tmp/s02.mp4')),
    frame && frame.getAttribute('src'));
}

async function testSubclipThumbnailFallsBackAndSyncAvoidsRerender() {
  section('subclip thumbnail fallback + stable sync');
  const payload = JSON.parse(JSON.stringify(projectPayload()));
  payload.hierarchy.model_shots[1].subclips.forEach((clip) => { clip.path = ''; });
  payload.playlist = [
    { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6, start: 0, end: 6 },
    { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5, start: 6, end: 14.5 },
  ];
  const env = await bootReady({
    projectPayload: payload,
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const player = document.getElementById('player');
  const mount = document.getElementById('shotNavigatorMount');

  app.navigator.selectShot('S02', { silent: true });
  await settle();
  const subThumbs = mount.querySelectorAll('[data-subclip-id] .subclip-thumb');
  const srcs = subThumbs.map((thumb) => String(thumb.getAttribute('src') || ''));
  check('subclips without own media fall back to rendered shot clip',
    srcs.length >= 2 && srcs.every((src) => src.includes(encodeURIComponent('/tmp/s02.mp4'))),
    JSON.stringify(srcs));

  let renders = 0;
  const originalRender = app.navigator._render.bind(app.navigator);
  app.navigator._render = function () {
    renders += 1;
    return originalRender();
  };
  player.currentTime = 6.2;
  player.dispatch('timeupdate');
  player.currentTime = 6.4;
  player.dispatch('timeupdate');
  await settle();
  check('timeupdate within same active subclip does not rerender navigator',
    renders <= 1,
    String(renders));
}

async function testPlaylistNavigatorDoesNotJumpBack() {
  section('playlist navigator absolute sync');
  const env = await bootReady({
    projectPayload: threeShotPlaylistPayload(),
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const player = document.getElementById('player');
  const mount = document.getElementById('shotNavigatorMount');

  const third = mount && mount.querySelector('[data-shot-id="S03"] .shot-card-main');
  if (!third) {
    check('third shot card exists', false, mount && mount.textContent);
    check('clicking third shot remains selected after timeupdate', false, 'missing third shot');
    return;
  }
  check('third shot card exists', true);
  third.click();
  await settle();
  player.currentTime = 0.3;
  player.dispatch('timeupdate');
  await settle();
  check('clicking third shot remains selected after timeupdate',
    app.navigator && app.navigator.selectedShotId === 'S03',
    app.navigator && app.navigator.selectedShotId);
}

async function testRunCompleteRefreshesCurrentProjectForBgm() {
  section('run complete refresh for preview bgm');
  const refreshed = Object.assign(projectPayload(), {
    stitch_pending: true,
    final_video_path: '',
    playlist: [
      { shot_id: 'S01', path: '/tmp/s01.mp4', duration_sec: 6, start: 0, end: 6 },
      { shot_id: 'S02', path: '/tmp/s02.mp4', duration_sec: 8.5, start: 6, end: 14.5 },
    ],
    preview_bgm: { volume: 0.35, tracks: [{ path: '/mnt/music/bgm_after_complete.mp3' }] },
  });
  const env = await bootReady({
    projectPayload: Object.assign(projectPayload(), {
      stitch_pending: false,
      final_video_path: '/tmp/old_final.mp4',
    }),
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
    onFetch: (url, method) => {
      if (method === 'GET' && url.includes('/api/0907/project?dir=')) {
        return { json: refreshed };
      }
      return null;
    },
  });
  const app = env.sandbox.window.aicut;
  app.onEvent({
    type: 'run_complete',
    status: 'completed',
    result: {},
    progress: { overall_pct: 100 },
  });
  await settle();
  check('run_complete without project_dir still reloads current project',
    env.fetchLog.some((f) => f.method === 'GET' && f.url.includes('/api/0907/project?dir=')),
    JSON.stringify(env.fetchLog.map((f) => f.url)));
  check('reloaded payload enables preview bgm playback',
    !!app._bgmAudio && String(app._bgmAudio.getAttribute('src') || '').includes('bgm_after_complete.mp3'),
    app._bgmAudio && app._bgmAudio.getAttribute('src'));
}

async function testLaunchPostsDedicatedEndpoint() {
  section('launch endpoint');
  const launched = [];
  const env = await bootReady({
    responses: { '/api/0907/project': projectPayload() },
    onFetch: (url, method, body) => {
      if (url.includes('/api/0907/runs') && method === 'POST') {
        launched.push(body);
        return { json: { run_id: 'run_new_0907', status: 'running', policy_version: '0907' } };
      }
      if (url.replace(/\?.*$/, '').endsWith('/api/runs') && method === 'POST') {
        launched.push({ legacy: true, body });
        return { json: { run_id: 'run_legacy', status: 'running' } };
      }
      return null;
    },
  });
  const app = env.sandbox.window.aicut;
  await app._launch('一段完整的口播稿，用于生成视频。');
  await settle();
  const posts = env.fetchLog.filter((f) => f.method === 'POST');
  check('launch posts to /api/0907/runs',
    posts.some((f) => f.url.includes('/api/0907/runs')),
    JSON.stringify(posts.map((f) => f.url)));
  check('launch does not post to bare /api/runs',
    !posts.some((f) => f.url.replace(/\?.*$/, '').endsWith('/api/runs')),
    JSON.stringify(posts.map((f) => f.url)));
  check('launch payload carries input_text',
    launched.length >= 1 && !launched[0].legacy && typeof launched[0].input_text === 'string',
    JSON.stringify(launched[0]));
}

function testResponsiveCss() {
  section('responsive css');
  const css = read(path.join(DIR0907, 'navigator.css'));
  check('navigator css enables horizontal scrolling',
    /overflow-x\s*:\s*auto/.test(css), 'missing overflow-x: auto');
  check('navigator css includes responsive media query',
    /@media[^{]*max-width\s*:\s*\d+px/.test(css), 'missing max-width media query');
  check('navigator css keeps mount below viewer without overlaying',
    /position\s*:\s*relative/.test(css) || /display\s*:\s*block/.test(css),
    'missing non-overlay block layout rule');
  check('navigator renders paragraph groups in one horizontal row',
    /\.shot-group\b/.test(css) && /data-shot-group/.test(read(path.join(DIR0907, 'shot_navigator.js'))),
    'missing grouped strip');
  check('thumbnail keeps full frame visible',
    /\.subclip-thumb[\s\S]*object-fit\s*:\s*contain/.test(css),
    'thumbnail still crops with cover');
}

async function testMentionPickerAndTimelineDrop() {
  section('@ mention + timeline drop');
  const env = await bootReady({
    localStore: {
      'aicut.newvis0907.lastSession': JSON.stringify({ projectDir: PROJECT_DIR, runId: 'run_demo_0907' }),
    },
  });
  const { document, sandbox } = env;
  const app = sandbox.window.aicut;
  const input = document.getElementById('composerInput');
  const picker = document.getElementById('mentionPicker');

  check('composer has a mention picker', !!picker, 'missing mentionPicker');
  if (!picker) return;

  input.value = '@';
  input.dispatch('input', {});
  await settle();
  const rows = picker.querySelectorAll('[data-mention-token]');
  check('typing @ lists the film\'s shots', !picker.hidden && rows.length >= 2,
    picker.hidden + ' rows=' + rows.length);
  check('mention rows show shot number and chapter',
    rows.length > 0 && rows[0].textContent.includes('01'),
    rows[0] && rows[0].textContent);

  input.value = '@02';
  input.dispatch('input', {});
  await settle();
  const filtered = picker.querySelectorAll('[data-mention-token]');
  check('mention list filters by what is typed',
    filtered.length === 1 && filtered[0].dataset.mentionToken === '@镜头02',
    filtered.map((r) => r.dataset.mentionToken).join(','));

  filtered[0].click();
  await settle();
  check('picking a shot inserts its mention token',
    input.value.trim() === '@镜头02', input.value);
  check('mention picker closes after picking', picker.hidden === true, String(picker.hidden));

  check('timeline shot clip maps to a shot mention',
    app.mentionForClip('shots', { shot_id: 'S02' }) === '@镜头02',
    app.mentionForClip('shots', { shot_id: 'S02' }));
  check('timeline footage clip maps to a subclip mention',
    app.mentionForClip('footage', { shot_id: 'S02', subclip_id: 'S02::subclip::2' }) === '@子镜头2.2',
    app.mentionForClip('footage', { shot_id: 'S02', subclip_id: 'S02::subclip::2' }));

  input.value = '';
  const composer = document.getElementById('composer');
  const transfer = {
    data: { 'text/plain': '@镜头01' },
    getData(type) { return this.data[type] || ''; },
    setData(type, value) { this.data[type] = value; },
  };
  composer.dispatch('drop', { dataTransfer: transfer });
  await settle();
  check('dropping a timeline clip inserts its mention',
    input.value.indexOf('@镜头01') === 0, input.value);

  input.value = '@镜头02 换成海边落日';
  await app.send();
  await settle();
  const chat = env.fetchLog.filter((f) => f.url.includes('/api/0907/project/chat')
    && f.method === 'POST');
  check('mention text is sent to the model as-is',
    chat.length > 0 && chat[chat.length - 1].body.message === '@镜头02 换成海边落日',
    JSON.stringify(chat.map((c) => c.body && c.body.message)));
}

(async function main() {
  testFilesExist();
  testSourceIsolationContract();
  testMarkupContract();
  testLoginBranding();
  testScriptSyntaxLoads();
  await testBootAndNavigatorRender();
  await testSelectionAndSync();
  await testTransactionsUxAndActions();
  await testSubclipOpsAndClear();
  await testQualityPanel();
  await testPreviewBgmAndTransitions();
  await testPreviewBgmFallbackAndReadyRetry();
  await testThumbnailFallsBackToRenderedShot();
  await testSubclipThumbnailFallsBackAndSyncAvoidsRerender();
  await testPlaylistNavigatorDoesNotJumpBack();
  await testRunCompleteRefreshesCurrentProjectForBgm();
  await testLaunchPostsDedicatedEndpoint();
  await testMentionPickerAndTimelineDrop();
  testResponsiveCss();

  console.log('\n' + '='.repeat(56));
  console.log(`pass ${PASS.length} / ${PASS.length + FAIL.length}`);
  if (FAIL.length) {
    console.log('\nfailures:');
    FAIL.forEach((f) => console.log('  - ' + f));
    process.exit(1);
  }
  console.log('all passed');
})();
