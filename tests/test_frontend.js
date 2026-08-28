/* Frontend behaviour test.
 *
 * Loads the real index.html and the real scripts into a minimal DOM, feeds them
 * payloads dumped from the real backend (tests/fixtures/api_payloads.json), and
 * asserts on the resulting DOM. That covers the things a syntax check cannot:
 * ids that exist in one file and not the other, event payload fields the
 * renderer reads but the server never sends, and clip geometry.
 *
 * Regenerate fixtures with: python tests/dump_fixtures.py
 * Run:                      node tests/test_frontend.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const { MiniDocument } = require('./minidom');

const ROOT = path.resolve(__dirname, '..');
const FIXTURES = path.join(ROOT, 'tests', 'fixtures', 'api_payloads.json');

const PASS = [];
const FAIL = [];

function check(name, cond, detail) {
  if (cond) { PASS.push(name); console.log('  ok   ' + name); }
  else { FAIL.push(name + (detail ? ' — ' + detail : '')); console.log('  FAIL ' + name + (detail ? '  (' + detail + ')' : '')); }
}
function section(title) { console.log('\n=== ' + title + ' ==='); }

if (!fs.existsSync(FIXTURES)) {
  console.error('缺少 fixtures，先运行: python tests/dump_fixtures.py');
  process.exit(2);
}
const FX = JSON.parse(fs.readFileSync(FIXTURES, 'utf8'));

// ---------------------------------------------------------------- harness
function boot(options) {
  options = options || {};
  const html = fs.readFileSync(path.join(ROOT, 'static', 'index.html'), 'utf8');
  const document = new MiniDocument(html);

  const jsErrors = [];
  const toasts = [];
  const fetchLog = [];
  const sockets = [];

  const responses = Object.assign({
    '/api/config': FX.config,
    '/api/status': FX.status,
    '/api/runs': FX.runs,
    '/api/projects': FX.projects,
    '/api/project': FX.project,
  }, options.responses || {});

  function resolve(url) {
    const p = url.split('?')[0].replace(/^https?:\/\/[^/]+/, '');
    if (Object.prototype.hasOwnProperty.call(responses, p)) return responses[p];
    if (p.startsWith('/api/runs/')) return { status: 'ok' };
    return null;
  }

  const fetchImpl = async (url, init) => {
    const method = (init && init.method) || 'GET';
    const body = init && init.body ? JSON.parse(init.body) : null;
    fetchLog.push({ url: String(url), method: method, body: body });

    const override = options.onFetch && options.onFetch(String(url), method, body);
    if (override) {
      return {
        ok: override.status ? override.status < 400 : true,
        status: override.status || 200,
        statusText: '',
        text: async () => JSON.stringify(override.json === undefined ? {} : override.json),
      };
    }
    const payload = resolve(String(url));
    return {
      ok: payload !== null,
      status: payload !== null ? 200 : 404,
      statusText: 'Not Found',
      text: async () => JSON.stringify(payload === null ? { detail: 'not found' } : payload),
    };
  };

  class FakeWebSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 1;
      this.sent = [];
      sockets.push(this);
      setTimeout(() => { if (this.onopen) this.onopen(); }, 0);
    }
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; if (this.onclose) this.onclose(); }
    emit(obj) { if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) }); }
  }
  FakeWebSocket.OPEN = 1;

  const timers = [];
  const sandbox = {
    console: console,
    document: document,
    fetch: fetchImpl,
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
    setTimeout: (fn, ms) => { timers.push(fn); return timers.length; },
    clearTimeout: () => {},
    setInterval: () => 0,          // polling would only add noise here
    clearInterval: () => {},
    alert: (msg) => { toasts.push({ kind: 'alert', text: msg }); },
    confirm: () => (options.confirm === undefined ? true : options.confirm),
    encodeURIComponent: encodeURIComponent,
    location: { origin: 'http://localhost:8097', host: 'localhost:8097', protocol: 'http:' },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.window.addEventListener = () => {};

  vm.createContext(sandbox);
  for (const file of ['timeline.js', 'editor.js', 'chat.js', 'studio.js']) {
    const code = fs.readFileSync(path.join(ROOT, 'static', 'js', file), 'utf8');
    try {
      vm.runInContext(code, sandbox, { filename: file });
    } catch (err) {
      jsErrors.push(file + ': ' + err.message);
    }
  }

  return { document, sandbox, jsErrors, fetchLog, sockets, toasts, timers, FakeWebSocket };
}

/** Boot and run through DOMContentLoaded + async config load. */
async function bootReady(options) {
  const env = boot(options);
  env.document.dispatch('DOMContentLoaded');
  await settle();
  return env;
}

function settle(rounds) {
  // Drain the microtask queue. boot() chains config -> status -> runs ->
  // projects, each with two awaits, so this needs to be generous.
  let p = Promise.resolve();
  for (let i = 0; i < (rounds || 80); i++) p = p.then(() => {});
  return p;
}

function toastTexts(document) {
  return document.getElementById('toasts').children.map((t) => t.textContent);
}

// ---------------------------------------------------------------- tests
async function testMarkupContract() {
  section('markup ↔ script contract');
  const env = boot();
  check('脚本加载无异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));

  const { document } = env;
  // every id the scripts look up must exist in index.html
  const ids = new Set();
  for (const file of ['timeline.js', 'editor.js', 'chat.js', 'studio.js']) {
    const code = fs.readFileSync(path.join(ROOT, 'static', 'js', file), 'utf8');
    const re = /getElementById\('([^']+)'\)/g;
    let m;
    while ((m = re.exec(code)) !== null) ids.add(m[1]);
  }
  const missing = [...ids].filter((id) => document.getElementById(id) === null);
  check('脚本引用的 id 都存在于 index.html', missing.length === 0, missing.join(', '));
  check('检查到的 id 数量合理', ids.size > 40, String(ids.size));

  // and every script index.html references must exist on disk
  const html = fs.readFileSync(path.join(ROOT, 'static', 'index.html'), 'utf8');
  const assets = [...html.matchAll(/(?:src|href)="\/static\/([^"]+)"/g)].map((m) => m[1]);
  const absent = assets.filter((a) => !fs.existsSync(path.join(ROOT, 'static', a)));
  check('index.html 引用的静态资源都存在', absent.length === 0, absent.join(', '));
  check('引用了全部 4 个脚本', assets.filter((a) => a.endsWith('.js')).length === 4);

  testHiddenIsEffective(document);
}

/* The whole show/hide mechanism rests on the `hidden` attribute, but a class
 * rule that sets `display` outranks the UA's `[hidden]` rule — which is how the
 * confirm dialog ended up permanently on screen. Verify that every element we
 * toggle is actually hideable. */
function testHiddenIsEffective(document) {
  const css = fs.readFileSync(path.join(ROOT, 'static', 'css', 'studio.css'), 'utf8');

  const override = /\[hidden\][^{]*\{[^}]*display\s*:\s*none\s*!important/.test(css);
  check('CSS 有 [hidden] 全局兜底规则', override,
        '缺少 [hidden] { display: none !important }');

  // selectors that set `display`, ignoring pseudo-classes and at-rules
  const displaySelectors = [];
  for (const m of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (!/(^|[;{\s])display\s*:/.test(m[2])) continue;
    const raw = m[1].split('\n').pop().trim();
    raw.split(',').map((s) => s.trim()).forEach((sel) => {
      if (!sel || sel.includes('@') || sel.includes(':') || sel.includes('>')) return;
      displaySelectors.push(sel);
    });
  }
  check('解析到设置 display 的选择器', displaySelectors.length > 5,
        String(displaySelectors.length));

  // Elements hidden in the markup, plus those the scripts toggle at runtime —
  // `playerEmpty` is only ever hidden from JS and was broken the same way.
  const toggled = new Set(
    document.querySelectorAll('[hidden]').map((el) => el.id).filter(Boolean));
  for (const file of ['timeline.js', 'editor.js', 'chat.js', 'studio.js']) {
    const code = fs.readFileSync(path.join(ROOT, 'static', 'js', file), 'utf8');
    for (const m of code.matchAll(/(?:\.|\b)(\w+)\.hidden\s*=/g)) {
      if (document.getElementById(m[1])) toggled.add(m[1]);
    }
    for (const m of code.matchAll(/getElementById\('([^']+)'\)\.hidden\s*=/g)) {
      toggled.add(m[1]);
    }
  }
  check('识别出被切换显示的元素', toggled.size >= 8, String(toggled.size));
  check('包含仅由 JS 切换的元素', toggled.has('playerEmpty'), [...toggled].join(','));

  const atRisk = [];
  toggled.forEach((id) => {
    const el = document.getElementById(id);
    if (!el || !el.parentElement) return;
    const hit = displaySelectors.find((sel) => {
      try { return el.parentElement.querySelectorAll(sel).includes(el); }
      catch (e) { return false; }
    });
    if (hit) atRisk.push(`${id} (被 ${hit} 的 display 覆盖)`);
  });
  check('被 display 规则覆盖的元素有兜底', !atRisk.length || override, atRisk.join('; '));
  check('确实存在需要兜底的元素', atRisk.length > 0,
        '若为 0 说明检测逻辑没生效');
}

async function testBootstrap() {
  section('bootstrap');
  const env = await bootReady();
  const { document } = env;

  check('启动无 JS 异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));
  check('拉取了 /api/config', env.fetchLog.some((f) => f.url.includes('/api/config')));
  check('拉取了 /api/status', env.fetchLog.some((f) => f.url.includes('/api/status')));

  const labels = document.getElementById('tlLabels').children;
  const expected = FX.config.tracks.length;
  check('时间轴按配置生成轨道标签', labels.length === expected + 1,
        `${labels.length} vs ${expected}+1`);
  const labelTexts = labels.slice(1).map((l) => l.textContent.trim());
  check('轨道标签用的是后端下发的中文名',
        labelTexts.join(',') === FX.config.tracks.map((t) => t.label).join(','),
        labelTexts.join(','));

  check('模型未就绪时禁用开始按钮', document.getElementById('startBtn').disabled);
  check('模型状态提示可读',
        document.getElementById('modelText').textContent.includes('未就绪'),
        document.getElementById('modelText').textContent);

  const note = document.getElementById('autonomyNote').textContent;
  check('展示完全自主模式说明', note.includes('完全自主'), note);
  check('说明里带上步数上限', note.includes(String(FX.config.agent.max_steps)), note);

  // config toggles must actually gate the UI
  const off = await bootReady({
    responses: {
      '/api/config': Object.assign({}, FX.config, {
        features: Object.assign({}, FX.config.features, { chat_edit: false, editor: false }),
      }),
    },
  });
  check('关闭对话编辑后 tab 禁用', off.document.getElementById('tabChat').disabled);
  check('关闭编辑后 tab 禁用', off.document.getElementById('tabEditor').disabled);
}

async function testInputGating() {
  section('input gating');
  const env = await bootReady({ responses: { '/api/status': modelReady() } });
  const { document } = env;

  const script = document.getElementById('scriptInput');
  script.value = '这是一段测试口播稿。';
  script.dispatch('input');
  check('字数统计更新',
        document.getElementById('scriptCount').textContent === '10 字',
        document.getElementById('scriptCount').textContent);
  check('模型就绪且有文本时可开始', document.getElementById('startBtn').disabled === false);

  script.value = '';
  script.dispatch('input');
  check('清空后禁用开始', document.getElementById('startBtn').disabled === true);
  check('提示说明为何不能开始',
        document.getElementById('startHint').textContent.includes('口播稿'),
        document.getElementById('startHint').textContent);
}

function modelReady() {
  return Object.assign({}, FX.status, {
    model: Object.assign({}, FX.status.model, { loaded: true, active_profile: 'sft_v6_merged' }),
  });
}

async function testLiveRun() {
  section('live run streaming');
  const env = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url, method) => {
      if (url.includes('/api/runs') && method === 'POST') {
        return { json: { run_id: 'run_test_0001', status: 'running' } };
      }
      return null;
    },
  });
  const { document } = env;

  document.getElementById('scriptInput').value = '测试稿件';
  document.getElementById('scriptInput').dispatch('input');
  document.getElementById('startBtn').click();
  await settle();

  check('已建立 WebSocket', env.sockets.length === 1, String(env.sockets.length));
  check('WebSocket 指向该 run',
        env.sockets[0] && env.sockets[0].url.endsWith('/ws/run_test_0001'),
        env.sockets[0] && env.sockets[0].url);
  check('运行中显示暂停/取消', !document.getElementById('pauseBtn').hidden
        && !document.getElementById('cancelBtn').hidden);

  const ws = env.sockets[0];
  ws.emit({ type: 'replay_start', event_count: 0 });
  ws.emit({ type: 'run_started', run_id: 'run_test_0001' });
  ws.emit({
    type: 'stage_update', stage: 'shot_planning', status: 'started',
    message: '镜头规划', progress_pct: 12,
  });
  check('阶段名转成中文',
        document.getElementById('phaseLabel').textContent === '镜头规划',
        document.getElementById('phaseLabel').textContent);
  check('总进度跟随 stage_update',
        document.getElementById('overallPct').textContent === '12%',
        document.getElementById('overallPct').textContent);

  ws.emit({
    type: 'thinking', stage: 'shot_planning', step: 3, tool_name: 'Tool_PlanShots',
    thinking: '先按段落切镜头，再决定重点镜头。', parse_ok: true,
    raw_output: '<tool_call>{"name":"Tool_PlanShots"}</tool_call>',
  });
  check('步数显示来自 thinking',
        document.getElementById('statSteps').textContent === '3',
        document.getElementById('statSteps').textContent);
  const trace = document.getElementById('trace');
  check('思考进入执行轨迹', trace.querySelectorAll('.ev.think').length === 1);
  check('思考正文可读',
        trace.textContent.includes('先按段落切镜头'), trace.textContent.slice(0, 60));

  ws.emit({
    type: 'tool_call', stage: 'shot_planning', step: 3, tool_name: 'Tool_PlanShots',
    arguments: { max_shots: 4 }, precondition_missing: [],
  });
  check('工具调用进入轨迹', trace.querySelectorAll('.ev.call').length === 1);
  check('工具名显示为中文', trace.textContent.includes('镜头规划'));
  check('参数被渲染', trace.textContent.includes('max_shots'));

  ws.emit({
    type: 'tool_result', stage: 'shot_planning', step: 3,
    tool_name: 'Tool_PlanShots', status: 'success',
  });
  check('工具结果进入轨迹', trace.querySelectorAll('.ev.result').length === 1);

  // the important one: a real progress snapshot must fill the lanes
  ws.emit({ type: 'progress', progress: FX.progress_partial });
  const lanes = document.getElementById('tlTracks').children;
  check('时间轴生成全部轨道', lanes.length === FX.config.tracks.length, String(lanes.length));

  const shotLane = lanes[FX.config.tracks.findIndex((t) => t.key === 'shots')];
  const shotClips = shotLane.querySelectorAll('.tl-clip');
  check('镜头轨渲染了片段', shotClips.length === FX.progress_partial.clips.shots.length,
        `${shotClips.length} vs ${FX.progress_partial.clips.shots.length}`);

  const first = shotClips[0];
  const expectLeft = FX.progress_partial.clips.shots[0].start * 60;
  check('片段按时间定位', first.style.left === expectLeft + 'px',
        `${first.style.left} vs ${expectLeft}px`);
  check('片段宽度为正',
        parseFloat(first.style.width) > 0, first.style.width);
  check('片段标题带镜头文案', first.textContent.length > 0, first.textContent);

  const renderLane = lanes[FX.config.tracks.findIndex((t) => t.key === 'render')];
  const pending = renderLane.querySelectorAll('.tl-clip.pending');
  check('未产出的片段渲染为待生成',
        pending.length === FX.progress_partial.clips.render.filter((c) => !c.ready).length,
        String(pending.length));

  const rail = document.getElementById('trackRail');
  check('进度导轨按轨道生成', rail.children.length === FX.config.tracks.length);
  const shotsStat = FX.progress_partial.stats.shots;
  const shotsRail = rail.querySelector('.rail-item[data-track="shots"]');
  check('导轨显示完成数',
        shotsRail.querySelector('.rail-count').textContent === `${shotsStat.done}/${shotsStat.total}`,
        shotsRail.querySelector('.rail-count').textContent);

  check('总时长已显示',
        document.getElementById('statDuration').textContent !== '0:00',
        document.getElementById('statDuration').textContent);

  const body = document.getElementById('consoleBody');
  const logsBefore = body.children.length;   // replay_start already logged one line
  ws.emit({ type: 'console_log', level: 'warn', text: '素材检索命中率偏低', timestamp: '10:00:01' });
  check('日志写入控制台', body.children.length === logsBefore + 1,
        `${body.children.length} vs ${logsBefore}+1`);
  const lastLine = body.children[body.children.length - 1];
  check('日志分级渲染', lastLine.classList.contains('warn'), lastLine.className);
  check('日志保留服务端时间戳', lastLine.textContent.includes('10:00:01'), lastLine.textContent);
  check('日志计数更新',
        document.getElementById('consoleCount').textContent === String(logsBefore + 1),
        document.getElementById('consoleCount').textContent);

  ws.emit({ type: 'error', stage: 'render', message: '第 3 镜渲染失败' });
  check('错误进入轨迹', trace.querySelectorAll('.ev.error').length === 1);
  check('错误也写入日志', body.textContent.includes('第 3 镜渲染失败'));

  ws.emit({
    type: 'tool_result', stage: 'narration', step: 4, tool_name: 'Tool_RenderShots',
    status: 'blocked_precondition',
    observation: { missing: ['narration_ready'] },
  });
  check('前置条件不足单独标记', trace.querySelectorAll('.ev.blocked').length === 1);
  check('前置条件缺失项可读', trace.textContent.includes('narration_ready'));

  return env;
}

async function testCompletionAndProject(env) {
  section('completion → editor handoff');
  const { document } = env;
  const ws = env.sockets[0];

  env.responses = null;
  ws.emit({
    type: 'run_complete', status: 'completed', elapsed_seconds: 123,
    result: { final_video_path: '/tmp/aicut/final.mp4', project_dir: FX.project_dir },
    project_dir: FX.project_dir,
    progress: FX.progress_final,
  });
  await settle(20);

  check('完成后总进度 100%',
        document.getElementById('overallPct').textContent === '100%',
        document.getElementById('overallPct').textContent);
  check('完成后隐藏暂停/取消',
        document.getElementById('pauseBtn').hidden && document.getElementById('cancelBtn').hidden);
  check('成片挂到播放器',
        (document.getElementById('player').getAttribute('src') || '').includes('/api/media?path='),
        document.getElementById('player').getAttribute('src'));
  check('播放器占位隐藏', document.getElementById('playerEmpty').hidden);
  check('完成后自动打开工程',
        env.fetchLog.some((f) => f.url.includes('/api/project?dir=')),
        'no project fetch');
  check('编辑面板已展开', document.getElementById('editorBody').hidden === false);
}

async function testEditor() {
  section('manual editor');
  let opsSent = null;
  const env = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url, method, body) => {
      if (url.includes('/api/project/ops') && method === 'POST') {
        opsSent = body;
        return { json: FX.project_after_edit };
      }
      if (url.includes('/api/project/regenerate')) {
        return { json: { status: 'started', job_id: 'edit_abc123', steps: [], restitch: true } };
      }
      return null;
    },
  });
  const { document } = env;
  const studio = env.sandbox.studio;

  await studio.openProject(FX.project_dir);
  await settle();
  check('工程载入无异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));
  check('载入后时间轴有内容',
        document.getElementById('tlTracks').querySelectorAll('.tl-clip').length > 0);

  // select a shot through the timeline, exactly as a user would
  const shotIdx = FX.config.tracks.findIndex((t) => t.key === 'shots');
  const clip = document.getElementById('tlTracks').children[shotIdx].querySelectorAll('.tl-clip')[0];
  clip.click();
  await settle();

  const shotId = FX.project.progress.clips.shots[0].shot_id;
  check('点击片段切到编辑页',
        document.querySelector('.tab-panel[data-panel="editor"]').classList.contains('active'));
  check('编辑面板显示镜头 id',
        document.getElementById('editClipId').textContent === shotId,
        document.getElementById('editClipId').textContent);
  check('旁白框预填原文',
        document.getElementById('editNarration').value.length > 0,
        document.getElementById('editNarration').value);
  check('字幕框预填原文', document.getElementById('editSubtitle').value.length > 0);

  const spotlightRow = FX.project.digest.shots.find((s) => s.is_spotlight);
  const plainRow = FX.project.digest.shots.find((s) => !s.is_spotlight);
  check('非重点镜头隐藏模板槽位',
        plainRow ? document.getElementById('fieldSlots').hidden : true);

  // apply a subtitle edit
  document.getElementById('editSubtitle').value = '修正后的字幕文案';
  document.querySelector('[data-op="set_subtitle"]').click();
  await settle();

  check('提交了 set_subtitle',
        opsSent && opsSent.ops.length === 1 && opsSent.ops[0].op === 'set_subtitle',
        JSON.stringify(opsSent && opsSent.ops));
  check('操作带上 shot_id', opsSent.ops[0].shot_id === shotId, opsSent.ops[0].shot_id);
  check('操作带上新文案', opsSent.ops[0].text === '修正后的字幕文案');
  check('操作带上工程目录', opsSent.project_dir === FX.project_dir);
  check('成功后弹出提示', toastTexts(document).length > 0, JSON.stringify(toastTexts(document)));

  const dirtyShots = Object.keys(FX.project_after_edit.dirty);
  check('后端返回了脏镜头', dirtyShots.length > 0, JSON.stringify(FX.project_after_edit.dirty));
  const dirtyClips = document.getElementById('tlTracks').querySelectorAll('.tl-clip.dirty');
  check('脏镜头在时间轴上标记', dirtyClips.length > 0, String(dirtyClips.length));
  check('展示待重新生成清单',
        document.getElementById('pendingBox').hidden === false);
  check('待生成清单非空',
        document.getElementById('pendingList').children.length ===
          FX.project_after_edit.pending_steps.length,
        String(document.getElementById('pendingList').children.length));
  check('重渲染按钮可用', document.getElementById('regenBtn').disabled === false);

  // guard rails: an empty required field must not reach the server
  opsSent = null;
  document.getElementById('editFootageQuery').value = '';
  document.getElementById('editFootagePath').value = '';
  document.querySelector('[data-op="replace_footage"]').click();
  await settle();
  check('缺检索词时不发请求', opsSent === null);
  check('缺检索词时给出提示',
        toastTexts(document).some((t) => t.includes('画面描述')),
        JSON.stringify(toastTexts(document)));

  document.getElementById('editFootageQuery').value = '城市夜景航拍';
  document.querySelector('[data-op="replace_footage"]').click();
  await settle();
  check('填写后提交 replace_footage',
        opsSent && opsSent.ops[0].op === 'replace_footage'
        && opsSent.ops[0].query === '城市夜景航拍',
        JSON.stringify(opsSent && opsSent.ops[0]));

  // regeneration
  document.getElementById('regenBtn').click();
  await settle();
  check('触发重渲染请求',
        env.fetchLog.some((f) => f.url.includes('/api/project/regenerate')));
  check('重渲染复用 run 视图并连上新 job',
        env.sockets.some((s) => s.url.endsWith('/ws/edit_abc123')),
        env.sockets.map((s) => s.url).join(','));

  check('全程无 JS 异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));
}

async function testChat() {
  section('conversational editing');
  let chatBody = null;
  const agentReply = {
    intent: 'edit',
    reply: '已把开头那句改得更有冲击力。',
    ops: [{ op: 'set_narration', shot_id: 'S01', text: '新的开场白' }],
    targets: ['S01'],
    confidence: 0.9,
    applied: true,
    needs_confirm: [],
    // The server kicks off a scoped re-render for the edited shots and hands
    // back its job id; the panel must pick it up instead of telling the user
    // to re-render by hand.
    regen_job_id: 'chat_regen_0001',
    regen_steps: [{ tool: 'Tool_BatchVisualAudio' }, { tool: 'Tool_StitchVideo' }],
    memory: { turns: [{}, {}], token_estimate: 820, budget: 12000, compressions: 0, summary: '' },
    project: FX.project_after_edit,
  };

  const env = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url, method, body) => {
      if (url.includes('/api/project/chat') && !url.includes('memory') && !url.includes('reset')) {
        chatBody = body;
        return { json: agentReply };
      }
      if (url.includes('/api/project/chat/memory')) {
        return { json: { turns: [], token_estimate: 0, budget: 12000, compressions: 0 } };
      }
      return null;
    },
  });
  const { document } = env;

  await env.sandbox.studio.openProject(FX.project_dir);
  await settle();

  document.getElementById('chatInput').value = '开头那句太平淡了';
  document.getElementById('chatSend').click();
  await settle();

  check('发送了对话请求', chatBody !== null);
  check('请求带上工程目录', chatBody && chatBody.project_dir === FX.project_dir);
  check('请求带上用户原话', chatBody && chatBody.message === '开头那句太平淡了');

  const log = document.getElementById('chatLog');
  check('用户气泡已渲染', log.querySelectorAll('.msg.user').length === 1);
  check('模型回复已渲染', log.querySelectorAll('.msg.bot').length === 1);
  check('思考中气泡已移除', !log.textContent.includes('正在定位'));
  check('回复正文可读', log.textContent.includes('冲击力'), log.textContent.slice(0, 80));
  check('展示了将执行的操作', log.querySelectorAll('.msg-ops').length === 1);
  check('操作描述用中文标签',
        log.querySelector('.msg-ops').textContent.includes('旁白'),
        log.querySelector('.msg-ops').textContent);
  check('记忆用量已显示',
        document.getElementById('chatMemory').textContent.includes('820'),
        document.getElementById('chatMemory').textContent);
  check('自动应用后触发自动重渲染',
        toastTexts(document).some((t) => t.includes('自动重渲染')),
        JSON.stringify(toastTexts(document)));
  check('告知只重渲染改动的镜头',
        log.textContent.includes('只改动的镜头'),
        log.textContent.slice(-120));

  // high-risk edits must wait for confirmation
  const risky = Object.assign({}, agentReply, {
    reply: '这会删掉第 4 个镜头。',
    ops: [{ op: 'delete_shot', shot_id: 'S04' }],
    applied: false,
    needs_confirm: ['delete_shot'],
  });
  const env2 = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url) => {
      if (url.includes('/api/project/chat/memory')) return { json: { turns: [] } };
      if (url.includes('/api/project/chat')) return { json: risky };
      return null;
    },
  });
  await env2.sandbox.studio.openProject(FX.project_dir);
  await settle();
  env2.document.getElementById('chatInput').value = '把最后那个镜头删了';
  env2.document.getElementById('chatSend').click();
  await settle();

  const actions = env2.document.querySelectorAll('.msg-actions .btn');
  check('高风险操作提供确认按钮', actions.length === 2, String(actions.length));
  check('确认按钮文案明确',
        actions[0] && actions[0].textContent.includes('确认'), actions[0] && actions[0].textContent);
  check('确认前未应用', env2.document.getElementById('confirmModal').hidden);
  actions[0].click();
  await settle();
  check('点击后弹出确认框', env2.document.getElementById('confirmModal').hidden === false);
  check('确认框列出具体操作',
        env2.document.getElementById('confirmOps').textContent.includes('S04'),
        env2.document.getElementById('confirmOps').textContent);

  // server failure must surface as a readable bubble, not an exception
  const env3 = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url) => {
      if (url.includes('/api/project/chat/memory')) return { json: { turns: [] } };
      if (url.includes('/api/project/chat')) {
        return { status: 503, json: { detail: '模型尚未加载完成，请稍候' } };
      }
      return null;
    },
  });
  await env3.sandbox.studio.openProject(FX.project_dir);
  await settle();
  env3.document.getElementById('chatInput').value = '改一下开头';
  env3.document.getElementById('chatSend').click();
  await settle();

  const errBubble = env3.document.querySelector('.msg.bot.reject');
  check('后端 503 转为可读气泡', errBubble !== null);
  check('气泡带上后端原因',
        errBubble && errBubble.textContent.includes('模型尚未加载'),
        errBubble && errBubble.textContent);
  check('失败后可以继续发送', env3.document.getElementById('chatSend').disabled === false);
  check('对话异常不抛 JS 错误', env3.jsErrors.length === 0, env3.jsErrors.join(' | '));
}

async function testDeletion() {
  section('deleting projects & runs');

  const runs = { runs: [
    { id: 'run_done', title: '已完成的稿件', status: 'completed', overall_pct: 100,
      elapsed_seconds: 90, started_at: '2026-08-12T09:00:00', project_dir: FX.project_dir },
    { id: 'run_live', title: '进行中的稿件', status: 'running', overall_pct: 40,
      elapsed_seconds: 12, started_at: '2026-08-12T09:30:00', project_dir: '' },
  ] };

  let deleted = null;
  const env = await bootReady({
    responses: { '/api/status': modelReady(), '/api/runs': runs },
    onFetch: (url, method) => {
      if (method === 'DELETE') {
        deleted = url;
        if (url.includes('/api/project')) return { json: { deleted: true, freed_bytes: 5242880 } };
        return { json: { deleted: 'run_done' } };
      }
      return null;
    },
  });
  const { document } = env;

  // -- project rows --
  const projItems = document.getElementById('projectList').querySelectorAll('.list-item');
  check('工程列表已渲染', projItems.length === FX.projects.projects.length,
        String(projItems.length));
  const delBtns = document.getElementById('projectList').querySelectorAll('.li-del');
  check('每个成片都有删除按钮', delBtns.length === projItems.length, String(delBtns.length));
  check('删除按钮有说明', delBtns[0].title.includes('删除'), delBtns[0].title);

  // clicking delete must not also open the project (the button sits inside the
  // clickable row, so this depends on stopPropagation)
  const opens = () => env.fetchLog.filter(
    (f) => f.method === 'GET' && f.url.includes('/api/project?dir=')).length;
  const opensBefore = opens();
  delBtns[0].click();
  await settle();
  check('删除请求已发出', deleted !== null && deleted.includes('/api/project?dir='), String(deleted));
  check('用 DELETE 方法',
        env.fetchLog.some((f) => f.method === 'DELETE' && f.url.includes('/api/project')));
  check('删除时不会顺手打开该工程', opens() === opensBefore,
        '点击冒泡到了列表行');
  check('提示里带上释放空间',
        toastTexts(document).some((t) => t.includes('5.0 MB')),
        JSON.stringify(toastTexts(document)));
  check('删除后刷新列表',
        env.fetchLog.filter((f) => f.url.includes('/api/projects')).length >= 2);

  // -- cancelled confirmation must not delete --
  const env2 = await bootReady({
    confirm: false,
    responses: { '/api/status': modelReady() },
    onFetch: (url, method) => {
      if (method === 'DELETE') return { json: { deleted: true } };
      return null;
    },
  });
  env2.document.getElementById('projectList').querySelectorAll('.li-del')[0].click();
  await settle();
  check('用户取消确认时不删除',
        !env2.fetchLog.some((f) => f.method === 'DELETE'), '仍发出了 DELETE');

  // -- deleting the project currently open must clear the workspace --
  const env3 = await bootReady({
    responses: { '/api/status': modelReady() },
    onFetch: (url, method) => (method === 'DELETE' ? { json: { deleted: true, freed_bytes: 0 } } : null),
  });
  await env3.sandbox.studio.openProject(FX.project_dir);
  await settle();
  check('打开后编辑面板可见', env3.document.getElementById('editorBody').hidden === false);

  const openRow = env3.document.getElementById('projectList')
    .querySelectorAll('.list-item')
    .find((el) => el.classList.contains('active'));
  check('当前工程行被标记为选中', !!openRow);
  openRow.querySelector('.li-del').click();
  await settle();

  check('删除当前工程后收起编辑面板',
        env3.document.getElementById('editorBody').hidden === true);
  check('删除当前工程后清空对话',
        env3.document.getElementById('chatLog').children.length === 0,
        String(env3.document.getElementById('chatLog').children.length));
  check('删除当前工程后清空时间轴',
        env3.document.getElementById('tlTracks').querySelectorAll('.tl-clip').length === 0);
  check('删除当前工程后恢复播放器占位',
        env3.document.getElementById('playerEmpty').hidden === false);
  check('删除当前工程无 JS 异常', env3.jsErrors.length === 0, env3.jsErrors.join(' | '));

  // -- run history rows --
  const runItems = document.getElementById('runList').querySelectorAll('.list-item');
  check('历史列表已渲染', runItems.length === 2, String(runItems.length));
  check('已结束的记录可删除', runItems[0].querySelector('.li-del') !== null);
  check('运行中的记录不给删除按钮', runItems[1].querySelector('.li-del') === null);

  deleted = null;
  runItems[0].querySelector('.li-del').click();
  await settle();
  check('删除记录调用 /api/runs/{id}',
        deleted === '/api/runs/run_done', String(deleted));

  check('删除流程无 JS 异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));
}

async function testTimelineControls() {
  section('timeline controls');
  const env = await bootReady({ responses: { '/api/status': modelReady() } });
  const { document } = env;

  await env.sandbox.studio.openProject(FX.project_dir);
  await settle();

  const label = document.getElementById('zoomLabel');
  const before = parseFloat(label.textContent);
  document.getElementById('zoomIn').click();
  const after = parseFloat(label.textContent);
  check('放大提高像素密度', after > before, `${before} -> ${after}`);

  document.getElementById('zoomOut').click();
  check('缩小恢复密度', parseFloat(label.textContent) < after);

  document.getElementById('zoomFit').click();
  const canvasWidth = parseFloat(document.getElementById('tlCanvas').style.width);
  check('适配后画布不为空', canvasWidth > 0, String(canvasWidth));
  check('适配后仍保留全部片段',
        document.getElementById('tlTracks').querySelectorAll('.tl-clip').length > 0);

  const ruler = document.getElementById('tlRuler').children;
  check('刻度尺已生成', ruler.length > 1, String(ruler.length));
  check('刻度带时间文字', ruler[1] && ruler[1].textContent.includes(':'),
        ruler[1] && ruler[1].textContent);

  const consoleBody = document.getElementById('consoleBody');
  check('日志面板默认折叠', consoleBody.hidden === true);
  document.getElementById('consoleToggle').click();
  check('点击后展开日志面板', consoleBody.hidden === false);
  document.getElementById('consoleToggle').click();
  check('再次点击折叠', consoleBody.hidden === true);

  ['agent', 'editor', 'chat'].forEach((name) => {
    document.querySelector(`#tabs .tab[data-tab="${name}"]`).click();
    check(`切换到 ${name} 面板`,
          document.querySelector(`.tab-panel[data-panel="${name}"]`).classList.contains('active'));
  });

  check('时间轴操作无 JS 异常', env.jsErrors.length === 0, env.jsErrors.join(' | '));
}

// ---------------------------------------------------------------- main
(async () => {
  try {
    await testMarkupContract();
    await testBootstrap();
    await testInputGating();
    const runEnv = await testLiveRun();
    await testCompletionAndProject(runEnv);
    await testEditor();
    await testChat();
    await testDeletion();
    await testTimelineControls();
  } catch (err) {
    console.error('\n测试自身异常:', err && err.stack || err);
    FAIL.push('harness: ' + (err && err.message));
  }

  console.log('\n' + '='.repeat(60));
  console.log(`通过 ${PASS.length} / ${PASS.length + FAIL.length}`);
  if (FAIL.length) {
    console.log('\n失败项:');
    FAIL.forEach((f) => console.log('  - ' + f));
    process.exit(1);
  }
  console.log('全部通过');
})();
