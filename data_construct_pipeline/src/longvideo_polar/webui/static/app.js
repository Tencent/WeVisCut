'use strict';

// Polling, not websockets: a campaign runs for days and the status document is
// small, so a plain interval survives sleep, proxies and reconnects with no
// reconnection logic to get wrong.
const POLL_MS = 3000;

const $ = (id) => document.getElementById(id);

let modes = [];
let lastStatus = null;

// ---------------------------------------------------------------- utilities

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch { payload = { detail: text }; }
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('bad', isError);
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), isError ? 6000 : 3000);
}

function duration(seconds) {
  if (!seconds || seconds <= 0) return '—';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h < 48) return `${h}h${String(m).padStart(2, '0')}m`;
  return `${Math.floor(h / 24)}天${h % 24}h`;
}

function clockAfter(seconds) {
  if (!seconds || seconds <= 0) return '—';
  const when = new Date(Date.now() + seconds * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  const sameDay = when.toDateString() === new Date().toDateString();
  const time = `${pad(when.getHours())}:${pad(when.getMinutes())}`;
  return sameDay ? time : `${when.getMonth() + 1}/${when.getDate()} ${time}`;
}

const PHASE_ZH = {
  idle: '空闲', running: '运行中', paused: '已暂停',
  stopping: '停止中', stopped: '已停止', finished: '已完成',
};

// ---------------------------------------------------------------- endpoints

async function refreshEndpoints(pool) {
  const data = pool || await api('/api/endpoints');
  const list = $('epList');
  list.innerHTML = '';

  (data.endpoints || []).forEach((ep) => {
    const row = document.createElement('div');
    row.className = 'ep';

    const dotClass = !ep.enabled ? 'off' : (ep.healthy ? 'up' : '');
    const meta = ep.healthy
      ? `${ep.model || '(未知模型)'} · ${ep.latency_ms}ms · 完成 ${ep.completed} 失败 ${ep.failed}`
      : '';
    const error = !ep.healthy && ep.enabled
      ? `<div class="ep-err" title="${escapeHtml(ep.last_error || '')}">${escapeHtml((ep.last_error || '不可达').slice(0, 90))}</div>`
      : '';

    row.innerHTML = `
      <div class="dot ${dotClass}"></div>
      <div class="ep-main">
        <div class="ep-host">${escapeHtml(ep.host)}</div>
        <div class="ep-meta">${escapeHtml(meta)}</div>
        ${error}
      </div>
      <div class="ep-actions">
        <span class="ep-load">${ep.in_flight}/${ep.capacity}</span>
        <input type="number" min="1" max="64" value="${ep.capacity}" style="width:56px"
               title="并发 rollout 数" data-cap="${escapeHtml(ep.base_url)}" />
        <button class="small" data-toggle="${escapeHtml(ep.base_url)}">${ep.enabled ? '停用' : '启用'}</button>
        <button class="small" data-del="${escapeHtml(ep.base_url)}">删除</button>
      </div>`;
    list.appendChild(row);
  });

  if (!(data.endpoints || []).length) {
    list.innerHTML = '<p class="hint">还没有机器。填 ip:端口 添加，例如 10.0.0.1:8000</p>';
  }

  $('poolSummary').textContent =
    `${data.healthy}/${data.total} 台在线 · 总并发上限 ${data.total_capacity} · 当前在跑 ${data.in_flight}`;

  list.querySelectorAll('[data-del]').forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/endpoints?address=${encodeURIComponent(btn.dataset.del)}`, { method: 'DELETE' });
      toast('已删除');
      refreshEndpoints();
    };
  });
  list.querySelectorAll('[data-toggle]').forEach((btn) => {
    btn.onclick = async () => {
      const address = btn.dataset.toggle;
      const enabled = btn.textContent.trim() === '启用';
      await api('/api/endpoints/toggle', {
        method: 'POST', body: JSON.stringify({ address, enabled }),
      });
      refreshEndpoints();
    };
  });
  list.querySelectorAll('[data-cap]').forEach((input) => {
    input.onchange = async () => {
      await api('/api/endpoints/capacity', {
        method: 'POST',
        body: JSON.stringify({ address: input.dataset.cap, capacity: Number(input.value) }),
      });
      toast('并发已更新');
      refreshEndpoints();
    };
  });
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

$('btnAddEp').onclick = async () => {
  const address = $('epAddress').value.trim();
  if (!address) return toast('请填写 ip:端口', true);
  $('btnAddEp').disabled = true;
  try {
    const ep = await api('/api/endpoints', {
      method: 'POST',
      body: JSON.stringify({ address, capacity: Number($('epCapacity').value) || 4 }),
    });
    toast(ep.healthy ? `已添加 ${ep.host}（${ep.model}）` : `已添加，但探测失败：${ep.last_error}`, !ep.healthy);
    $('epAddress').value = '';
    refreshEndpoints();
  } catch (err) {
    toast(err.message, true);
  } finally {
    $('btnAddEp').disabled = false;
  }
};

$('epAddress').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btnAddEp').click(); });

$('btnProbe').onclick = async () => {
  $('btnProbe').disabled = true;
  $('btnProbe').textContent = '探测中…';
  try {
    refreshEndpoints(await api('/api/endpoints/probe', { method: 'POST' }));
    toast('探测完成');
  } catch (err) {
    toast(err.message, true);
  } finally {
    $('btnProbe').disabled = false;
    $('btnProbe').textContent = '重新探测';
  }
};

// ---------------------------------------------------------------- selection

function selectionBody() {
  return {
    min_chars: Number($('fMinChars').value) || 0,
    max_chars: Number($('fMaxChars').value) || 0,
    languages: [...$('fLangs').querySelectorAll('input:checked')].map((el) => el.value),
    limit: Number($('fLimit').value) || 0,
    stratify: $('fStratify').checked,
  };
}

$('btnPreview').onclick = async () => {
  $('btnPreview').disabled = true;
  $('previewBox').innerHTML = '<p class="hint">统计中…</p>';
  try {
    renderPreview(await api('/api/selection/preview', {
      method: 'POST', body: JSON.stringify(selectionBody()),
    }));
  } catch (err) {
    $('previewBox').innerHTML = `<p class="hint" style="color:var(--bad)">${escapeHtml(err.message)}</p>`;
  } finally {
    $('btnPreview').disabled = false;
  }
};

function renderPreview(data) {
  const stats = data.stats || {};
  const chars = stats.script_chars || {};
  const dur = stats.duration_sec || {};
  const rejected = Object.entries(data.rejected || {})
    .map(([k, v]) => `${k} ${v}`).join(' · ') || '无';
  const langs = Object.entries(stats.languages || {})
    .map(([k, v]) => `${k} ${v}`).join(' · ');

  $('previewBox').innerHTML = `
    <table>
      <tr><td class="k">入选</td><td class="v">${stats.count || 0} 条</td></tr>
      <tr><td class="k">排除</td><td class="v">${escapeHtml(rejected)}</td></tr>
      <tr><td class="k">语种</td><td class="v">${escapeHtml(langs)}</td></tr>
      <tr><td class="k">字数 min/p50/p90/max</td><td class="v">${chars.min}/${chars.p50}/${chars.p90}/${chars.max}</td></tr>
      <tr><td class="k">时长 p50/p90</td><td class="v">${Math.round(dur.p50 || 0)}s / ${Math.round(dur.p90 || 0)}s，共 ${dur.total_hours || 0} 小时</td></tr>
      <tr><td class="k">目录命名示例</td><td class="v">${escapeHtml((data.preview || []).slice(0, 3).map((i) => i.episode_id).join('  '))}</td></tr>
    </table>`;
}

$('btnCommit').onclick = async () => {
  if (!confirm('应用为待标注清单？这会覆盖 worklist.jsonl。')) return;
  $('btnCommit').disabled = true;
  try {
    const result = await api('/api/selection/commit', {
      method: 'POST', body: JSON.stringify(selectionBody()),
    });
    toast(`清单已生成：${result.selected} 条`);
    refreshWorklist();
  } catch (err) {
    toast(err.message, true);
  } finally {
    $('btnCommit').disabled = false;
  }
};

async function refreshWorklist() {
  try {
    const data = await api('/api/worklist');
    $('tableHint').textContent = data.exists
      ? `待标注清单：${data.count} 条（生成于 ${data.generated_at}）`
      : '尚未生成待标注清单 — 先在左侧设置筛选条件并「应用为待标注清单」。';
  } catch { /* status poll will surface real problems */ }
}

// ---------------------------------------------------------------- campaign

$('btnStart').onclick = async () => {
  const mode = $('selMode').value;
  const label = (modes.find((m) => m.id === mode) || {}).label || mode;
  if (!confirm(`启动批量标注？\n\n模式：${label}\nprofile：${$('selProfile').value}`)) return;
  $('btnStart').disabled = true;
  try {
    await api('/api/campaign/start', {
      method: 'POST',
      body: JSON.stringify({
        mode,
        profile: $('selProfile').value,
        max_steps: Number($('fMaxSteps').value) || 48,
        resume: $('fResume').checked,
        retry_failed: $('fRetry').checked,
        limit: Number($('fLimit').value) || 0,
      }),
    });
    toast('已启动');
  } catch (err) {
    toast(err.message, true);
  } finally {
    $('btnStart').disabled = false;
  }
};

for (const [id, path, message] of [
  ['btnPause', 'pause', '已暂停（在跑的条目会跑完）'],
  ['btnResume', 'resume', '已继续'],
  ['btnStop', 'stop', '停止中（在跑的条目会跑完）'],
]) {
  $(id).onclick = async () => {
    if (id === 'btnStop' && !confirm('确认停止？已完成的轨迹会保留，可续跑。')) return;
    try {
      await api(`/api/campaign/${path}`, { method: 'POST' });
      toast(message);
    } catch (err) {
      toast(err.message, true);
    }
  };
}

$('btnExport').onclick = async () => {
  $('btnExport').disabled = true;
  $('btnExport').textContent = '导出中…';
  try {
    const result = await api('/api/export', { method: 'POST' });
    const coverage = result.quality.template_coverage || {};
    const slots = result.quality.slot_quality || {};
    toast(`已导出：${result.aggregate.episode_count} 条 · 模版覆盖 ` +
          `${coverage.distinct_templates}/${coverage.registry_size} · ` +
          `照抄率 ${((slots.verbatim_rate || 0) * 100).toFixed(1)}%`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    $('btnExport').disabled = false;
    $('btnExport').textContent = '导出数据集';
  }
};

// ---------------------------------------------------------------- status

async function poll() {
  try {
    const status = await api('/api/status');
    lastStatus = status;
    renderStatus(status);
  } catch (err) {
    $('phaseBadge').textContent = '连接失败';
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

function renderStatus(status) {
  const summary = status.summary || {};
  const phase = summary.phase || 'idle';

  const badge = $('phaseBadge');
  badge.textContent = PHASE_ZH[phase] || phase;
  badge.className = `phase ${phase}`;

  $('outputRoot').textContent = status.output_root || '';
  $('barFill').style.width = `${summary.progress_pct || 0}%`;
  $('mTotal').textContent = summary.total || 0;
  $('mDone').textContent = summary.done || 0;
  $('mRunning').textContent = summary.running || 0;
  $('mPending').textContent = summary.pending || 0;
  $('mFailed').textContent = summary.failed || 0;
  $('mSkipped').textContent = summary.skipped || 0;
  $('mMean').textContent = duration(summary.mean_episode_seconds);
  $('mRate').textContent = summary.throughput_per_hour
    ? `${summary.throughput_per_hour}/小时` : '—';
  $('mEta').textContent = duration(summary.eta_seconds);
  $('mFinish').textContent = clockAfter(summary.eta_seconds);

  $('modeHint').textContent = status.mode_label
    ? `模式：${status.mode_label} · 进度 ${summary.settled || 0}/${summary.total || 0}` +
      ` (${(summary.progress_pct || 0).toFixed(1)}%) · 已运行 ${duration(summary.elapsed_seconds)}`
    : '尚未启动';

  const running = phase === 'running' || phase === 'paused' || phase === 'stopping';
  $('btnStart').disabled = running;
  $('btnPause').disabled = phase !== 'running';
  $('btnResume').disabled = phase !== 'paused';
  $('btnStop').disabled = !running;

  if (status.pool) refreshEndpointsLight(status.pool);
  renderEpisodes(status.episodes || []);
}

// The full endpoint re-render rebinds handlers and fights with typing in the
// capacity inputs; the poll only refreshes the numbers that change.
function refreshEndpointsLight(pool) {
  const rows = $('epList').querySelectorAll('.ep');
  if (rows.length !== (pool.endpoints || []).length) return refreshEndpoints(pool);
  (pool.endpoints || []).forEach((ep, i) => {
    const row = rows[i];
    const load = row.querySelector('.ep-load');
    if (load) load.textContent = `${ep.in_flight}/${ep.capacity}`;
    const dot = row.querySelector('.dot');
    if (dot) dot.className = `dot ${!ep.enabled ? 'off' : (ep.healthy ? 'up' : '')}`;
  });
  $('poolSummary').textContent =
    `${pool.healthy}/${pool.total} 台在线 · 总并发上限 ${pool.total_capacity} · 当前在跑 ${pool.in_flight}`;
}

function renderEpisodes(episodes) {
  const needle = $('epFilter').value.trim().toLowerCase();
  const wanted = $('epStatusFilter').value;

  let rows = episodes;
  if (wanted) rows = rows.filter((e) => e.status === wanted);
  if (needle) {
    rows = rows.filter((e) =>
      e.episode_id.toLowerCase().includes(needle) ||
      (e.asset_id || '').toLowerCase().includes(needle) ||
      (e.bv_id || '').toLowerCase().includes(needle));
  }

  // Running episodes first: they are the only ones changing.
  const rank = { running: 0, failed: 1, pending: 2, done: 3, skipped: 4 };
  rows = [...rows].sort((a, b) =>
    (rank[a.status] ?? 9) - (rank[b.status] ?? 9) || a.index - b.index);

  const shown = rows.slice(0, 300);
  const body = $('epTable').querySelector('tbody');
  body.innerHTML = shown.map((e) => `
    <tr>
      <td class="mono">${e.index}</td>
      <td class="mono" title="${escapeHtml(e.asset_id)}">${escapeHtml(e.episode_id)}</td>
      <td><span class="tag ${e.status}">${escapeHtml(e.status)}</span></td>
      <td>${escapeHtml(e.stage || '—')}</td>
      <td>
        <span class="mini-bar"><span class="mini-fill" style="width:${e.progress_pct || 0}%"></span></span>
        <span class="mono"> ${e.step}/${e.max_steps}</span>
      </td>
      <td class="mono">${escapeHtml((e.endpoint || '').replace(/^https?:\/\//, '').replace(/\/v1$/, ''))}</td>
      <td class="mono">${duration(e.elapsed_seconds)}</td>
      <td class="mono">${e.llm_calls || 0}</td>
      <td><button class="small" data-log="${escapeHtml(e.episode_id)}">查看</button></td>
    </tr>`).join('');

  body.querySelectorAll('[data-log]').forEach((btn) => {
    btn.onclick = () => showLog(btn.dataset.log);
  });

  if (lastStatus && lastStatus.summary && lastStatus.summary.total) {
    $('tableHint').textContent =
      `显示 ${shown.length} / ${rows.length} 条（共 ${episodes.length}）` +
      (rows.length > 300 ? ' — 已截断，请用搜索或状态筛选' : '');
  }
}

$('epFilter').oninput = () => lastStatus && renderEpisodes(lastStatus.episodes || []);
$('epStatusFilter').onchange = () => lastStatus && renderEpisodes(lastStatus.episodes || []);

async function showLog(episodeId) {
  $('logTitle').textContent = episodeId;
  $('logBody').textContent = '加载中…';
  $('logModal').classList.remove('hidden');
  try {
    const data = await api(`/api/episode/${encodeURIComponent(episodeId)}/log?tail=400`);
    $('logBody').textContent = data.lines.join('\n') || '(日志为空)';
    $('logBody').scrollTop = $('logBody').scrollHeight;
  } catch (err) {
    $('logBody').textContent = err.message;
  }
}
$('btnCloseLog').onclick = () => $('logModal').classList.add('hidden');
$('logModal').onclick = (e) => { if (e.target === $('logModal')) $('logModal').classList.add('hidden'); };

// ---------------------------------------------------------------- bootstrap

async function init() {
  try {
    modes = (await api('/api/modes')).modes || [];
    $('selMode').innerHTML = modes
      .map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join('');
    $('selMode').onchange = renderModeSwitches;
    renderModeSwitches();
  } catch (err) {
    toast(err.message, true);
  }
  await refreshEndpoints();
  await refreshWorklist();
  poll();
}

function renderModeSwitches() {
  const mode = modes.find((m) => m.id === $('selMode').value);
  if (!mode) return;
  const on = Object.entries(mode.tool_switches).filter(([, v]) => v).map(([k]) => k);
  const off = Object.entries(mode.tool_switches).filter(([, v]) => !v).map(([k]) => k);
  $('modeSwitches').innerHTML =
    `开启：${on.length ? escapeHtml(on.join(', ')) : '（无）'}<br />关闭：${off.length ? escapeHtml(off.join(', ')) : '（无）'}`;
}

init();
