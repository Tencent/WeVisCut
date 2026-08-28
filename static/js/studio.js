/* Studio shell: config bootstrap, run lifecycle, WebSocket event routing.
 *
 * One event renderer serves both generation runs and post-edit regeneration
 * jobs, because the backend emits the same vocabulary for both.
 */
(function () {
  'use strict';

  const PHASE_LABELS = {
    agent_loop: 'Agent 自主执行',
    segmentation: '文稿分段',
    shot_planning: '镜头规划',
    narration: '配音生成',
    footage: '素材检索',
    spotlight: '重点镜头装饰',
    bgm: '配乐',
    render: '渲染',
    final_stitch: '拼接收尾',
    rerender: '重新渲染',
    done: '完成',
  };

  const TOOL_LABELS = {
    Tool_SegmentArticle: '文稿分段',
    Tool_PlanShots: '镜头规划',
    Tool_GenerateNarration: '生成配音',
    Tool_RetrieveVisuals: '检索画面素材',
    Tool_ScoreSeverity: '重点镜头打分',
    Tool_DecorateTimeline: '模板装饰',
    Tool_SearchBGM: '检索配乐',
    Tool_RenderShots: '渲染镜头',
    Tool_StitchVideo: '拼接成片',
    Tool_Reflect: '自我复核',
    Tool_Respond: '状态回填',
  };

  const RUNNING_STATUSES = new Set(['running', 'paused', 'queued', 'cancelling']);

  function formatBytes(n) {
    if (!n) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let v = n;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `，释放 ${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
  }

  // ---------------- tiny fetch wrapper ----------------
  const api = {
    async _json(res) {
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (e) { /* non-JSON error body */ }
      if (!res.ok) {
        const detail = (data && (data.detail || data.message)) || text || res.statusText;
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      return data;
    },
    async get(path, params) {
      const url = new URL(path, window.location.origin);
      Object.entries(params || {}).forEach(([k, v]) => url.searchParams.set(k, v));
      return this._json(await fetch(url.toString()));
    },
    async post(path, body) {
      return this._json(await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body === null || body === undefined ? null : JSON.stringify(body),
      }));
    },
    async del(path) {
      return this._json(await fetch(path, { method: 'DELETE' }));
    },
  };

  // ---------------- toasts ----------------
  function toast(text, kind) {
    const stack = document.getElementById('toasts');
    const el = document.createElement('div');
    el.className = 'toast' + (kind ? ' ' + kind : '');
    el.textContent = text;
    stack.appendChild(el);
    setTimeout(() => el.remove(), kind === 'err' ? 7000 : 4000);
  }

  // ---------------- main ----------------
  class Studio {
    constructor() {
      this.cfg = null;
      this.runId = null;
      this.ws = null;
      this.wsRetry = 0;
      this.running = false;
      this.paused = false;
      this.timerHandle = null;
      this.startedAt = 0;
      this.logs = { console: [], raw: [] };
      this.logView = 'console';
      this.currentProjectDir = '';

      this.el = {
        script: document.getElementById('scriptInput'),
        scriptCount: document.getElementById('scriptCount'),
        startBtn: document.getElementById('startBtn'),
        pauseBtn: document.getElementById('pauseBtn'),
        cancelBtn: document.getElementById('cancelBtn'),
        startHint: document.getElementById('startHint'),
        runList: document.getElementById('runList'),
        refreshRuns: document.getElementById('refreshRuns'),
        projectList: document.getElementById('projectList'),
        modelDot: document.getElementById('modelDot'),
        modelText: document.getElementById('modelText'),
        modePill: document.getElementById('modePill'),
        runTimer: document.getElementById('runTimer'),
        warningsBtn: document.getElementById('warningsBtn'),
        warningsCount: document.getElementById('warningsCount'),
        player: document.getElementById('player'),
        playerEmpty: document.getElementById('playerEmpty'),
        phaseLabel: document.getElementById('phaseLabel'),
        phaseNote: document.getElementById('phaseNote'),
        overallBar: document.getElementById('overallBar'),
        overallPct: document.getElementById('overallPct'),
        statDuration: document.getElementById('statDuration'),
        statShots: document.getElementById('statShots'),
        statSpotlight: document.getElementById('statSpotlight'),
        statSteps: document.getElementById('statSteps'),
        trace: document.getElementById('trace'),
        autonomyNote: document.getElementById('autonomyNote'),
        consoleToggle: document.getElementById('consoleToggle'),
        consoleCaret: document.getElementById('consoleCaret'),
        consoleBody: document.getElementById('consoleBody'),
        consoleCount: document.getElementById('consoleCount'),
        tabEditor: document.getElementById('tabEditor'),
        tabChat: document.getElementById('tabChat'),
      };

      this.timeline = new window.AICutTimeline({
        onSelect: (track, clip) => this.onClipSelected(track, clip),
      });
      this.rail = new window.AICutRail();
      this.editor = new window.AICutEditor({
        api: api, toast: toast, timeline: this.timeline,
        onProjectUpdate: (p, opts) => this.onProjectUpdate(p, opts),
        onRegenStarted: (jobId) => this.attachRun(jobId),
      });
      this.chat = new window.AICutChat({
        api: api, toast: toast, editor: this.editor,
        onRegenStarted: (jobId) => this.attachRun(jobId),
      });

      this._bind();
      this.boot();
    }

    _bind() {
      this.el.script.addEventListener('input', () => {
        this.el.scriptCount.textContent = this.el.script.value.length + ' 字';
        this._syncStartButton();
      });
      this.el.startBtn.onclick = () => this.start();
      this.el.pauseBtn.onclick = () => this.togglePause();
      this.el.cancelBtn.onclick = () => this.cancel();
      this.el.refreshRuns.onclick = () => { this.loadRuns(); this.loadProjects(); };

      this.el.consoleToggle.onclick = () => {
        const hidden = this.el.consoleBody.hidden;
        this.el.consoleBody.hidden = !hidden;
        this.el.consoleCaret.classList.toggle('open', hidden);
      };
      document.querySelectorAll('[data-log]').forEach((btn) => {
        btn.onclick = () => {
          document.querySelectorAll('[data-log]').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          this.logView = btn.dataset.log;
          this._renderLogs();
        };
      });
      document.querySelectorAll('#tabs .tab').forEach((btn) => {
        btn.onclick = () => this.switchTab(btn.dataset.tab);
      });

      window.addEventListener('beforeunload', () => { if (this.ws) this.ws.close(); });
    }

    switchTab(name) {
      document.querySelectorAll('#tabs .tab').forEach((b) => {
        b.classList.toggle('active', b.dataset.tab === name);
      });
      document.querySelectorAll('.tab-panel').forEach((p) => {
        p.classList.toggle('active', p.dataset.panel === name);
      });
    }

    // ---------------- boot ----------------
    async boot() {
      try {
        this.cfg = await api.get('/api/config');
      } catch (err) {
        toast('无法读取配置: ' + err.message, 'err');
        return;
      }

      document.title = this.cfg.name || 'AICut Studio';
      window.AICutOpLabels = (this.cfg.editor && this.cfg.editor.op_labels) || {};

      this.timeline.setTracks(this.cfg.tracks || []);
      this.editor.configure(this.cfg);

      const feat = this.cfg.features || {};
      if (!feat.editor) {
        this.el.tabEditor.disabled = true;
        this.el.tabEditor.title = '配置中已关闭编辑功能';
      }
      if (!feat.chat_edit) {
        this.el.tabChat.disabled = true;
        this.el.tabChat.title = '配置中已关闭对话编辑 (chat_edit.enabled=false)';
      }

      const off = Object.entries(feat.services || {})
        .filter(([, on]) => !on).map(([k]) => k);
      const autonomous = this.cfg.agent && this.cfg.agent.fully_autonomous;
      this.el.autonomyNote.textContent = autonomous
        ? `完全自主模式：不改写工具选择、不强制收尾，非法调用只作为 observation 回灌。步数上限 ${this.cfg.agent.max_steps}。`
        : '兼容模式：保留参考实现的路由校正。';
      if (off.length) {
        this.el.autonomyNote.textContent += ` 已关闭的服务: ${off.join(', ')}。`;
      }

      const warnings = this.cfg.warnings || [];
      if (warnings.length) {
        this.el.warningsBtn.hidden = false;
        this.el.warningsCount.textContent = warnings.length;
        this.el.warningsBtn.onclick = () => alert('配置提醒:\n\n' + warnings.join('\n'));
      }

      this.pollStatus();
      this.statusHandle = setInterval(() => this.pollStatus(), 4000);
      this.loadRuns();
      this.loadProjects();
    }

    async pollStatus() {
      let st;
      try {
        st = await api.get('/api/status');
      } catch (err) {
        this._setModel('err', '服务不可用');
        return;
      }
      const m = st.model || {};
      const pill = document.getElementById('modelPill');
      if (m.loaded) {
        this._setModel('ok', '模型已就绪');
        pill.title = [m.base_model, m.adapter && ('adapter: ' + m.adapter), m.backend]
          .filter(Boolean).join('\n');
      } else if (m.load_error) {
        this._setModel('err', '模型未就绪');
        pill.title = m.load_error;
      } else if (m.skipped) {
        this._setModel('err', '模型未就绪');
        pill.title = '服务以 --no-model 启动，界面可浏览，但无法生成或重渲染';
      } else {
        this._setModel('loading', '模型未就绪');
        pill.title = '首次加载权重通常需要 30-60 秒';
      }
      this.el.modePill.textContent = st.agent_mode || '—';
      this._syncStartButton();
    }

    _setModel(state, text) {
      this.el.modelDot.className = 'dot ' + state;
      this.el.modelText.textContent = text;
      this.modelReady = state === 'ok';
      this.modelBlockedReason = state === 'ok' ? '' : text;
    }

    _syncStartButton() {
      const hasText = this.el.script.value.trim().length > 0;
      const ok = this.modelReady && hasText && !this.running;
      this.el.startBtn.disabled = !ok;
      if (this.running) this.el.startHint.textContent = '正在生成中…';
      else if (!this.modelReady) {
        this.el.startHint.textContent = this.modelBlockedReason || '等待模型加载完成…';
      } else if (!hasText) this.el.startHint.textContent = '请粘贴口播稿';
      else this.el.startHint.textContent = '模型将自主决定每一步工具调用';
    }

    // ---------------- runs ----------------
    async start() {
      const text = this.el.script.value.trim();
      if (!text) return;
      this._resetView();
      try {
        const data = await api.post('/api/runs', { input_text: text });
        this.attachRun(data.run_id);
        this.loadRuns();
      } catch (err) {
        toast('启动失败: ' + err.message, 'err');
      }
    }

    attachRun(runId) {
      this.runId = runId;
      this.running = true;
      this.paused = false;
      this.startedAt = Date.now();
      this.el.pauseBtn.hidden = false;
      this.el.cancelBtn.hidden = false;
      this.el.pauseBtn.textContent = '暂停';
      this.el.runTimer.hidden = false;
      this._startTimer();
      this._syncStartButton();
      this.switchTab('agent');
      this._connect(runId);
    }

    async togglePause() {
      if (!this.runId) return;
      try {
        if (this.paused) {
          await api.post(`/api/runs/${this.runId}/resume`, {});
          this.paused = false;
          this.el.pauseBtn.textContent = '暂停';
        } else {
          await api.post(`/api/runs/${this.runId}/pause`, {});
          this.paused = true;
          this.el.pauseBtn.textContent = '继续';
        }
      } catch (err) {
        toast(err.message, 'err');
      }
    }

    async cancel() {
      if (!this.runId) return;
      if (!confirm('取消当前生成？当前步骤结束后停止。')) return;
      try {
        await api.post(`/api/runs/${this.runId}/cancel`, {});
      } catch (err) {
        toast(err.message, 'err');
      }
    }

    _resetView() {
      this.logs = { console: [], raw: [] };
      this._renderLogs();
      this.el.trace.innerHTML = '';
      this.el.player.removeAttribute('src');
      this.el.playerEmpty.hidden = false;
      this.el.overallBar.style.width = '0%';
      this.el.overallPct.textContent = '0%';
      this.timeline.update({ track_order: this.timeline.order, clips: {}, total_duration: 0, stats: {} });
    }

    _startTimer() {
      clearInterval(this.timerHandle);
      this.timerHandle = setInterval(() => {
        const s = Math.floor((Date.now() - this.startedAt) / 1000);
        this.el.runTimer.textContent =
          String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
      }, 1000);
    }

    // ---------------- websocket ----------------
    _connect(runId) {
      if (this.ws) { this.ws.onclose = null; this.ws.close(); }
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/${runId}`);
      this.ws = ws;

      ws.onopen = () => {
        this.wsRetry = 0;
        this.pingHandle = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 25000);
      };
      ws.onmessage = (evt) => {
        let data;
        try { data = JSON.parse(evt.data); } catch (e) { return; }
        this.onEvent(data);
      };
      ws.onclose = () => {
        clearInterval(this.pingHandle);
        // A run outlives its socket; reconnect and replay rather than losing the view.
        if (this.running && this.runId === runId && this.wsRetry < 5) {
          this.wsRetry += 1;
          setTimeout(() => this._connect(runId), 1000 * this.wsRetry);
        }
      };
    }

    onEvent(ev) {
      switch (ev.type) {
        case 'replay_start':
          this._log('console', 'system', `回放 ${ev.event_count} 条历史事件`);
          break;

        case 'progress':
          this._onProgress(ev.progress);
          break;

        case 'stage_update':
          this.el.phaseLabel.textContent = PHASE_LABELS[ev.stage] || ev.stage || '';
          if (ev.message) this.el.phaseNote.textContent = ev.message;
          if (typeof ev.progress_pct === 'number') this._setOverall(ev.progress_pct);
          break;

        case 'thinking':
          this._onThinking(ev);
          break;

        case 'tool_call':
          this._traceEvent('call', ev);
          break;

        case 'tool_result':
          this._traceEvent(ev.status === 'blocked_precondition' ? 'blocked' : 'result', ev);
          break;

        case 'console_log':
          this._log('console', ev.level || 'info', ev.text || '', ev.timestamp);
          break;

        case 'raw_output':
          this._log('raw', 'info', `[step ${ev.step || '-'}] ${ev.text || ''}`, ev.timestamp);
          break;

        case 'error':
          this._log('console', 'stderr', `[${ev.stage || '?'}] ${ev.message || ''}`);
          this._traceEvent('error', ev);
          break;

        case 'preview_update':
          if (ev.preview && ev.preview.final_video_path) this._setVideo(ev.preview.final_video_path);
          break;

        case 'pipeline_complete':
          if (ev.results) {
            this.el.statSteps.textContent = ev.results.steps_used || 0;
            if (ev.results.final_video_path) this._setVideo(ev.results.final_video_path, { reload: true });
          }
          break;

        case 'run_paused':
          this._log('console', 'warn', '已暂停');
          break;

        case 'run_resumed':
          this._log('console', 'system', '已继续');
          break;

        case 'run_complete':
          this._onComplete(ev);
          break;

        default:
          break;
      }
    }

    _onProgress(p) {
      if (!p) return;
      this.timeline.update(p);
      this.rail.update(p);
      this._setOverall(p.overall_pct || 0);
      this.el.statDuration.textContent = window.AICutFmtTime(p.total_duration || 0);
      const stats = p.stats || {};
      this.el.statShots.textContent =
        `${(stats.shots || {}).done || 0}/${(stats.shots || {}).total || 0}`;
      this.el.statSpotlight.textContent =
        `${(stats.spotlight || {}).done || 0}/${(stats.spotlight || {}).total || 0}`;
      if (p.final_video_path) this._setVideo(p.final_video_path);
    }

    _setOverall(pct) {
      const v = Math.max(0, Math.min(100, Math.round(pct)));
      this.el.overallBar.style.width = v + '%';
      this.el.overallPct.textContent = v + '%';
    }

    _onThinking(ev) {
      if (typeof ev.step === 'number') this.el.statSteps.textContent = ev.step;
      if (ev.raw_output) {
        this._log('raw', 'info', `[step ${ev.step || '-'}] ${ev.raw_output}`);
      }
      if (ev.thinking || ev.summary) this._traceEvent('think', ev);
      if (ev.parse_ok === false) {
        this._log('console', 'warn', `step ${ev.step}: 输出解析失败`);
      }
    }

    _traceEvent(kind, ev) {
      const el = document.createElement('div');
      el.className = 'ev ' + kind;

      const head = document.createElement('div');
      head.className = 'ev-head';
      if (ev.step) {
        const st = document.createElement('span');
        st.className = 'ev-step';
        st.textContent = '#' + ev.step;
        head.appendChild(st);
      }
      const tool = document.createElement('span');
      tool.className = 'ev-tool';
      const name = ev.tool_name || ev.stage || '';
      tool.textContent = TOOL_LABELS[name] || name;
      head.appendChild(tool);

      if (kind === 'result' || kind === 'blocked' || kind === 'error') {
        const status = document.createElement('span');
        const s = ev.status || (kind === 'error' ? 'error' : '');
        status.className = 'ev-status ' +
          (kind === 'blocked' ? 'blocked' : (kind === 'error' ? 'err' : 'ok'));
        status.textContent = kind === 'error' ? '失败' : (s || 'ok');
        head.appendChild(status);
      }
      el.appendChild(head);

      const bodyText =
        kind === 'think' ? (ev.thinking || ev.summary || '') :
        kind === 'error' ? (ev.message || '') :
        kind === 'blocked' ? ('前置条件不足: ' + (ev.observation && ev.observation.missing
            ? ev.observation.missing.join(', ')
            : (ev.precondition_missing || []).join(', '))) : '';
      if (bodyText) {
        const body = document.createElement('div');
        body.className = 'ev-body';
        body.textContent = bodyText.length > 600 ? bodyText.slice(0, 600) + '…' : bodyText;
        el.appendChild(body);
      }

      if (kind === 'call' && ev.arguments && Object.keys(ev.arguments).length) {
        const args = document.createElement('div');
        args.className = 'ev-args';
        args.textContent = JSON.stringify(ev.arguments);
        args.title = JSON.stringify(ev.arguments, null, 2);
        el.appendChild(args);
      }

      const empty = this.el.trace.querySelector('.empty');
      if (empty) empty.remove();
      this.el.trace.appendChild(el);
      // Only follow the tail when the user has not scrolled up to read history.
      const panel = this.el.trace.parentElement;
      if (panel.scrollTop + panel.clientHeight > panel.scrollHeight - 120) {
        panel.scrollTop = panel.scrollHeight;
      }
    }

    async _onComplete(ev) {
      this.running = false;
      this.paused = false;
      clearInterval(this.timerHandle);
      this.el.pauseBtn.hidden = true;
      this.el.cancelBtn.hidden = true;
      this._syncStartButton();

      const label = {
        completed: '已完成',
        completed_with_errors: '完成（有错误）',
        cancelled: '已取消',
        error: '失败',
      }[ev.status] || ev.status;
      this.el.phaseLabel.textContent = label;
      if (ev.status === 'completed') this._setOverall(100);

      const result = ev.result || {};
      if (result.final_video_path) this._setVideo(result.final_video_path, { reload: true });
      if (ev.progress) this._onProgress(ev.progress);

      const isChatRegen = (this.runId || '').startsWith('chat_');
      if (isChatRegen) {
        toast(
          ev.status === 'completed'
            ? '对话修改已完成，新视频已生成'
            : '重渲染' + label,
          ev.status === 'completed' ? 'ok' : 'warn',
        );
      } else {
        toast('生成结束: ' + label, ev.status === 'completed' ? 'ok' : 'warn');
      }
      this.loadRuns();
      this.loadProjects();

      const dir = ev.project_dir || result.project_dir || '';
      if (dir && (this.cfg.features || {}).editor) {
        await this.openProject(dir);
        this.el.phaseNote.textContent = isChatRegen
          ? '修改已渲染完毕，可在下方预览最新成片'
          : '可在右侧「编辑」修改文字与素材，或用「对话」直接说明想改什么';
      }
    }

    _setVideo(path, opts) {
      if (!path) return;
      // The final cut is always written to the same filename, so after a chat
      // edit the URL is byte-identical to the one already loaded. Without a
      // cache-buster the browser keeps serving the previous file from cache and
      // the edited shot appears unchanged even though the new video is on disk.
      // `reload` is passed only by run/edit completion handlers — progress ticks
      // repeat the same path many times and must not restart playback.
      const base = '/api/media?path=' + encodeURIComponent(path);
      const reload = !!(opts && opts.reload);
      if (this._videoBase === base && !reload) return;
      this._videoBase = base;
      const src = reload ? base + '&v=' + Date.now() : base;
      const wasPlaying = !this.el.player.paused && !this.el.player.ended;
      this.el.player.setAttribute('src', src);
      // Drop the stale buffer so the element refetches instead of reusing cache.
      this.el.player.load();
      if (wasPlaying) this.el.player.play().catch(function () {});
      this.el.playerEmpty.hidden = true;
    }

    // ---------------- logs ----------------
    _log(view, level, text, timestamp) {
      const entry = {
        level: level,
        text: text,
        time: timestamp || new Date().toLocaleTimeString('zh-CN', { hour12: false }),
      };
      this.logs[view].push(entry);
      if (this.logs[view].length > 4000) this.logs[view].shift();
      if (this.logView === view) this._appendLogLine(entry);
      this.el.consoleCount.textContent = this.logs[this.logView].length;
    }

    _appendLogLine(entry) {
      const body = this.el.consoleBody;
      const atBottom = body.scrollTop + body.clientHeight > body.scrollHeight - 40;
      const line = document.createElement('div');
      line.className = 'log-line ' + entry.level;
      const t = document.createElement('span');
      t.className = 'log-time';
      t.textContent = entry.time;
      const tx = document.createElement('span');
      tx.className = 'log-text';
      tx.textContent = entry.text;
      line.appendChild(t);
      line.appendChild(tx);
      body.appendChild(line);
      if (atBottom) body.scrollTop = body.scrollHeight;
    }

    _renderLogs() {
      this.el.consoleBody.innerHTML = '';
      this.logs[this.logView].forEach((e) => this._appendLogLine(e));
      this.el.consoleCount.textContent = this.logs[this.logView].length;
      this.el.consoleBody.scrollTop = this.el.consoleBody.scrollHeight;
    }

    // ---------------- history & projects ----------------
    async loadRuns() {
      try {
        const data = await api.get('/api/runs');
        const list = this.el.runList;
        list.innerHTML = '';
        if (!data.runs.length) {
          list.innerHTML = '<p class="empty">暂无记录</p>';
          return;
        }
        data.runs.forEach((r) => {
          const item = document.createElement('div');
          item.className = 'list-item' + (r.id === this.runId ? ' active' : '');
          item.innerHTML =
            '<div class="li-title"></div>' +
            '<div class="li-meta">' +
              '<span class="li-status"></span><span class="li-pct"></span><span class="li-time"></span>' +
            '</div>';
          item.querySelector('.li-title').textContent = r.title;
          const st = item.querySelector('.li-status');
          st.textContent = r.status;
          st.classList.add(r.status);
          item.querySelector('.li-pct').textContent = (r.overall_pct || 0) + '%';
          item.querySelector('.li-time').textContent = Math.round(r.elapsed_seconds) + 's';
          item.onclick = () => this.openRun(r);
          // A running job cannot be removed; cancel it first.
          if (!RUNNING_STATUSES.has(r.status)) {
            item.appendChild(this._deleteButton('删除这条记录', () => this.deleteRun(r)));
          }
          list.appendChild(item);
        });
      } catch (err) { /* transient */ }
    }

    async openRun(summary) {
      try {
        const detail = await api.get('/api/runs/' + summary.id);
        this.el.script.value = detail.input_text || '';
        this.el.scriptCount.textContent = this.el.script.value.length + ' 字';
        this._resetView();
        this.runId = summary.id;
        if (detail.progress) this._onProgress(detail.progress);
        this.running = detail.status === 'running' || detail.status === 'paused';
        if (this.running) {
          this.startedAt = Date.now() - detail.elapsed_seconds * 1000;
          this._startTimer();
          this.el.pauseBtn.hidden = false;
          this.el.cancelBtn.hidden = false;
        }
        this._connect(summary.id);
        this._syncStartButton();
        this.loadRuns();
        if (summary.project_dir) await this.openProject(summary.project_dir);
      } catch (err) {
        toast('无法打开该记录: ' + err.message, 'err');
      }
    }

    async loadProjects() {
      try {
        const data = await api.get('/api/projects');
        const list = this.el.projectList;
        list.innerHTML = '';
        if (!data.projects.length) {
          list.innerHTML = '<p class="empty">暂无成片</p>';
          return;
        }
        data.projects.slice(0, 30).forEach((p) => {
          const item = document.createElement('div');
          item.className = 'list-item' + (p.project_dir === this.currentProjectDir ? ' active' : '');
          item.innerHTML = '<div class="li-title"></div><div class="li-meta"></div>';
          item.querySelector('.li-title').textContent = p.title || p.project_id;
          item.querySelector('.li-meta').textContent =
            [(p.shot_count || 0) + ' 镜', p.has_video ? '有成片' : '无成片',
             (p.modified || '').replace('T', ' ').slice(5, 16)].join(' · ');
          item.onclick = () => this.openProject(p.project_dir);
          item.appendChild(this._deleteButton('删除该成片', () => this.deleteProject(p)));
          list.appendChild(item);
        });
      } catch (err) { /* transient */ }
    }

    /** A hover-revealed ✕ on a list row. Stops propagation so it never opens
     *  the very item the user is trying to remove. */
    _deleteButton(title, onDelete) {
      const btn = document.createElement('button');
      btn.className = 'li-del';
      btn.title = title;
      btn.textContent = '✕';
      btn.onclick = (ev) => {
        if (ev && ev.stopPropagation) ev.stopPropagation();
        onDelete();
      };
      return btn;
    }

    async deleteProject(p) {
      const name = p.title || p.project_id;
      const warn = p.has_video ? '，包括已渲染的成片文件' : '';
      if (!confirm(`删除「${name}」？\n\n工程目录会被永久删除${warn}，无法恢复。`)) return;
      try {
        let data;
        try {
          data = await api.del('/api/project?dir=' + encodeURIComponent(p.project_dir));
        } catch (delErr) {
          if (delErr.message && delErr.message.includes('Method Not Allowed')) {
            data = await api.post('/api/project/delete?dir=' + encodeURIComponent(p.project_dir), {});
          } else {
            throw delErr;
          }
        }
        if (this.currentProjectDir === p.project_dir) this._closeProject();
        toast(`已删除「${name}」${formatBytes(data.freed_bytes)}`, 'ok');
        this.loadProjects();
      } catch (err) {
        toast('删除失败: ' + err.message, 'err');
      }
    }

    async deleteRun(r) {
      if (!confirm(`删除记录「${r.title}」？\n\n只删除这条运行记录，不影响已生成的成片。`)) return;
      try {
        try {
          await api.del('/api/runs/' + r.id);
        } catch (delErr) {
          if (delErr.message && delErr.message.includes('Method Not Allowed')) {
            await api.post('/api/runs/' + r.id + '/delete', {});
          } else {
            throw delErr;
          }
        }
        if (this.runId === r.id) this.runId = null;
        this.loadRuns();
      } catch (err) {
        toast('删除失败: ' + err.message, 'err');
      }
    }

    /** Return the workspace to its "nothing selected" state. */
    _closeProject() {
      this.currentProjectDir = '';
      this.editor.close();
      this.chat.clear();
      this.timeline.update({ track_order: this.timeline.order, clips: {}, stats: {}, total_duration: 0 });
      this.rail.update({ track_order: this.timeline.order, labels: {}, stats: {} });
      this.el.player.removeAttribute('src');
      this.el.playerEmpty.hidden = false;
      this.switchTab('agent');
    }

    async openProject(projectDir) {
      if (!(this.cfg.features || {}).editor) {
        toast('编辑功能已在配置中关闭', 'warn');
        return;
      }
      this.currentProjectDir = projectDir;
      await this.editor.open(projectDir);
      if ((this.cfg.features || {}).chat_edit) this.chat.setProject(projectDir);
      this.loadProjects();
    }

    onProjectUpdate(payload, opts) {
      // Reflect edited state on the timeline without waiting for a re-render.
      if (payload.progress) {
        this.timeline.update(payload.progress);
        this.rail.update(payload.progress);
      }
      if (!(opts && opts.skipVideo) && payload.final_video_path) {
        this._setVideo(payload.final_video_path, { reload: true });
      }
      this.currentProjectDir = payload.project_dir || this.currentProjectDir;
    }

    onClipSelected(track, clip) {
      if (!clip.shot_id) {
        // Segment and BGM lanes have no per-shot editor; jump the player instead.
        if (this.el.player.getAttribute('src')) this.el.player.currentTime = clip.start || 0;
        return;
      }
      this.timeline.select(clip.shot_id);
      if (this.el.player.getAttribute('src')) this.el.player.currentTime = clip.start || 0;
      if (!this.currentProjectDir) return;
      this.switchTab('editor');
      this.editor.inspect(clip);
    }
  }

  document.addEventListener('DOMContentLoaded', () => { window.studio = new Studio(); });
})();
