/* Manual editing panel.
 *
 * Builds the same operation objects the chat agent emits and posts them to the
 * same endpoint, so a hand edit and a conversational edit are indistinguishable
 * to the backend and validated identically.
 */
(function () {
  'use strict';

  class Editor {
    constructor(ctx) {
      this.ctx = ctx;              // { api, toast, timeline, onProjectUpdate }
      this.projectDir = '';
      this.project = null;
      this.selected = null;        // currently inspected shot
      this.enabledOps = {};
      this.autoRestitch = true;

      this.el = {
        empty: document.getElementById('editorEmpty'),
        body: document.getElementById('editorBody'),
        clipId: document.getElementById('editClipId'),
        clipTime: document.getElementById('editClipTime'),
        narration: document.getElementById('editNarration'),
        subtitle: document.getElementById('editSubtitle'),
        footageQuery: document.getElementById('editFootageQuery'),
        footagePath: document.getElementById('editFootagePath'),
        slots: document.getElementById('editSlots'),
        templateIds: document.getElementById('editTemplateIds'),
        structure: document.getElementById('editStructure'),
        motion: document.getElementById('editMotion'),
        visual: document.getElementById('editVisual'),
        bgmQuery: document.getElementById('editBgmQuery'),
        bgmVolume: document.getElementById('editBgmVolume'),
        bgmVolumeVal: document.getElementById('editBgmVolumeVal'),
        fieldSlots: document.getElementById('fieldSlots'),
        fieldTemplate: document.getElementById('fieldTemplate'),
        fieldNarration: document.getElementById('fieldNarration'),
        fieldSubtitle: document.getElementById('fieldSubtitle'),
        fieldFootage: document.getElementById('fieldFootage'),
        fieldDelete: document.getElementById('fieldDelete'),
        pendingBox: document.getElementById('pendingBox'),
        pendingList: document.getElementById('pendingList'),
        regenBtn: document.getElementById('regenBtn'),
        versionList: document.getElementById('versionList'),
      };

      this._bind();
    }

    configure(cfg) {
      this.enabledOps = (cfg.features && cfg.features.manual_ops) || {};
      this.autoRestitch = !!(cfg.editor && cfg.editor.auto_restitch);
      // Hide controls for operations the config disabled, rather than letting
      // the user click something the server will reject.
      const show = (el, flag) => { if (el) el.hidden = !this._enabled(flag); };
      show(this.el.fieldNarration, 'edit_narration');
      show(this.el.fieldSubtitle, 'edit_subtitle');
      show(this.el.fieldFootage, 'replace_footage');
      show(this.el.fieldDelete, 'delete_shot');
    }

    _enabled(flag) {
      return this.enabledOps[flag] !== false;
    }

    _bind() {
      document.querySelectorAll('[data-op]').forEach((btn) => {
        btn.onclick = () => this._submit(btn.dataset.op);
      });
      if (this.el.bgmVolume) {
        this.el.bgmVolume.oninput = () => {
          this.el.bgmVolumeVal.textContent = Number(this.el.bgmVolume.value).toFixed(2);
        };
      }
      if (this.el.regenBtn) this.el.regenBtn.onclick = () => this.regenerate();
    }

    // ---------------- project ----------------
    async open(projectDir) {
      this.projectDir = projectDir;
      try {
        const data = await this.ctx.api.get('/api/project', { dir: projectDir });
        this.setProject(data);
        this.ctx.toast('已载入成片工程，可点击时间轴片段编辑', 'ok');
      } catch (err) {
        this.ctx.toast('载入工程失败: ' + err.message, 'err');
      }
    }

    setProject(data, opts) {
      this.project = data;
      this.projectDir = data.project_dir || this.projectDir;
      this.el.empty.hidden = true;
      this.el.body.hidden = false;
      this._renderPending();
      this._renderVersions();
      if (this.ctx.onProjectUpdate) this.ctx.onProjectUpdate(data, opts);
      if (this.selected) {
        // re-resolve the inspected shot against refreshed state
        const shots = (data.progress.clips.shots) || [];
        const again = shots.find((s) => s.shot_id === this.selected.shot_id);
        if (again) this.inspect(again);
        else this.clearSelection();
      }
    }

    clearSelection() {
      this.selected = null;
      this.el.clipId.textContent = '—';
      this.el.clipTime.textContent = '';
    }

    /** Forget the current project, e.g. after it was deleted. */
    close() {
      this.projectDir = '';
      this.project = null;
      this.clearSelection();
      this.el.body.hidden = true;
      this.el.empty.hidden = false;
      this.el.pendingBox.hidden = true;
      this.el.regenBtn.disabled = true;
      this.el.versionList.innerHTML = '<p class="empty">暂无历史版本</p>';
    }

    /** Called when any clip on the timeline is clicked. */
    inspect(clip) {
      if (!clip || !clip.shot_id) return;
      const shotId = clip.shot_id;
      const digest = (this.project && this.project.digest) || {};
      const row = (digest.shots || []).find((s) => s.shot_id === shotId) || {};

      this.selected = { shot_id: shotId, row: row };
      this.el.clipId.textContent = shotId;
      this.el.clipTime.textContent =
        window.AICutFmtTime(row.start || clip.start) + ' – ' +
        window.AICutFmtTime(row.end || clip.end) +
        (row.is_spotlight ? '  · 重点镜头' : '');

      const text = row.text || '';
      this.el.narration.value = text;
      this.el.subtitle.value = text;
      this.el.footageQuery.value = '';
      this.el.footagePath.value = '';

      const isSpot = !!row.is_spotlight;
      this.el.fieldSlots.hidden = !(isSpot && this._enabled('edit_spotlight_slots'));
      this.el.fieldTemplate.hidden = !(isSpot && this._enabled('swap_template'));

      if (isSpot) {
        const tpl = row.template || {};
        this.el.templateIds.textContent =
          `structure: ${tpl.structure_id || '—'}\nmotion: ${tpl.motion_id || '—'}\nvisual: ${tpl.visual_id || '—'}`;
        this.el.structure.value = '';
        this.el.motion.value = '';
        this.el.visual.value = '';
        this._renderSlots(row.slots || {});
      }
    }

    _renderSlots(slots) {
      this.el.slots.innerHTML = '';
      const keys = Object.keys(slots);
      if (!keys.length) {
        this.el.slots.innerHTML = '<p class="empty">该镜头暂无模板槽位</p>';
        return;
      }
      keys.forEach((k) => {
        const row = document.createElement('div');
        row.className = 'slot-row';
        const label = document.createElement('label');
        label.textContent = k;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = slots[k];
        input.dataset.slot = k;
        row.appendChild(label);
        row.appendChild(input);
        this.el.slots.appendChild(row);
      });
    }

    // ---------------- ops ----------------
    _buildOp(kind) {
      const sel = this.selected;
      const sid = sel && sel.shot_id;

      switch (kind) {
        case 'set_narration':
          if (!sid) return null;
          return { op: 'set_narration', shot_id: sid, text: this.el.narration.value.trim() };

        case 'set_subtitle':
          if (!sid) return null;
          return { op: 'set_subtitle', shot_id: sid, text: this.el.subtitle.value.trim() };

        case 'replace_footage': {
          if (!sid) return null;
          const q = this.el.footageQuery.value.trim();
          if (!q) { this.ctx.toast('请先填写画面描述', 'warn'); return null; }
          return { op: 'replace_footage', shot_id: sid, query: q };
        }

        case 'replace_footage_path': {
          if (!sid) return null;
          const p = this.el.footagePath.value.trim();
          if (!p) { this.ctx.toast('请填写素材文件绝对路径', 'warn'); return null; }
          return { op: 'replace_footage', shot_id: sid, clip_path: p };
        }

        case 'set_spotlight_slots': {
          if (!sid) return null;
          const slots = {};
          this.el.slots.querySelectorAll('input[data-slot]').forEach((inp) => {
            slots[inp.dataset.slot] = inp.value;
          });
          if (!Object.keys(slots).length) { this.ctx.toast('没有可提交的槽位', 'warn'); return null; }
          return { op: 'set_spotlight_slots', shot_id: sid, slots: slots };
        }

        case 'swap_template': {
          if (!sid) return null;
          const op = { op: 'swap_template', shot_id: sid };
          const s = this.el.structure.value.trim();
          const m = this.el.motion.value.trim();
          const v = this.el.visual.value.trim();
          if (s) op.structure_id = s;
          if (m) op.motion_id = m;
          if (v) op.visual_id = v;
          if (!s && !m && !v) { this.ctx.toast('至少填写一个模板 id', 'warn'); return null; }
          return op;
        }

        case 'delete_shot':
          if (!sid) return null;
          if (!confirm(`确认删除镜头 ${sid}？`)) return null;
          return { op: 'delete_shot', shot_id: sid };

        case 'set_bgm': {
          const op = { op: 'set_bgm', volume: Number(this.el.bgmVolume.value) };
          const q = this.el.bgmQuery.value.trim();
          if (q) op.query = q;
          return op;
        }

        default:
          return null;
      }
    }

    async _submit(kind) {
      if (!this.projectDir) { this.ctx.toast('请先选择一个成片工程', 'warn'); return; }
      const op = this._buildOp(kind);
      if (!op) return;
      if (!op.op.startsWith('set_bgm') && !this.selected) {
        this.ctx.toast('请先在时间轴上选择一个镜头', 'warn');
        return;
      }
      await this.applyOps([op]);
    }

    async applyOps(ops) {
      try {
        const data = await this.ctx.api.post('/api/project/ops', {
          project_dir: this.projectDir, ops: ops, dry_run: false,
        });
        this.setProject(data);
        const msgs = (data.results || []).map((r) => r.message).join('；');
        this.ctx.toast(msgs || '已应用', 'ok');
        return true;
      } catch (err) {
        this.ctx.toast(err.message, 'err');
        return false;
      }
    }

    // ---------------- regeneration ----------------
    _renderPending() {
      const steps = (this.project && this.project.pending_steps) || [];
      const dirty = (this.project && this.project.dirty) || {};
      const anyDirty = Object.keys(dirty).length > 0 || (this.project && this.project.bgm_dirty);

      this.el.pendingBox.hidden = steps.length === 0;
      this.el.pendingList.innerHTML = '';
      steps.forEach((s) => {
        const li = document.createElement('li');
        li.textContent = s.label;
        this.el.pendingList.appendChild(li);
      });

      this.el.regenBtn.disabled = !anyDirty && steps.length === 0;
      this.el.regenBtn.textContent = this.autoRestitch
        ? '重新渲染并拼接' : '重新渲染';
      if (this.ctx.timeline) this.ctx.timeline.setDirty(dirty);
    }

    _renderVersions() {
      const versions = (this.project && this.project.versions) || [];
      this.el.versionList.innerHTML = '';
      if (!versions.length) {
        this.el.versionList.innerHTML = '<p class="empty">暂无历史版本</p>';
        return;
      }
      versions.forEach((v) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML = '<div class="li-title"></div><div class="li-meta"></div>';
        item.querySelector('.li-title').textContent = v.name.replace(/\.json$/, '');
        item.querySelector('.li-meta').textContent = v.modified.replace('T', ' ').slice(0, 19);
        item.onclick = async () => {
          if (!confirm('回滚到该版本？当前状态会先存为一个新版本。')) return;
          try {
            const data = await this.ctx.api.post('/api/project/restore', {
              project_dir: this.projectDir, name: v.name,
            });
            this.setProject(data);
            this.ctx.toast('已回滚到 ' + v.name, 'ok');
          } catch (err) {
            this.ctx.toast('回滚失败: ' + err.message, 'err');
          }
        };
        this.el.versionList.appendChild(item);
      });
    }

    async regenerate() {
      if (!this.projectDir) return;
      this.el.regenBtn.disabled = true;
      try {
        const data = await this.ctx.api.post('/api/project/regenerate', {
          project_dir: this.projectDir,
        });
        if (data.status === 'noop') {
          this.ctx.toast(data.message || '没有需要重新生成的内容', 'warn');
          this.el.regenBtn.disabled = false;
          return;
        }
        this.ctx.toast('已开始重新渲染，进度见时间轴与日志', 'ok');
        if (this.ctx.onRegenStarted) this.ctx.onRegenStarted(data.job_id);
      } catch (err) {
        this.ctx.toast('重渲染失败: ' + err.message, 'err');
        this.el.regenBtn.disabled = false;
      }
    }

    async refresh() {
      if (!this.projectDir) return;
      try {
        const data = await this.ctx.api.get('/api/project', { dir: this.projectDir });
        this.setProject(data);
      } catch (err) { /* project may have been removed */ }
    }
  }

  window.AICutEditor = Editor;
})();
