/* Conversational editing panel.
 *
 * The agent returns validated ops but never applies them itself. Low-risk
 * batches are applied server-side in the same request; risky ones come back
 * with `needs_confirm` and are held here until the user approves, which is
 * what keeps "改一下字幕" and "把这个镜头删掉" from feeling equally cheap.
 */
(function () {
  'use strict';

  const INTENT_LABEL = {
    edit: '已定位',
    clarify: '需要澄清',
    question: '回答',
    reject: '无法执行',
  };

  class Chat {
    constructor(ctx) {
      this.ctx = ctx;              // { api, toast, editor }
      this.projectDir = '';
      this.busy = false;
      this.pending = null;         // ops awaiting explicit confirmation

      this.el = {
        log: document.getElementById('chatLog'),
        input: document.getElementById('chatInput'),
        send: document.getElementById('chatSend'),
        reset: document.getElementById('chatReset'),
        memory: document.getElementById('chatMemory'),
        modal: document.getElementById('confirmModal'),
        modalTitle: document.getElementById('confirmTitle'),
        modalText: document.getElementById('confirmText'),
        modalOps: document.getElementById('confirmOps'),
        modalOk: document.getElementById('confirmOk'),
        modalCancel: document.getElementById('confirmCancel'),
      };

      this._bind();
    }

    _bind() {
      this.el.send.onclick = () => this.send();
      this.el.input.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.send();
        }
      };
      this.el.reset.onclick = () => this.reset();
      this.el.modalCancel.onclick = () => this._closeModal(false);
      this.el.modal.onclick = (e) => { if (e.target === this.el.modal) this._closeModal(false); };
    }

    /** Drop the conversation, e.g. after the project was deleted. */
    clear() {
      this.projectDir = '';
      this.pending = null;
      this.el.log.innerHTML = '';
      this.el.memory.textContent = '记忆 0 轮';
    }

    setProject(projectDir) {
      if (this.projectDir === projectDir) return;
      this.projectDir = projectDir;
      this.el.log.innerHTML = '';
      this.pending = null;
      this._loadHistory();
      this._loadMemory();
    }

    // ---------------- messaging ----------------
    async send() {
      const text = this.el.input.value.trim();
      if (!text) return;
      if (!this.projectDir) { this.ctx.toast('请先选择一个成片工程', 'warn'); return; }
      if (this.busy) return;

      this.el.input.value = '';
      this._bubble('user', text);
      this._setBusy(true);
      const thinking = this._bubble('bot', '正在定位…', { tag: '思考中' });

      try {
        const data = await this.ctx.api.post('/api/project/chat', {
          project_dir: this.projectDir, message: text,
        });
        thinking.remove();
        this._renderReply(data);
        this._updateMemory(data.memory);
        if (data.project) {
          this.ctx.editor.setProject(data.project, { skipVideo: !!data.regen_job_id });
        }
        if (data.compression && data.compression.compressed) {
          this._sys(`对话历史已压缩（${data.compression.turns_compressed} 轮 → 摘要，` +
                    `${data.compression.tokens_before} → ${data.compression.tokens_after} tokens）`);
        }
      } catch (err) {
        thinking.remove();
        this._bubble('bot', '出错了：' + err.message, { cls: 'reject', tag: '错误' });
      } finally {
        this._setBusy(false);
      }
    }

    _renderReply(data) {
      const intent = data.intent || 'clarify';
      const cls = intent === 'reject' ? 'reject' : (intent === 'edit' ? '' : 'clarify');
      const bubble = this._bubble('bot', data.reply || '', {
        cls: cls, tag: INTENT_LABEL[intent] || intent,
      });

      const ops = data.ops || [];
      if (ops.length) {
        const box = document.createElement('div');
        box.className = 'msg-ops';
        ops.forEach((op) => {
          const row = document.createElement('div');
          row.textContent = this._describeOp(op);
          box.appendChild(row);
        });
        bubble.appendChild(box);
      }

      if (data.applied) {
        if (data.regen_job_id) {
          this._sys(
            `修改已生效，正在重渲染 ${(data.regen_steps || []).length} 个步骤` +
            `（只改动的镜头）…`
          );
          this.ctx.toast('修改已生效，正在自动重渲染改动镜头', 'ok');
          if (this.ctx.onRegenStarted) this.ctx.onRegenStarted(data.regen_job_id);
        } else {
          this.ctx.toast('修改已保存（模型未就绪，需手动重渲染）', 'ok');
        }
        return;
      }

      const needsConfirm = data.needs_confirm || [];
      if (ops.length && needsConfirm.length) {
        this._addConfirmAction(bubble, ops, needsConfirm);
      } else if (ops.length) {
        // Validated but not auto-applied (e.g. server declined) — offer a button
        // instead of silently dropping the proposal.
        this._addApplyAction(bubble, ops);
      }
    }

    _addApplyAction(bubble, ops) {
      const row = document.createElement('div');
      row.className = 'msg-actions';
      const btn = document.createElement('button');
      btn.className = 'btn btn-sm btn-primary';
      btn.textContent = '应用';
      btn.onclick = async () => { btn.disabled = true; await this._apply(ops); };
      row.appendChild(btn);
      bubble.appendChild(row);
    }

    _addConfirmAction(bubble, ops, needsConfirm) {
      const row = document.createElement('div');
      row.className = 'msg-actions';

      const ok = document.createElement('button');
      ok.className = 'btn btn-sm btn-warn';
      ok.textContent = '确认执行';
      ok.onclick = () => this._openModal(ops, needsConfirm, async () => {
        ok.disabled = true;
        skip.disabled = true;
        await this._apply(ops);
      });

      const skip = document.createElement('button');
      skip.className = 'btn btn-sm';
      skip.textContent = '不改了';
      skip.onclick = () => { ok.disabled = true; skip.disabled = true; };

      row.appendChild(ok);
      row.appendChild(skip);
      bubble.appendChild(row);
    }

    async _apply(ops) {
      try {
        const data = await this.ctx.api.post('/api/project/chat', {
          project_dir: this.projectDir, apply: true, ops: ops,
        });
        if (data.project) {
          this.ctx.editor.setProject(data.project, { skipVideo: !!data.regen_job_id });
        }
        const msg = (data.results || []).map((r) => r.message).join('；');
        this._sys('已应用：' + (msg || '完成'));
        if (data.regen_job_id) {
          this._sys('正在自动重渲染改动镜头…');
          this.ctx.toast('修改已生效，正在自动重渲染', 'ok');
          if (this.ctx.onRegenStarted) this.ctx.onRegenStarted(data.regen_job_id);
        } else {
          this.ctx.toast('修改已保存', 'ok');
        }
      } catch (err) {
        this._bubble('bot', '应用失败：' + err.message, { cls: 'reject', tag: '错误' });
      }
    }

    _describeOp(op) {
      const kind = op.op || '?';
      const target = op.shot_id || op.segment_id || '';
      const extras = [];
      if (op.text !== undefined) extras.push('"' + String(op.text).slice(0, 40) + '"');
      if (op.query) extras.push('检索: ' + op.query);
      if (op.clip_path) extras.push('文件: ' + op.clip_path);
      if (op.slots) extras.push(Object.keys(op.slots).join(','));
      if (op.structure_id || op.motion_id || op.visual_id) {
        extras.push([op.structure_id, op.motion_id, op.visual_id].filter(Boolean).join('/'));
      }
      if (op.volume !== undefined) extras.push('音量 ' + op.volume);
      if (op.order) extras.push('顺序 ' + op.order.join('→'));
      const label = (window.AICutOpLabels || {})[kind] || kind;
      return `${label}${target ? ' · ' + target : ''}${extras.length ? '  ' + extras.join('  ') : ''}`;
    }

    // ---------------- modal ----------------
    _openModal(ops, needsConfirm, onOk) {
      this.el.modalText.textContent =
        `以下 ${needsConfirm.length} 处属于高风险操作（会改变成片结构或删除内容），确认后才会执行：`;
      this.el.modalOps.textContent = ops.map((o) => this._describeOp(o)).join('\n');
      this.el.modal.hidden = false;
      this.el.modalOk.onclick = async () => {
        this._closeModal(true);
        await onOk();
      };
    }

    _closeModal() {
      this.el.modal.hidden = true;
      this.el.modalOk.onclick = null;
    }

    // ---------------- history & memory ----------------
    async _loadHistory() {
      if (!this.projectDir) return;
      try {
        const data = await this.ctx.api.get('/api/project/chat/history', { dir: this.projectDir });
        const msgs = data.messages || [];
        if (!msgs.length) {
          this._sys('已切换到当前成片，可以直接说想改什么。');
          return;
        }
        if (data.compressions) {
          this._sys(`（已加载历史对话，之前有 ${data.compressions} 次压缩）`);
        }
        for (const m of msgs) {
          if (m.role === 'user') {
            this._bubble('user', m.content);
          } else {
            const meta = m.meta || {};
            const intent = meta.intent || '';
            const cls = intent === 'reject' ? 'reject' : (intent === 'edit' ? '' : 'clarify');
            const tag = meta.parse_failed ? '解析失败'
              : (meta.rejected ? '校验失败' : '');
            this._bubble('bot', m.content, { cls, tag });
          }
        }
        this._sys('以上为历史对话，可继续对话。');
      } catch (err) {
        this._sys('已切换到当前成片，可以直接说想改什么。');
      }
    }

    async _loadMemory() {
      if (!this.projectDir) return;
      try {
        const data = await this.ctx.api.get('/api/project/chat/memory', { dir: this.projectDir });
        this._updateMemory(data);
      } catch (err) { /* no conversation yet */ }
    }

    _updateMemory(mem) {
      if (!mem) return;
      const turns = (mem.turns || []).length;
      let text = `记忆 ${turns} 轮`;
      if (mem.token_estimate) {
        text += ` · ${mem.token_estimate}/${mem.budget || 0} tokens`;
      }
      if (mem.compressions) text += ` · 压缩 ${mem.compressions} 次`;
      this.el.memory.textContent = text;
      this.el.memory.title = mem.summary ? '当前摘要:\n' + mem.summary : '尚无摘要';
    }

    async reset() {
      if (!this.projectDir) return;
      try {
        await this.ctx.api.post('/api/project/chat/reset?dir=' + encodeURIComponent(this.projectDir), null);
        this.el.log.innerHTML = '';
        this.pending = null;
        this._sys('对话已重置。');
        this._loadMemory();
      } catch (err) {
        this.ctx.toast('重置失败: ' + err.message, 'err');
      }
    }

    // ---------------- rendering helpers ----------------
    _setBusy(busy) {
      this.busy = busy;
      this.el.send.disabled = busy;
      this.el.send.textContent = busy ? '…' : '发送';
    }

    _bubble(role, text, opts) {
      opts = opts || {};
      const el = document.createElement('div');
      el.className = 'msg ' + role + (opts.cls ? ' ' + opts.cls : '');
      if (opts.tag) {
        const tag = document.createElement('span');
        tag.className = 'msg-tag';
        tag.textContent = opts.tag;
        el.appendChild(tag);
      }
      el.appendChild(document.createTextNode(text));
      this.el.log.appendChild(el);
      this.el.log.scrollTop = this.el.log.scrollHeight;
      return el;
    }

    _sys(text) {
      const el = document.createElement('div');
      el.className = 'chat-intro';
      el.textContent = text;
      this.el.log.appendChild(el);
      this.el.log.scrollTop = this.el.log.scrollHeight;
    }
  }

  window.AICutChat = Chat;
})();
