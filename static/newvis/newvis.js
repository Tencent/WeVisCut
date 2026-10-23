/* WeVisCut 创作台 — conversation-first shell.
 *
 * The backend speaks one event vocabulary for generation runs, chat-triggered
 * regeneration and manual re-renders. This file turns that stream into a
 * readable narrative: consecutive tools that belong to the same production
 * phase collapse into one numbered chapter, and each tool becomes a sentence
 * describing what it produced rather than what it is called.
 */
(function () {
  'use strict';

  const BRAND = 'WeVisCut';

  // ─────────────────────────── vocabulary ───────────────────────────

  const PHASE_TITLE = {
    input:   '理解创作输入',
    plan:    '完成分段与镜头规划',
    review:  '自检与修补规划',
    assets:  '生成配音与素材',
    compose: '装饰与生成片段',
    deliver: '检查并交付',
    edit:    '按你的要求调整',
    other:   '其他执行',
  };

  const TOOL_META = {
    Tool_SegmentArticle:          { phase: 'plan',    title: '拆解稿件结构' },
    Tool_PlanShots:               { phase: 'plan',    title: '规划镜头' },
    Tool_SelfReflect:             { phase: 'review',  title: '检查文字规划' },
    Tool_SelfReflect_Patch:       { phase: 'review',  title: '修补规划' },
    Tool_BatchVisualAudio:        { phase: 'assets',  title: '生成配音与画面素材', pro: true },
    Tool_MusicSearch:             { phase: 'assets',  title: '挑选配乐', pro: true },
    Tool_BatchDecorateSpotlights: { phase: 'compose', title: '装饰重点镜头' },
    Tool_BatchTimelineCompose:    { phase: 'compose', title: '生成片段', pro: true },
    Tool_SeverityRank:            { phase: 'deliver', title: '挑选质检镜头' },
    Tool_Stage4DimensionJudge:    { phase: 'deliver', title: '检查视觉结果' },
    Tool_Stage4DimensionalRepair: { phase: 'deliver', title: '修复不合格片段' },
    Tool_StitchVideo:             { phase: 'deliver', title: '合成并交付', pro: true },
  };

  const STAGE_LABEL = {
    agent_loop: '模型自主执行',
    plan: '理解稿件与镜头规划',
    reflect_repair_plan: '自检与局部修复',
    asset_and_compose: '素材与合成',
    quality_repair: '质量检查与修复',
    final_stitch: '最终拼接',
    rerender: '重新渲染',
    done: '完成',
  };

  const RUN_STATUS_LABEL = {
    completed: '已完成',
    completed_with_errors: '完成（有错误）',
    cancelled: '已取消',
    error: '失败',
  };

  const INTENT_LABEL = {
    edit: '已定位', clarify: '需要澄清', question: '回答', reject: '无法执行',
  };

  const ACTIVE_STATUSES = new Set(['running', 'paused', 'queued', 'cancelling']);

  /* Ready-to-run briefs. Each states a duration and word count so the length
   * parser locks a target, then an angle, a structure and a tone — a vague
   * brief is the main reason a draft comes back generic or the wrong length.
   * The chip shows the short label; clicking loads the full text. */
  const SAMPLE_PROMPTS = [
    {
      tag: '科技', title: '量子计算机为什么这么难造', sec: '90 秒',
      text: '做一支 90 秒、约 410 字的科技科普短片，讲清楚量子计算机为什么造起来这么难。'
          + '开头从「普通比特只能是 0 或 1，量子比特可以同时是两者」切入。'
          + '中间重点讲退相干这一个核心障碍：为什么要把芯片冷却到接近绝对零度，'
          + '为什么外界一点振动、一点电磁噪声就会让整次计算作废，'
          + '以及为什么纠错要用成百上千个物理量子比特才能撑起一个可用的逻辑比特。'
          + '结尾落在「我们不是在造更快的电脑，而是在造一种完全不同的机器」。'
          + '语气冷静克制，多用具体数字和实物画面，不要科幻式的夸张形容。',
    },
    {
      tag: '金融', title: '通货膨胀到底偷走了什么', sec: '60 秒',
      text: '做一支 60 秒、约 280 字的金融科普短片，用普通人的视角讲清楚通货膨胀到底偷走了什么。'
          + '开头用一个具体场景切入：十年前一碗面多少钱，现在多少钱，同期工资涨了多少。'
          + '中间讲清楚「钱变多」和「东西变贵」之间的关系，'
          + '以及为什么放在银行里不动的钱会在无声中缩水。'
          + '结尾给一句克制的提醒，不要推荐任何具体理财产品、不要给投资建议。'
          + '语气平实，像朋友坐下来聊天，避免专业术语堆砌。',
    },
    {
      tag: '历史', title: '敦煌藏经洞是怎么流散的', sec: '120 秒',
      text: '做一支 120 秒、约 550 字的历史科普短片，讲敦煌藏经洞是怎么被发现、又是怎么流散的。'
          + '从 1900 年王圆箓清理积沙意外发现密室讲起，'
          + '中间讲斯坦因和伯希和如何用极低的代价带走大批经卷写本，'
          + '并还原当时清政府和地方官对这批东西的真实态度。'
          + '结尾不煽情、不谴责某一个人，落在「一个文明的记忆，是怎么在无人在意的时候被搬空的」。'
          + '多用具体年份、人名和数字，画面感要强。',
    },
    {
      tag: '人体', title: '人为什么一定要睡觉', sec: '75 秒',
      text: '做一支 75 秒、约 350 字的人体科普短片，讲人为什么一定要睡觉。'
          + '核心讲清楚大脑在深睡时会启动类淋巴系统，'
          + '把白天堆积的代谢废物冲洗出去，其中就包括与阿尔茨海默病相关的 β-淀粉样蛋白。'
          + '中间对比一晚不睡和连续熬夜，对反应速度、情绪控制和免疫力分别有什么影响。'
          + '结尾落在「睡眠不是停机，而是维护」。'
          + '语气温和但有说服力，多用可视化的比喻，不要贩卖焦虑。',
    },
  ];

  /* The composer has two entry modes. `brief` asks the policy for a script
   * first and then runs the pipeline on it; `script` runs the pipeline on
   * exactly what was pasted. Once a project exists both give way to editing. */
  const INPUT_MODES = {
    brief: {
      desc: '模型先写出有头有尾的口播稿，再自动开拍',
      placeholder: '描述你想要的片子，例如：一支 60 秒、280 字的城市清晨短片，温柔口吻…',
    },
    script: {
      desc: '按你给的稿子直接开拍，一个字都不改写',
      placeholder: '粘贴完整口播稿…',
    },
  };
  const MODE_STORAGE_KEY = 'aicut.newvis.inputMode';
  const SESSION_STORAGE_KEY = 'aicut.newvis.lastSession';
  /* Past this many characters a "brief" may really be a pasted script, and
   * below the lower bound a "script" is almost certainly a brief. */
  const LOOKS_LIKE_SCRIPT = 200;
  const LOOKS_LIKE_BRIEF = 40;
  /* Length alone misfires: a well-written brief runs long too. What separates
   * them is that a brief gives instructions — a runtime, a tone, a structure —
   * while narration just narrates. */
  const BRIEF_MARKER_RE =
    /\d+\s*(?:秒|分钟|字)|[做剪写]一[支个条段篇]|语气|口吻|风格|结尾落在|开头(?:用|从)|讲清楚|不要/;

  function looksLikeScript(text) {
    return text.length > LOOKS_LIKE_SCRIPT && !BRIEF_MARKER_RE.test(text);
  }

  // ─────────────────────────── helpers ───────────────────────────

  const fmtTime = window.AICutFmtTime || function (sec) {
    if (!sec || sec < 0) sec = 0;
    return Math.floor(sec / 60) + ':' + String(Math.floor(sec % 60)).padStart(2, '0');
  };

  function num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function icon(kind) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('class', 'step-mark ' + kind);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    path.setAttribute('d', kind === 'ok' ? 'M3 8.4 6.4 12 13 4.6'
      : kind === 'err' ? 'M4 4l8 8M12 4l-8 8'
      : 'M8 3.5v5.6M8 12.2v.4');
    svg.appendChild(path);
    return svg;
  }

  /** Session expiry looks like a burst of 401s; bounce to login once. */
  function toLogin() {
    if (window.__wedRedirecting) return;
    window.__wedRedirecting = true;
    const next = encodeURIComponent(location.pathname + location.search);
    location.href = '/login?next=' + next;
  }

  const api = {
    async _fetch(url, init, opts) {
      const timeoutMs = Number((opts && opts.timeoutMs) || 0);
      if (!timeoutMs || timeoutMs <= 0) return fetch(url, init);
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), timeoutMs);
      try {
        return await fetch(url, { ...(init || {}), signal: ctl.signal });
      } catch (err) {
        if (err && err.name === 'AbortError') {
          throw new Error('请求超时，请重试');
        }
        throw err;
      } finally {
        clearTimeout(timer);
      }
    },
    async _json(res) {
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (e) { /* non-JSON error body */ }
      if (!res.ok) {
        if (res.status === 401) toLogin();
        const detail = (data && (data.detail || data.message)) || text || res.statusText;
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      return data;
    },
    async get(path, params, opts) {
      const url = new URL(path, window.location.origin);
      Object.entries(params || {}).forEach(([k, v]) => url.searchParams.set(k, v));
      return this._json(await this._fetch(url.toString(), null, opts));
    },
    async post(path, body, opts) {
      return this._json(await this._fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body === null || body === undefined ? null : JSON.stringify(body),
      }, opts));
    },
    async del(path, opts) {
      return this._json(await this._fetch(path, { method: 'DELETE' }, opts));
    },
  };

  function toast(text, kind) {
    const stack = document.getElementById('toasts');
    const node = el('div', 'toast' + (kind ? ' ' + kind : ''), text);
    stack.appendChild(node);
    setTimeout(() => node.remove(), kind === 'err' ? 7000 : 4200);
  }

  /** Turn a tool observation into one sentence a non-engineer can read. */
  function summarise(tool, status, obs) {
    const s = (obs && obs.summary) || {};
    const message = String(s.message || (obs && obs.message) || '').trim();

    if (status === 'blocked_precondition') {
      const missing = [].concat((obs && obs.missing) || s.missing || []);
      return {
        tone: 'warn',
        text: '前置条件还没准备好' + (missing.length ? '：' + missing.join('、') : '') +
              '，已交回模型重新决策。',
      };
    }
    if (status === 'budget_exhausted') {
      return { tone: 'warn', text: message || '这一步的重试预算已用尽，转入收尾。' };
    }
    if (status === 'error' || status === 'timeout') {
      return { tone: 'err', text: message || '这一步没有执行成功。' };
    }

    switch (tool) {
      case 'Tool_SegmentArticle': {
        const k = num(s.num_segments);
        const style = String(s.music_style || '').trim();
        return {
          outputs: k,
          text: `已把稿件拆成 ${k} 个章节` + (style ? `，整体基调定为「${style}」` : '') + '。',
        };
      }
      case 'Tool_PlanShots': {
        const added = num(s.num_shots_added);
        const total = num(s.total_shots);
        return {
          outputs: added,
          text: `完成 ${added} 个镜头的规划` +
                (total && total !== added ? `，累计 ${total} 个镜头` : '') + '。',
        };
      }
      case 'Tool_SelfReflect': {
        const issues = num(s.num_issues);
        return {
          text: issues
            ? `自检发现 ${issues} 处需要调整的地方。`
            : '故事、镜头与节奏已完成检查，未发现问题。',
        };
      }
      case 'Tool_SelfReflect_Patch': {
        const ids = [].concat(s.patched_shot_ids || []);
        return {
          outputs: ids.length,
          text: ids.length ? `已修补 ${ids.length} 个镜头的规划。` : '规划已按自检结果调整。',
        };
      }
      case 'Tool_BatchVisualAudio': {
        const tts = num(s.tts_success_count);
        const vis = num(s.visual_success_count);
        const shots = num(s.shot_count);
        // A re-render of one shot sometimes reports no counts at all. Saying
        // "完成 0 条" then contradicts the video that just changed, so fall
        // back to what is known for certain: the shots that were processed.
        if (!tts && !vis && shots) {
          return { outputs: shots, text: `已重做 ${shots} 个镜头的配音与画面素材。` };
        }
        return { outputs: tts + vis, text: `完成 ${tts} 条配音与 ${vis} 段画面素材。` };
      }
      case 'Tool_BatchDecorateSpotlights': {
        const ids = [].concat(s.decorated_shot_ids || []);
        return {
          outputs: ids.length,
          text: ids.length ? `完成 ${ids.length} 个重点镜头的模板装饰。` : '重点镜头装饰已完成。',
        };
      }
      case 'Tool_BatchTimelineCompose': {
        const n = num(s.render_success_count) || num(s.shot_count);
        return { outputs: n, text: n ? `完成 ${n} 个片段。` : '片段合成已完成。' };
      }
      case 'Tool_MusicSearch': {
        const list = [].concat(s.tracks || []);
        const first = list[0] || {};
        const name = String(first.name || first.title || first.track_id || '').trim();
        return {
          outputs: num(s.selected_count) || list.length,
          text: name ? `已选定配乐「${name}」。` : '已完成配乐检索。',
        };
      }
      case 'Tool_SeverityRank': {
        const top = [].concat(s.top_shot_ids || []);
        const ranked = num(s.ranked_count);
        return {
          text: `已从 ${ranked || '全部'} 个镜头中挑出 ${top.length} 个优先质检。`,
        };
      }
      case 'Tool_Stage4DimensionJudge': {
        const passed = num(s.passed_count);
        const failed = num(s.failed_count) || [].concat(s.failed_shot_ids || []).length;
        return failed
          ? { tone: 'warn', text: `质检完成，${passed} 个通过，${failed} 个需要修复。` }
          : { text: `已检查片段质量、镜头衔接与全片连贯性，${passed || '全部'} 个镜头通过。` };
      }
      case 'Tool_Stage4DimensionalRepair': {
        const ids = [].concat(s.repaired_shot_ids || []);
        return {
          outputs: ids.length,
          text: ids.length ? `已修复 ${ids.length} 个片段。` : '修复已执行。',
        };
      }
      case 'Tool_StitchVideo': {
        const ok = !!(s.has_final_video || s.final_video_path || s.output_path);
        const dur = num(s.total_duration_sec);
        if (ok) {
          return { text: `通过检查的片段已合成为成片${dur ? `，全片 ${fmtTime(dur)}` : ''}。` };
        }
        // Deferred stitching is the normal path: the shots are validated here
        // and the file is only written when the user exports.
        const ready = num(s.ready_shot_count) || num(s.shot_count);
        if (s.deferred || ready) {
          return {
            outputs: ready,
            text: `已完成合成检查${ready ? `，${ready} 个片段可用` : ''}` +
                  `${dur ? `，全片约 ${fmtTime(dur)}` : ''}，等待导出成片。`,
          };
        }
        return { tone: 'warn', text: '拼接已执行，但没有拿到成片文件。' };
      }
      default:
        return { text: message || '完成。' };
    }
  }

  // ---- input-chapter copy, shared by the live run and the replay rebuild ----
  const inputStep = {
    brief: () => ({ text: '已收到创作需求，先把口播稿写出来。' }),
    script: (script) => ({
      text: `已理解创作要求，共 ${script.length} 字，开始自主执行。`,
    }),
    draft: (meta) => {
      const t = (meta && meta.target) || {};
      const target = t.duration_sec
        ? `（目标 ${t.duration_sec} 秒 · 约 ${t.word_count} 字）` : '';
      // State the length, do not editorialise about it: the run carries on
      // either way, so a warning tone only asks the user to worry about a
      // number they can already see.
      return {
        text: `已写好口播稿，共 ${meta.char_count} 字，` +
              `预计朗读 ${fmtTime(meta.estimated_sec)}${target}。`,
      };
    },
  };

  // ─────────────────────────── narrative stream ───────────────────────────

  class Stream {
    constructor(root, hero) {
      this.root = root;
      this.hero = hero;
      this.reset();
    }

    reset() {
      this.root.innerHTML = '';
      if (this.hero) this.root.appendChild(this.hero);
      this.groups = [];
      this.chapters = 0;
      this.steps = 0;
      this.outputs = 0;
      this.openStep = null;
      this.pendingThought = '';
      this.liveThought = null;
      this.allFolded = false;
    }

    _hideHero() {
      if (this.hero && this.hero.parentNode) this.hero.remove();
    }

    _atBottom() {
      return this.root.scrollTop + this.root.clientHeight > this.root.scrollHeight - 160;
    }

    _append(node) {
      const follow = this._atBottom();
      this._hideHero();
      this.root.appendChild(node);
      if (follow) this.scrollToEnd();
    }

    scrollToEnd() {
      requestAnimationFrame(() => { this.root.scrollTop = this.root.scrollHeight; });
    }

    // ---- plain blocks ----
    /** Any prose block ends the running chapter: steps that arrive afterwards
     *  must start a new one, or they would be appended above the prose. */
    _breakChain() {
      this.settle();
      this.groups = [];
    }

    userMessage(text) {
      this._breakChain();
      const wrap = el('div', 'user-msg');
      wrap.appendChild(el('span', 'user-msg-who', '我'));
      wrap.appendChild(el('div', 'user-msg-body', text));
      this._append(wrap);
    }

    botMessage(text, opts) {
      opts = opts || {};
      this._breakChain();
      const node = el('div', 'bot-msg' + (opts.cls ? ' ' + opts.cls : ''));
      if (opts.tag) node.appendChild(el('span', 'bot-msg-tag', opts.tag));
      node.appendChild(document.createTextNode(text));
      this._append(node);
      return node;
    }

    note(text) {
      this._breakChain();
      const node = el('div', 'sys-note');
      node.appendChild(el('span', null, text));
      this._append(node);
      return node;
    }

    // ---- chapters ----
    /** Reuse the trailing chapter while the phase holds; a phase change opens
     *  a new numbered chapter so the column always reads top to bottom. */
    chapter(phaseId, titleOverride) {
      const last = this.groups[this.groups.length - 1];
      if (last && last.phase === phaseId && last.node.parentNode === this.root) return last;

      if (last) last.node.classList.replace('active', 'done');

      this.chapters += 1;
      const node = el('section', 'phase active');

      const head = el('button', 'phase-head');
      head.type = 'button';
      const numEl = el('span', 'phase-num', String(this.chapters).padStart(2, '0'));
      const title = el('span', 'phase-title', titleOverride || PHASE_TITLE[phaseId] || phaseId);
      const rule = el('span', 'phase-rule');
      const meta = el('span', 'phase-meta', '');
      const caret = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      caret.setAttribute('viewBox', '0 0 16 16');
      caret.setAttribute('class', 'phase-caret');
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', 'M3 6.2 8 10.6l5-4.4');
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', 'currentColor');
      p.setAttribute('stroke-width', '1.6');
      p.setAttribute('stroke-linecap', 'round');
      p.setAttribute('stroke-linejoin', 'round');
      caret.appendChild(p);
      head.append(numEl, title, rule, meta, caret);

      const stepsOuter = el('div', 'phase-steps');
      const stepsInner = el('div');
      stepsOuter.appendChild(stepsInner);

      node.append(head, stepsOuter);

      const group = { phase: phaseId, node, body: stepsInner, meta, count: 0 };
      head.onclick = () => node.classList.toggle('collapsed');

      this._append(node);
      this.groups.push(group);
      return group;
    }

    /** Open a step inside `phaseId`. Returns a handle used to finish it. */
    beginStep(phaseId, title, opts) {
      opts = opts || {};
      const group = this.chapter(phaseId, opts.chapterTitle);
      const node = el('div', 'step running');

      const head = el('div', 'step-head');
      head.appendChild(el('span', 'spinner'));
      head.appendChild(el('span', 'step-title', title));
      const chips = el('div', 'step-chips');
      if (opts.pro) chips.appendChild(el('span', 'chip chip-pro', 'Pro'));
      head.appendChild(chips);
      node.appendChild(head);

      const thought = this.pendingThought.trim();
      this.pendingThought = '';
      if (this.liveThought) {
        this.liveThought.node.classList.remove('live');
        const badge = this.liveThought.node.querySelector('.step-think-live');
        if (badge) badge.remove();
        node.appendChild(this.liveThought.node);
        this.liveThought = null;
      } else if (thought) {
        node.appendChild(this._thinkBlock(thought));
      }

      const card = el('div', 'step-card', opts.placeholder || '执行中…');
      node.appendChild(card);

      group.body.appendChild(node);
      group.count += 1;
      this.steps += 1;
      group.meta.textContent = group.count + ' 步';

      const follow = this._atBottom();
      if (follow) this.scrollToEnd();

      const handle = {
        node, card, chips,
        /** Hang the full drafted narration under the step's one-line result.
         *  `count` comes from the backend so the card and the sentence above
         *  it quote the same number (raw length would include punctuation). */
        attachScript: (text, count) => {
          const box = el('div', 'script-card');
          const bar = el('div', 'script-card-head');
          bar.append(
            el('span', 'script-card-title', '口播稿'),
            el('span', 'script-card-count', (count || text.length) + ' 字'));
          const copy = el('button', 'mini-btn', '复制');
          copy.type = 'button';
          copy.onclick = () => {
            const done = () => { copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制'; }, 1600); };
            if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
            else done();
          };
          bar.appendChild(copy);
          box.append(bar, el('div', 'script-card-body', text));
          node.appendChild(box);
          return box;
        },
        finish: (result) => {
          const spinner = head.querySelector('.spinner');
          const tone = result.tone || '';
          node.classList.remove('running');
          if (tone) node.classList.add(tone);
          if (spinner) spinner.replaceWith(icon(tone === 'err' ? 'err' : tone === 'warn' ? 'warn' : 'ok'));
          card.textContent = result.text || '完成。';
          if (result.outputs > 0) {
            this.outputs += result.outputs;
            chips.appendChild(el('span', 'chip chip-out', result.outputs + ' 项产出'));
          }
          if (this._atBottom()) this.scrollToEnd();
        },
      };
      this.openStep = handle;
      return handle;
    }

    /** A step that is already resolved (no spinner phase). */
    instantStep(phaseId, title, result, opts) {
      const handle = this.beginStep(phaseId, title, opts);
      handle.finish(result);
      this.openStep = null;
      return handle;
    }

    _thinkBlock(text, opts) {
      const wrap = el('div', 'step-think');
      const body = el('div', 'step-think-text', text);
      wrap._body = body;
      wrap.appendChild(body);
      const toggle = el('button', 'step-think-toggle', '展开思考');
      toggle.type = 'button';
      toggle.onclick = () => {
        const open = wrap.classList.toggle('open');
        toggle.textContent = open ? '收起思考' : '展开思考';
      };
      wrap.appendChild(toggle);
      // Only offer the toggle when there is something clipped to reveal. A block
      // that is still streaming starts empty, so keep its toggle regardless.
      if (!(opts && opts.keepToggle)) {
        requestAnimationFrame(() => {
          if (body.scrollHeight <= body.clientHeight + 2) toggle.remove();
        });
      }
      return wrap;
    }

    addThoughtDelta(text) {
      const t = String(text || '');
      if (!t.trim()) return;
      if (!this.liveThought) {
        const node = this._thinkBlock('', { keepToggle: true });
        node.classList.add('live');
        const badge = el('span', 'step-think-live', '思考中');
        node.insertBefore(badge, node.firstChild);
        this.liveThought = { node, body: node._body, text: '' };
        this._append(node);
      }
      this.liveThought.text += t;
      this.liveThought.body.textContent = this.liveThought.text.trimStart();
      if (this._atBottom()) this.scrollToEnd();
    }

    addThought(text) {
      const t = String(text || '').trim();
      if (!t) return;
      if (this.liveThought) {
        this.liveThought.text = t;
        this.liveThought.body.textContent = t;
        return;
      }
      this.pendingThought = this.pendingThought ? this.pendingThought + '\n' + t : t;
    }

    settle() {
      // A block still marked "思考中" when nothing follows it would sit there
      // spinning forever, so close it out here too.
      if (this.liveThought) {
        this.liveThought.node.classList.remove('live');
        const badge = this.liveThought.node.querySelector('.step-think-live');
        if (badge) badge.remove();
        this.liveThought = null;
      }
      const last = this.groups[this.groups.length - 1];
      if (last) last.node.classList.replace('active', 'done');
    }

    toggleAll(fold) {
      this.allFolded = fold;
      this.groups.forEach((g) => g.node.classList.toggle('collapsed', fold));
    }
  }

  // ─────────────────────────── app ───────────────────────────

  class App {
    constructor() {
      this.cfg = null;
      this.runId = null;
      this.ws = null;
      this.wsRetry = 0;
      this.running = false;
      this.projectDir = '';
      this.logs = { console: [], raw: [] };
      this.logView = 'console';
      this.busy = false;
      this.projectTitle = '';
      // What produced the current run, so a socket replay can rebuild the
      // input chapter the same way the live run drew it.
      this.session = null;

      let stored = null;
      try { stored = localStorage.getItem(MODE_STORAGE_KEY); } catch (e) { /* private mode */ }
      this.inputMode = INPUT_MODES[stored] ? stored : 'brief';

      this.el = {};
      [
        'stream', 'hero', 'heroChips', 'convoStat', 'btnFoldAll',
        'composerInput', 'btnSend', 'btnStop', 'btnPlus', 'composerHint',
        'composerBar', 'modeSwitch', 'modeDesc', 'editBar', 'editLabel', 'btnExitEdit',
        'modelDot', 'modelText', 'modelPill', 'runTimer',
        'btnNew', 'btnLibrary', 'userChip', 'userName', 'btnLogout',
        'player', 'playerEmpty', 'phaseLabel', 'overallBar', 'overallPct',
        'viewerBar', 'viewerTag', 'btnExport',
        'versionStrip', 'versionList',
        'statDuration', 'statShots', 'statSpotlight', 'statSteps',
        'foldRail', 'foldTimeline', 'foldLog', 'railNote', 'railEmpty', 'timelineNote',
        'consoleBody', 'consoleCount',
        'drawer', 'scrim', 'drawerClose', 'runList', 'projectList', 'refreshRuns',
        'confirmModal', 'confirmText', 'confirmOps', 'confirmOk', 'confirmCancel',
      ].forEach((id) => { this.el[id] = document.getElementById(id); });

      this.stream = new Stream(this.el.stream, this.el.hero);
      this.timeline = new window.AICutTimeline({
        onSelect: (track, clip) => this.onClipSelected(track, clip),
      });
      this.rail = new window.AICutRail();

      this._bind();
      this.boot();
    }

    // ---------------- wiring ----------------
    _bind() {
      const input = this.el.composerInput;
      input.addEventListener('input', () => { this._autoGrow(); this._syncComposer(); });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          e.preventDefault();
          this.send();
        }
      });
      this.el.btnSend.onclick = () => this.send();
      this.el.btnStop.onclick = () => this.stopRun();
      this.el.btnExport.onclick = () => this.exportVideo();
      this.el.btnPlus.onclick = () => this.openDrawer();
      this.el.btnExitEdit.onclick = () => this.newSession();

      this.el.modeSwitch.querySelectorAll('.mode-tab').forEach((tab) => {
        tab.onclick = () => this.setInputMode(tab.dataset.mode);
      });

      this.el.btnNew.onclick = () => this.newSession();
      this.el.btnLibrary.onclick = () => this.openDrawer();
      this.el.btnLogout.onclick = () => this.logout();
      this.el.drawerClose.onclick = () => this.closeDrawer();
      this.el.scrim.onclick = () => this.closeDrawer();
      this.el.refreshRuns.onclick = () => { this.loadRuns(); this.loadProjects({ force: true }); };

      this.el.btnFoldAll.onclick = () => {
        const fold = !this.stream.allFolded;
        this.stream.toggleAll(fold);
        this.el.btnFoldAll.classList.toggle('folded', fold);
      };

      // The timeline measures its viewport on render, which is zero while the
      // <details> is closed — re-fit once it actually has a width.
      this.el.foldTimeline.addEventListener('toggle', () => {
        if (this.el.foldTimeline.open) requestAnimationFrame(() => this.timeline.fit());
      });
      this.el.foldTimeline.querySelector('.fold-tools').onclick = (e) => e.preventDefault();

      document.querySelectorAll('[data-log]').forEach((btn) => {
        btn.onclick = (e) => {
          e.preventDefault();
          document.querySelectorAll('[data-log]').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          this.logView = btn.dataset.log;
          this._renderLogs();
        };
      });

      this.el.confirmCancel.onclick = () => { this.el.confirmModal.hidden = true; };
      this.el.confirmModal.onclick = (e) => {
        if (e.target === this.el.confirmModal) this.el.confirmModal.hidden = true;
      };

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.closeDrawer();
          this.el.confirmModal.hidden = true;
        }
      });
      window.addEventListener('beforeunload', () => { if (this.ws) this.ws.close(); });
    }

    _autoGrow() {
      const t = this.el.composerInput;
      t.style.height = 'auto';
      t.style.height = Math.min(t.scrollHeight, 168) + 'px';
    }

    // ---------------- boot ----------------
    async boot() {
      SAMPLE_PROMPTS.forEach((sample) => {
        const chip = el('button', 'hero-chip');
        chip.type = 'button';
        chip.title = sample.text;
        chip.append(
          el('span', 'chip-tag', sample.tag),
          el('span', 'chip-title', sample.title),
          el('span', 'chip-meta', sample.sec));
        chip.onclick = () => {
          this.setInputMode('brief');
          this.el.composerInput.value = sample.text;
          this._autoGrow();
          this._syncComposer();
          this.el.composerInput.focus();
        };
        this.el.heroChips.appendChild(chip);
      });
      this._syncComposer();

      try {
        this.cfg = await api.get('/api/config');
      } catch (err) {
        toast('无法读取配置: ' + err.message, 'err');
        this.cfg = { features: {}, tracks: [] };
      }

      // Not cfg.name: that is shared with the legacy shell at `/`, which keeps
      // its own product name.
      document.title = BRAND + ' · 创作台';
      window.AICutOpLabels = (this.cfg.editor && this.cfg.editor.op_labels) || {};
      this.timeline.setTracks(this.cfg.tracks || []);
      this._showUser(this.cfg.user, (this.cfg.features || {}).auth);

      this.pollStatus();
      setInterval(() => this.pollStatus(), 4000);
      this.loadRuns();
      // `/api/projects` walks the whole workspace and can take a minute on a
      // network filesystem, so it is fetched on demand rather than at boot.
      await this._restoreLastSession();
    }

    // ---------------- account ----------------
    _showUser(user, authEnabled) {
      if (!authEnabled || !user) { this.el.userChip.hidden = true; return; }
      this.el.userChip.hidden = false;
      this.el.userName.textContent = user.display_name || user.username;
      this.el.userName.classList.toggle('admin', !!user.is_admin);
      this.el.userName.title = user.is_admin
        ? '超级管理员：可以看到所有人的作品'
        : '只能看到自己的作品';
    }

    async logout() {
      try { await api.post('/api/auth/logout', null); } catch (e) { /* going anyway */ }
      try { localStorage.removeItem(SESSION_STORAGE_KEY); } catch (e) { /* noop */ }
      location.href = '/login';
    }

    // ---------------- session continuity ----------------
    _rememberSession() {
      try {
        if (this.projectDir) {
          localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
            projectDir: this.projectDir, runId: this.runId || '',
          }));
        } else {
          localStorage.removeItem(SESSION_STORAGE_KEY);
        }
      } catch (e) { /* private mode */ }
    }

    /** Reopen whatever the user was working on before the page went away. */
    async _restoreLastSession() {
      if (!(this.cfg.features || {}).editor) return;
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || 'null'); }
      catch (e) { saved = null; }
      if (!saved || !saved.projectDir) return;
      await this.loadProject(saved.projectDir, { restoring: true });
    }

    /** Rebuild a finished run's narrative from the recorded journal. */
    async _restoreTrajectory(runId) {
      if (!runId) return false;
      let data;
      try {
        data = await api.get(`/api/runs/${runId}/events`);
      } catch (err) {
        return false;                     // journal pruned or run deleted
      }
      if (data.input_text) {
        this.currentInput = data.input_text;
        this.session = {
          mode: 'script', brief: '', script: data.input_text, draft: null,
        };
        this._seedStream();
      }
      // `run_started` / `run_complete` drive live state and would re-trigger
      // project loading; `progress` is already applied from the project.
      const skip = /^(run_started|run_complete|progress|replay_start|replay_end)$/;
      let shown = 0;
      (data.events || []).forEach((ev) => {
        if (!ev || skip.test(ev.type || '')) return;
        try { this.onEvent(ev); shown += 1; } catch (e) { /* keep going */ }
      });
      this.stream.settle();
      return shown > 0;
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
      if (m.loaded) {
        const cluster = await this._cluster();
        this._setModel('ok', cluster.label || '模型已就绪');
        this.el.modelPill.title = [
          m.base_model, m.adapter && ('adapter: ' + m.adapter), m.backend, cluster.detail,
        ].filter(Boolean).join('\n');
      } else if (m.load_error) {
        this._setModel('err', '模型未就绪');
        this.el.modelPill.title = m.load_error;
      } else if (m.skipped) {
        this._setModel('err', '模型未加载');
        this.el.modelPill.title = '服务以 --no-model 启动：界面可浏览，但无法生成或重渲染';
      } else {
        this._setModel('loading', '模型加载中');
        this.el.modelPill.title = '首次加载权重通常需要 30-60 秒';
      }
      this._syncComposer();
    }

    /** Multi-GPU load, read through the studio port since the router is
     *  loopback-only. Returns empty strings on a single-GPU deployment. */
    async _cluster() {
      let c;
      try {
        c = await api.get('/api/cluster');
      } catch (err) {
        return {};
      }
      if (!c.enabled) return {};
      if (!c.reachable) return { label: '模型已就绪', detail: '集群路由不可达：' + (c.error || '') };
      const lines = (c.backends || []).map((b, i) => {
        const gpu = 'GPU' + i + ' ' + b.url.replace(/^https?:\/\//, '');
        const load = b.inflight + ' 并发 / 累计 ' + b.total;
        const lat = b.last_latency_ms ? ' · ' + b.last_latency_ms + 'ms' : '';
        return gpu + '  ' + (b.healthy ? load + lat : '离线（失败 ' + b.failures + '）');
      });
      return {
        label: '模型已就绪 · ' + c.healthy + '/' + c.total + ' 卡',
        detail: ['集群并发 ' + c.inflight, ...lines].join('\n'),
      };
    }

    _setModel(state, text) {
      this.el.modelDot.className = 'dot ' + state;
      this.el.modelText.textContent = text;
      this.modelReady = state === 'ok';
      this.modelBlocked = state === 'ok' ? '' : text;
    }

    // ---------------- composer ----------------
    /** `edit` wins over the input modes: once a project exists in this session
     *  the composer talks to it, until the user starts a new creation. */
    get mode() { return this.projectDir ? 'edit' : this.inputMode; }

    setInputMode(name) {
      if (!INPUT_MODES[name] || this.inputMode === name) return;
      this.inputMode = name;
      try { localStorage.setItem(MODE_STORAGE_KEY, name); } catch (e) { /* private mode */ }
      this._syncComposer();
      this.el.composerInput.focus();
    }

    _syncComposer() {
      const editing = this.mode === 'edit';
      const text = this.el.composerInput.value.trim();
      const ready = this.modelReady && !this.running && !this.busy;
      // The stop button takes the send button's place while a run is in flight,
      // so there is always exactly one action at the end of the composer.
      this.el.btnStop.hidden = !this.running;
      this.el.btnSend.hidden = !!this.running;
      this.el.btnSend.disabled = !(ready && text.length > 0);

      this.el.composerBar.hidden = editing;
      this.el.editBar.hidden = !editing;
      if (editing) {
        this.el.editLabel.textContent =
          '正在编辑' + (this.projectTitle ? `《${this.projectTitle}》` : '当前成片');
      } else {
        this.el.modeSwitch.querySelectorAll('.mode-tab').forEach((tab) => {
          tab.classList.toggle('active', tab.dataset.mode === this.inputMode);
        });
        this.el.modeDesc.textContent = INPUT_MODES[this.inputMode].desc;
      }

      this.el.composerInput.placeholder = editing
        ? '继续要求调整片段或风格…'
        : INPUT_MODES[this.inputMode].placeholder;

      this.el.composerHint.textContent = this._hint(editing, text);
    }

    _hint(editing, text) {
      if (this.running) return '正在生成中，完成后可以继续提要求';
      if (this.busy) return '正在处理…';
      if (!this.modelReady) return this.modelBlocked || '等待模型加载完成…';
      if (editing) return 'Enter 发送 · 说明想改什么，模型只会重渲染受影响的镜头';
      // Pasting a script into "brief" would have the model rewrite it, and
      // sending a one-line brief as a "script" would narrate the brief itself.
      if (this.inputMode === 'brief' && looksLikeScript(text)) {
        return '这段看起来已经是完整口播稿了，切到「口播稿」可以直接开拍';
      }
      if (this.inputMode === 'script' && text.length > 0 && text.length < LOOKS_LIKE_BRIEF) {
        return '这更像是需求描述，切到「创作需求」可以让模型先写稿';
      }
      return 'Enter 发送 · Shift + Enter 换行';
    }

    async send() {
      const text = this.el.composerInput.value.trim();
      if (!text || this.running || this.busy) return;
      if (!this.modelReady) { toast(this.modelBlocked || '模型尚未就绪', 'warn'); return; }

      const mode = this.mode;
      this.el.composerInput.value = '';
      this._autoGrow();

      if (mode === 'edit') {
        this.stream.userMessage(text);
        await this._sendEdit(text);
      } else if (mode === 'brief') {
        await this._startFromBrief(text);
      } else {
        await this._startFromScript(text);
      }
      this._syncComposer();
    }

    /** Redraw the input chapter, which the socket cannot replay for us. */
    _seedStream() {
      const s = this.session;
      if (!s) return;
      if (s.mode === 'brief') {
        this.stream.userMessage(s.brief);
        this.stream.instantStep('input', '读取创作需求', inputStep.brief());
        if (!s.script) return;
        const step = this.stream.instantStep(
          'input', '撰写口播稿', inputStep.draft(s.draft), { pro: true });
        step.attachScript(s.script, (s.draft || {}).char_count);
      } else if (s.script) {
        this.stream.userMessage(s.script);
        this.stream.instantStep('input', '读取创作输入', inputStep.script(s.script));
      }
    }

    // --- mode 1: the user pasted a finished narration script ---
    async _startFromScript(script) {
      this._resetStage();
      this.session = { mode: 'script', brief: '', script: script, draft: null };
      this.currentInput = script;
      this.stream.userMessage(script);
      this.stream.instantStep('input', '读取创作输入', inputStep.script(script));
      await this._launch(script);
    }

    // --- mode 2: the user described the film; draft the script first ---
    async _startFromBrief(brief) {
      this._resetStage();
      this.session = { mode: 'brief', brief: brief, script: '', draft: null };
      this.currentInput = '';
      this.stream.userMessage(brief);
      this.stream.instantStep('input', '读取创作需求', inputStep.brief());

      const step = this.stream.beginStep('input', '撰写口播稿', {
        pro: true, placeholder: '正在根据需求撰写有头有尾的口播稿…',
      });
      this.busy = true;
      this._syncComposer();

      let data;
      try {
        data = await api.post('/api/script/draft', { brief: brief });
      } catch (err) {
        step.finish({ tone: 'err', text: '写稿失败：' + err.message });
        this.stream.openStep = null;
        toast('生成口播稿失败: ' + err.message, 'err');
        return;
      } finally {
        this.busy = false;
        this.stream.openStep = null;
      }

      step.finish(inputStep.draft(data));
      step.attachScript(data.script, data.char_count);
      this.session.script = data.script;
      this.session.draft = data;
      this.currentInput = data.script;
      await this._launch(data.script);
    }

    /** Interrupt now, rather than at the next step boundary. */
    async stopRun() {
      if (!this.runId || !this.running) return;
      this.el.btnStop.disabled = true;
      this.stream.note('正在中断本次执行…');
      try {
        await api.post(`/api/runs/${this.runId}/cancel?force=true`, null);
        toast('已中断', 'warn');
      } catch (err) {
        toast('中断失败: ' + err.message, 'err');
      } finally {
        this.el.btnStop.disabled = false;
      }
    }

    async _launch(script) {
      try {
        const data = await api.post('/api/runs', { input_text: script });
        this.attachRun(data.run_id);
        this.loadRuns();
      } catch (err) {
        this.stream.botMessage('启动失败：' + err.message, { cls: 'reject', tag: '错误' });
        toast('启动失败: ' + err.message, 'err');
      }
    }

    async _sendEdit(text) {
      this.busy = true;
      this._syncComposer();
      const thinking = this.stream.botMessage('正在定位要修改的地方…', { cls: 'muted' });
      try {
        const data = await api.post('/api/project/chat', {
          project_dir: this.projectDir, message: text,
        });
        thinking.remove();
        this._renderChatReply(data);
      } catch (err) {
        thinking.remove();
        this.stream.botMessage('出错了：' + err.message, { cls: 'reject', tag: '错误' });
      } finally {
        this.busy = false;
        this._syncComposer();
      }
    }

    _renderChatReply(data) {
      const intent = data.intent || 'clarify';
      const cls = intent === 'reject' ? 'reject' : (intent === 'edit' ? '' : 'muted');
      const bubble = this.stream.botMessage(data.reply || '', {
        cls, tag: INTENT_LABEL[intent] || intent,
      });

      const ops = data.ops || [];
      if (ops.length) {
        const box = el('div', 'msg-ops');
        ops.forEach((op) => box.appendChild(el('div', null, this._describeOp(op))));
        bubble.appendChild(box);
      }
      if (data.project) this._applyProjectPayload(data.project, { skipVideo: !!data.regen_job_id });

      if (data.applied) {
        if (data.regen_job_id) {
          this.stream.note(`修改已生效，正在重渲染 ${(data.regen_steps || []).length} 个步骤`);
          this.attachRun(data.regen_job_id);
        } else {
          toast('修改已保存（模型未就绪，需手动重渲染）', 'warn');
        }
        return;
      }

      const needsConfirm = data.needs_confirm || [];
      if (!ops.length) return;

      const row = el('div', 'msg-actions');
      if (needsConfirm.length) {
        const ok = el('button', 'primary-btn', '确认执行');
        const skip = el('button', 'ghost-btn', '不改了');
        ok.onclick = () => this._confirm(ops, needsConfirm, async () => {
          ok.disabled = true; skip.disabled = true;
          await this._applyOps(ops);
        });
        skip.onclick = () => { ok.disabled = true; skip.disabled = true; };
        row.append(ok, skip);
      } else {
        const apply = el('button', 'primary-btn', '应用');
        apply.onclick = async () => { apply.disabled = true; await this._applyOps(ops); };
        row.appendChild(apply);
      }
      bubble.appendChild(row);
    }

    _confirm(ops, needsConfirm, onOk) {
      this.el.confirmText.textContent =
        `以下 ${needsConfirm.length} 处属于高风险操作（会改变成片结构或删除内容），确认后才会执行：`;
      this.el.confirmOps.textContent = ops.map((o) => this._describeOp(o)).join('\n');
      this.el.confirmModal.hidden = false;
      this.el.confirmOk.onclick = async () => {
        this.el.confirmModal.hidden = true;
        await onOk();
      };
    }

    async _applyOps(ops) {
      try {
        const data = await api.post('/api/project/chat', {
          project_dir: this.projectDir, apply: true, ops: ops,
        });
        if (data.project) this._applyProjectPayload(data.project, { skipVideo: !!data.regen_job_id });
        const msg = (data.results || []).map((r) => r.message).filter(Boolean).join('；');
        this.stream.note('已应用：' + (msg || '完成'));
        if (data.regen_job_id) this.attachRun(data.regen_job_id);
        else toast('修改已保存', 'ok');
      } catch (err) {
        this.stream.botMessage('应用失败：' + err.message, { cls: 'reject', tag: '错误' });
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

    // ---------------- run lifecycle ----------------
    newSession() {
      if (this.running && !confirm('当前还有任务在运行，仍要开始新的创作吗？')) return;
      if (this.ws) { this.ws.onclose = null; this.ws.close(); this.ws = null; }
      this.runId = null;
      this.running = false;
      this.projectDir = '';
      this.projectTitle = '';
      this.currentInput = '';
      this.session = null;
      this._replaySeen = false;
      clearInterval(this.timerHandle);
      this.el.runTimer.hidden = true;
      this.stream.reset();
      this._resetStage();
      this._updateConvoStat();
      this._syncComposer();
      this._rememberSession();
      this.el.composerInput.focus();
    }

    attachRun(runId) {
      this.runId = runId;
      this._replaySeen = false;
      this.running = true;
      this.startedAt = Date.now();
      this.el.runTimer.hidden = false;
      this._startTimer();
      this._syncComposer();
      this._connect(runId);
    }


    _startTimer() {
      clearInterval(this.timerHandle);
      this.timerHandle = setInterval(() => {
        const s = Math.floor((Date.now() - this.startedAt) / 1000);
        this.el.runTimer.textContent =
          String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
      }, 1000);
    }

    _resetStage() {
      this.logs = { console: [], raw: [] };
      this._renderLogs();
      this.el.player.removeAttribute('src');
      this.el.player.hidden = true;
      this.el.playerEmpty.hidden = false;
      this._videoBase = '';
      this._setOverall(0);
      this.el.phaseLabel.textContent = '待启动';
      ['statDuration', 'statShots', 'statSpotlight', 'statSteps'].forEach((k, i) => {
        this.el[k].textContent = i === 0 ? '0:00' : '0';
      });
      this.timeline.update({ track_order: this.timeline.order, clips: {}, stats: {}, total_duration: 0 });
      this.rail.el.innerHTML = '';
      this.rail.built = '';
      this.el.railEmpty.hidden = false;
      this.el.railNote.textContent = '';
      this.el.timelineNote.textContent = '';
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
        if (this.running && this.runId === runId && this.wsRetry < 5) {
          this.wsRetry += 1;
          setTimeout(() => this._connect(runId), 1000 * this.wsRetry);
        }
      };
    }

    onEvent(ev) {
      switch (ev.type) {
        case 'replay_start':
          // Every (re)connect replays the whole run. The first replay fills an
          // empty column; later ones would append a second copy, so rebuild.
          // Edit and export jobs are stitched into an existing conversation, so
          // a rebuild would throw away history the socket cannot replay.
          if (this._replaySeen && !/^(chat_|edit_|export_)/.test(this.runId || '')) {
            this.stream.reset();
            this.logs = { console: [], raw: [] };
            this._renderLogs();
            this._seedStream();
          }
          this._replaySeen = true;
          break;

        case 'progress':
          this._onProgress(ev.progress);
          break;

        case 'stage_update':
          this.el.phaseLabel.textContent =
            ev.message || STAGE_LABEL[ev.stage] || ev.stage || '';
          if (typeof ev.progress_pct === 'number') this._setOverall(ev.progress_pct);
          break;

        case 'thinking':
          if (typeof ev.step === 'number') this.el.statSteps.textContent = ev.step;
          if (ev.raw_output) this._log('raw', 'info', `[step ${ev.step || '-'}] ${ev.raw_output}`);
          this.stream.addThought(ev.thinking || ev.summary || '');
          if (ev.parse_ok === false) this._log('console', 'warn', `step ${ev.step}: 输出解析失败`);
          break;

        case 'thinking_delta':
          if (typeof ev.step === 'number') this.el.statSteps.textContent = ev.step;
          this.stream.addThoughtDelta(ev.delta || ev.text || '');
          break;

        case 'tool_call': {
          const meta = TOOL_META[ev.tool_name] || { phase: 'other', title: ev.tool_name };
          if (typeof ev.step === 'number') this.el.statSteps.textContent = ev.step;
          this.stream.beginStep(meta.phase, meta.title, { pro: meta.pro });
          this._updateConvoStat();
          break;
        }

        case 'tool_result': {
          const meta = TOOL_META[ev.tool_name] || { phase: 'other', title: ev.tool_name };
          const result = summarise(ev.tool_name, ev.status, ev.observation);
          // Regeneration jobs emit results without a preceding call.
          if (!this.stream.openStep) this.stream.beginStep(meta.phase, meta.title, { pro: meta.pro });
          this.stream.openStep.finish(result);
          this.stream.openStep = null;
          this._updateConvoStat();
          break;
        }

        case 'console_log':
          this._log('console', ev.level || 'info', ev.text || '', ev.timestamp);
          break;

        case 'raw_output':
          this._log('raw', 'info', `[step ${ev.step || '-'}] ${ev.text || ''}`, ev.timestamp);
          break;

        case 'error':
          this._log('console', 'stderr', `[${ev.stage || '?'}] ${ev.message || ''}`);
          if (this.stream.openStep) {
            this.stream.openStep.finish({ tone: 'err', text: ev.message || '执行出错。' });
            this.stream.openStep = null;
          } else {
            this.stream.botMessage(ev.message || '执行出错。', { cls: 'reject', tag: '错误' });
          }
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

        case 'run_paused': this._log('console', 'warn', '已暂停'); break;
        case 'run_resumed': this._log('console', 'system', '已继续'); break;

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
      this.el.statDuration.textContent = fmtTime(p.total_duration || 0);
      const stats = p.stats || {};
      const shots = stats.shots || {};
      const spot = stats.spotlight || {};
      this.el.statShots.textContent = `${shots.done || 0}/${shots.total || 0}`;
      this.el.statSpotlight.textContent = `${spot.done || 0}/${spot.total || 0}`;
      const render = stats.render || {};
      this.el.railNote.textContent = render.total ? `${render.done || 0}/${render.total} 片段已渲染` : '';
      this.el.railEmpty.hidden = Object.keys(stats).length > 0;
      this.el.timelineNote.textContent = p.total_duration ? fmtTime(p.total_duration) : '';
      if (p.final_video_path) this._setVideo(p.final_video_path);
    }

    _setOverall(pct) {
      const v = Math.max(0, Math.min(100, Math.round(pct)));
      this.el.overallBar.style.width = v + '%';
      this.el.overallPct.textContent = v + '%';
    }

    async _onComplete(ev) {
      this.running = false;
      clearInterval(this.timerHandle);

      if (this.stream.openStep) {
        this.stream.openStep.finish({ tone: 'warn', text: '任务结束时这一步尚未返回结果。' });
        this.stream.openStep = null;
      }

      const label = RUN_STATUS_LABEL[ev.status] || ev.status;
      this.el.phaseLabel.textContent = label;
      if (ev.status === 'completed') this._setOverall(100);

      const result = ev.result || {};
      if (result.final_video_path) this._setVideo(result.final_video_path, { reload: true });
      if (ev.progress) this._onProgress(ev.progress);

      this.stream.settle();
      const isExport = this._exporting || /^export_/.test(this.runId || '');
      const isEdit = /^(chat_|edit_)/.test(this.runId || '');

      if (isExport) {
        this._exporting = false;
        this.el.btnExport.disabled = false;
        if (result.final_video_path) {
          this.stream.note('成片已合成完成，正在下载');
          this._download(result.final_video_path);
        } else {
          this.stream.note('合成' + label + '，成片未产出');
        }
        toast('导出' + label, ev.status === 'completed' ? 'ok' : 'warn');
      } else {
        this.stream.note(
          isEdit
            ? (ev.status === 'completed' ? '修改已渲染完毕，右侧是最新预览' : '重渲染' + label)
            : '本次执行' + label
        );
        toast((isEdit ? '重渲染' : '生成') + label,
              ev.status === 'completed' ? 'ok' : 'warn');
      }

      this.loadRuns();
      this._projects = null;   // new output landed; rescan next time it is shown

      // Staying on the same project turns the composer into the editor, which
      // is what makes "改一下第三个镜头" work without picking anything first.
      const dir = ev.project_dir || result.project_dir || '';
      const wasEditing = !!this.projectDir;
      if (dir && (this.cfg.features || {}).editor) {
        await this.loadProject(dir, { quiet: true });
        if (!wasEditing && this.projectDir) {
          this.stream.note('已进入编辑模式：直接说想改什么，或点「新建创作」重新开始');
        }
      }
      this._updateConvoStat();
      this._syncComposer();
      this._rememberSession();
    }

    _updateConvoStat() {
      const s = this.stream;
      this.el.convoStat.textContent = s.steps
        ? `${s.steps} 步执行 · ${s.outputs} 项产出`
        : '等待开始';
    }

    /* Play the per-shot clips back to back.
     *
     * Generation no longer concatenates anything (`render.defer_stitch`), so
     * there is no single file to point at. Chaining the shot mp4s on `ended`
     * previews the same cut without paying for a transcode after every edit;
     * what it cannot show is cross-shot transitions, which is why the bar says
     * 预览 until a real export exists. BGM is mixed in by ffmpeg at export, so
     * the clips are pure narration — `_bgmAttach` runs the tracks in a parallel
     * audio element to make the preview sound like the cut. */
    _setPlaylist(items, bgm) {
      const list = (items || []).filter((it) => it && it.path);
      this._playlist = list;
      this._playIndex = 0;
      this._videoBase = null;
      const player = this.el.player;

      if (!list.length) {
        player.hidden = true;
        player.removeAttribute('src');
        this._bgmAttach(null);
        this.el.playerEmpty.hidden = false;
        this.el.viewerBar.hidden = true;
        return;
      }

      player.onended = () => {
        if (this._playIndex + 1 >= (this._playlist || []).length) {
          this._bgmStop();
          return;
        }
        this._playIndex += 1;
        player.src = '/api/media?path=' +
          encodeURIComponent(this._playlist[this._playIndex].path);
        player.play().catch(() => {});
      };

      player.src = '/api/media?path=' + encodeURIComponent(list[0].path);
      player.hidden = false;
      player.load();
      this._bgmAttach(bgm);
      this.el.playerEmpty.hidden = true;
      this.el.viewerBar.hidden = false;
      const hasBgm = !!(this._bgmTracks && this._bgmTracks.length);
      this.el.viewerTag.textContent = `预览 · ${list.length} 镜连播`;
      this.el.viewerTag.title = hasBgm
        ? '逐镜头连播并叠加背景音乐，未含转场；导出后为完整成片'
        : '逐镜头连播，未含转场与背景音乐；导出后为完整成片';
    }

    /* ---- preview BGM ----
     * A second media element rather than a re-encode: the mix the user will
     * actually get is produced by ffmpeg at export (with ducking and trimmed to
     * length), and doing that on every edit is the cost the deferred stitch
     * exists to avoid. Here the music simply follows the video's play state. */
    _bgmAttach(bgm) {
      const tracks = ((bgm && bgm.tracks) || []).filter((t) => t && t.path);
      this._bgmTracks = tracks;
      this._bgmIndex = 0;

      if (!tracks.length) {
        this._bgmStop();
        return;
      }

      let audio = this._bgmAudio;
      if (!audio) {
        audio = document.createElement('audio');
        audio.preload = 'auto';
        audio.hidden = true;
        this.el.player.parentNode.appendChild(audio);
        this._bgmAudio = audio;

        const player = this.el.player;
        player.addEventListener('play', () => {
          if (this._bgmTracks && this._bgmTracks.length) {
            this._bgmAudio.play().catch(() => {});
          }
        });
        player.addEventListener('pause', () => this._bgmAudio.pause());
        // Muting the video should mute the whole preview, not just the voice.
        player.addEventListener('volumechange', () => {
          this._bgmAudio.muted = player.muted;
        });
        // One track rarely covers the whole cut, so walk the playlist and wrap.
        audio.addEventListener('ended', () => {
          const all = this._bgmTracks || [];
          if (!all.length) return;
          this._bgmIndex = (this._bgmIndex + 1) % all.length;
          audio.src = '/api/media?path=' + encodeURIComponent(all[this._bgmIndex].path);
          audio.play().catch(() => {});
        });
      }

      const volume = typeof (bgm && bgm.volume) === 'number' ? bgm.volume : 0.5;
      audio.volume = Math.max(0, Math.min(1, volume));
      audio.muted = this.el.player.muted;
      const src = '/api/media?path=' + encodeURIComponent(tracks[0].path);
      if (audio.getAttribute('src') !== src) {
        audio.setAttribute('src', src);
        audio.load();
      }
      if (!this.el.player.paused && !this.el.player.ended) audio.play().catch(() => {});
    }

    _bgmStop() {
      if (!this._bgmAudio) return;
      this._bgmAudio.pause();
      try { this._bgmAudio.currentTime = 0; } catch (e) { /* not seekable yet */ }
    }

    /** Keep the music where the cut is when the user jumps between shots. */
    _bgmSeekTo(index) {
      const audio = this._bgmAudio;
      const tracks = this._bgmTracks || [];
      if (!audio || !tracks.length) return;
      let offset = 0;
      (this._playlist || []).slice(0, index).forEach((it) => {
        offset += Number(it.duration_sec) || 0;
      });
      const span = audio.duration;
      if (offset > 0 && span && isFinite(span)) {
        try { audio.currentTime = offset % span; } catch (e) { /* not seekable yet */ }
      }
    }

    _setVideo(path, opts) {
      if (!path) return;
      this._playlist = null;
      this.el.player.onended = null;
      // An exported cut already carries the mixed BGM; a second track playing
      // over it would double the music.
      this._bgmTracks = [];
      this._bgmStop();
      // The final cut always lands on the same filename, so an edited version
      // is byte-identical in URL terms — without the cache-buster the browser
      // keeps serving the previous file.
      const base = '/api/media?path=' + encodeURIComponent(path);
      const reload = !!(opts && opts.reload);
      if (this._videoBase === base && !reload) return;
      this._videoBase = base;
      const wasPlaying = !this.el.player.paused && !this.el.player.ended;
      this.el.player.setAttribute('src', reload ? base + '&v=' + Date.now() : base);
      this.el.player.hidden = false;
      this.el.player.load();
      if (wasPlaying) this.el.player.play().catch(() => {});
      this.el.playerEmpty.hidden = true;
      this.el.viewerBar.hidden = false;
      this.el.viewerTag.textContent = '成片';
      this.el.viewerTag.title = '';
      this.el.btnExport.hidden = true;
    }

    // ---------------- logs ----------------
    _log(view, level, text, timestamp) {
      const entry = {
        level, text,
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
      const line = el('div', 'log-line ' + entry.level);
      line.appendChild(el('span', 'log-time', entry.time));
      line.appendChild(el('span', 'log-text', entry.text));
      body.appendChild(line);
      if (atBottom) body.scrollTop = body.scrollHeight;
    }

    _renderLogs() {
      this.el.consoleBody.innerHTML = '';
      this.logs[this.logView].forEach((e) => this._appendLogLine(e));
      this.el.consoleCount.textContent = this.logs[this.logView].length;
      this.el.consoleBody.scrollTop = this.el.consoleBody.scrollHeight;
    }

    // ---------------- library ----------------
    openDrawer() {
      this.el.drawer.hidden = false;
      this.el.scrim.hidden = false;
      this.loadRuns();
      this.loadProjects();
    }

    closeDrawer() {
      this.el.drawer.hidden = true;
      this.el.scrim.hidden = true;
    }

    async loadRuns() {
      try {
        const data = await api.get('/api/runs');
        const list = this.el.runList;
        list.innerHTML = '';
        if (!data.runs.length) {
          list.appendChild(el('p', 'list-empty', '暂无记录'));
          return;
        }
        data.runs.forEach((r) => {
          const item = el('div', 'list-item' + (r.id === this.runId ? ' active' : ''));
          item.appendChild(el('div', 'li-title', r.title));
          const meta = el('div', 'li-meta');
          const st = el('span', 'li-status ' + r.status, r.status);
          meta.append(st,
            el('span', null, (r.overall_pct || 0) + '%'),
            el('span', null, Math.round(r.elapsed_seconds) + 's'));
          item.appendChild(meta);
          item.onclick = () => this.openRun(r);
          if (!ACTIVE_STATUSES.has(r.status)) {
            item.appendChild(this._deleteBtn('删除这条记录', () => this.deleteRun(r)));
          }
          list.appendChild(item);
        });
      } catch (err) { /* transient */ }
    }

    /** Scanning the workspace is expensive, so results are cached and only
     *  refetched when the drawer is opened after new output landed. */
    async loadProjects(opts) {
      opts = opts || {};
      if (this._projectsLoading) return;
      if (this._projects && !opts.force) { this._renderProjects(this._projects); return; }

      this._projectsLoading = true;
      const list = this.el.projectList;
      list.innerHTML = '';
      list.appendChild(el('p', 'list-empty', '正在扫描工作区…'));
      try {
        const data = await api.get('/api/projects');
        this._projects = data.projects || [];
        this._renderProjects(this._projects);
      } catch (err) {
        list.innerHTML = '';
        list.appendChild(el('p', 'list-empty', '读取失败：' + err.message));
      } finally {
        this._projectsLoading = false;
      }
    }

    _renderProjects(projects) {
      const list = this.el.projectList;
      list.innerHTML = '';
      if (!projects.length) {
        list.appendChild(el('p', 'list-empty', '暂无成片'));
        return;
      }
      projects.slice(0, 40).forEach((p) => {
        const item = el('div', 'list-item' + (p.project_dir === this.projectDir ? ' active' : ''));
        item.appendChild(el('div', 'li-title', p.title || p.project_id));
        const meta = el('div', 'li-meta');
        meta.append(
          el('span', null, (p.shot_count || 0) + ' 镜'),
          el('span', null, p.has_video ? '有成片' : '无成片'),
          el('span', null, (p.modified || '').replace('T', ' ').slice(5, 16)));
        item.appendChild(meta);
        item.onclick = () => { this.closeDrawer(); this.loadProject(p.project_dir); };
        item.appendChild(this._deleteBtn('删除该成片', () => this.deleteProject(p)));
        list.appendChild(item);
      });
    }

    _deleteBtn(title, onDelete) {
      const btn = el('button', 'li-del', '✕');
      btn.title = title;
      btn.onclick = (ev) => { ev.stopPropagation(); onDelete(); };
      return btn;
    }

    async deleteRun(r) {
      if (!confirm(`删除记录「${r.title}」？只删除记录，不影响已生成的成片。`)) return;
      try {
        try { await api.del('/api/runs/' + r.id); }
        catch (e) {
          if (String(e.message).includes('Method Not Allowed')) await api.post('/api/runs/' + r.id + '/delete', {});
          else throw e;
        }
        if (this.runId === r.id) this.runId = null;
        this.loadRuns();
      } catch (err) {
        toast('删除失败: ' + err.message, 'err');
      }
    }

    async deleteProject(p) {
      const name = p.title || p.project_id;
      if (!confirm(`删除「${name}」？工程目录会被永久删除，无法恢复。`)) return;
      try {
        const q = '?dir=' + encodeURIComponent(p.project_dir);
        try { await api.del('/api/project' + q); }
        catch (e) {
          if (String(e.message).includes('Method Not Allowed')) await api.post('/api/project/delete' + q, {});
          else throw e;
        }
        if (this.projectDir === p.project_dir) { this.projectDir = ''; this._resetStage(); }
        toast(`已删除「${name}」`, 'ok');
        this.loadProjects({ force: true });
        this._syncComposer();
      } catch (err) {
        toast('删除失败: ' + err.message, 'err');
      }
    }

    async openRun(summary) {
      this.closeDrawer();
      try {
        const detail = await api.get('/api/runs/' + summary.id);
        this.stream.reset();
        this._resetStage();
        // A stored run keeps the script it ran, not the brief behind it, so a
        // reopened brief-mode run replays as if the script had been pasted.
        this.currentInput = detail.input_text || summary.title || '';
        this.session = { mode: 'script', brief: '', script: this.currentInput, draft: null };
        this._seedStream();
        this.runId = summary.id;
        this._replaySeen = false;
        if (detail.progress) this._onProgress(detail.progress);
        this.running = ACTIVE_STATUSES.has(detail.status);
        if (this.running) {
          this.startedAt = Date.now() - (detail.elapsed_seconds || 0) * 1000;
          this._startTimer();
          this.el.runTimer.hidden = false;
        }
        this._connect(summary.id);
        if (summary.project_dir) await this.loadProject(summary.project_dir, { quiet: true });
        this._syncComposer();
        this.loadRuns();
      } catch (err) {
        toast('无法打开该记录: ' + err.message, 'err');
      }
    }

    async loadProject(projectDir, opts) {
      opts = opts || {};
      if (!(this.cfg.features || {}).editor) {
        if (!opts.quiet) toast('编辑功能已在配置中关闭', 'warn');
        return;
      }
      const switching = this.projectDir !== projectDir;
      try {
        const payload = await api.get('/api/project', { dir: projectDir });
        this.projectDir = projectDir;
        this.projectTitle = payload.project_id || '';
        this._applyProjectPayload(payload);
        if (!opts.quiet && switching) {
          const title = payload.project_id || '';
          this.stream.reset();
          this.stream.note(
            (opts.restoring ? '继续上次的工程《' : '已载入成片《') + title + '》');
          // Order matters: the generation trajectory came first in time, then
          // the editing conversation on top of it.
          this.runId = payload.run_id || null;
          await this._restoreTrajectory(payload.run_id);
          await this._loadChatHistory(projectDir);
          this.stream.botMessage('可以直接说想改什么，例如「0:15 的画面换成城市夜景」「配乐压一下」。', { cls: 'muted' });
        }
        this._updateConvoStat();
        this._syncComposer();
        this._rememberSession();
        if (this._projects) this._renderProjects(this._projects);   // refresh the active row
      } catch (err) {
        if (!opts.quiet) toast('打开失败: ' + err.message, 'err');
        if (opts.restoring) {
          try { localStorage.removeItem(SESSION_STORAGE_KEY); } catch (e) { /* noop */ }
        }
      }
    }

    async _loadChatHistory(projectDir) {
      if (!(this.cfg.features || {}).chat_edit) return;
      try {
        const data = await api.get('/api/project/chat/history', { dir: projectDir });
        const msgs = data.messages || [];
        if (!msgs.length) return;
        msgs.forEach((m) => {
          if (m.role === 'user') this.stream.userMessage(m.content);
          else {
            const meta = m.meta || {};
            this.stream.botMessage(m.content, {
              cls: meta.intent === 'reject' ? 'reject' : (meta.intent === 'edit' ? '' : 'muted'),
            });
          }
        });
        this.stream.note('以上为历史对话');
      } catch (err) { /* no conversation yet */ }
    }

    _applyProjectPayload(payload, opts) {
      if (!payload) return;
      this._currentProjectPayload = payload;
      this.projectDir = payload.project_dir || this.projectDir;
      if (payload.progress) this._onProgress(payload.progress);
      if (payload.dirty) this.timeline.setDirty(payload.dirty);
      this._renderVersions(payload);
      if (opts && opts.skipVideo) return;

      if (payload.final_video_path) {
        this._setVideo(payload.final_video_path, { reload: true });
      } else if (payload.stitch_pending) {
        this._setPlaylist(payload.playlist, payload.preview_bgm);
      }
      this.el.btnExport.hidden = !payload.stitch_pending;
    }

    _renderVersions(payload) {
      if (!this.el.versionStrip || !this.el.versionList) return;
      const versions = (payload.versions || []).filter((v) => v && (v.final_video_path || v.project_dir));
      if (!versions.length) {
        this.el.versionStrip.hidden = true;
        this.el.versionList.innerHTML = '';
        return;
      }
      this.el.versionStrip.hidden = false;
      const list = this.el.versionList;
      list.innerHTML = '';

      const current = el('button', 'version-pill', '当前版本');
      current.type = 'button';
      current.onclick = () => {
        if (this._currentProjectPayload) this._applyProjectPayload(this._currentProjectPayload);
      };
      list.appendChild(current);

      versions.slice(0, 8).forEach((v, idx) => {
        const label = (v.modified || v.name || '').replace('T', ' ').slice(5, 16) || `版本 ${idx + 1}`;
        const btn = el('button', 'version-pill', label);
        btn.type = 'button';
        btn.title = v.name || '历史版本';
        btn.onclick = () => {
          if (v.final_video_path) {
            this._setVideo(v.final_video_path, { reload: true });
            this.el.viewerTag.textContent = '历史效果';
            this.el.viewerTag.title = '正在预览一次编辑前保存的历史效果';
            toast('已切到历史效果预览', 'ok');
          } else if (v.project_dir) {
            this.loadProject(v.project_dir, { quiet: true });
          }
        };
        list.appendChild(btn);
      });
    }

    /** Run the real concat + BGM mix, then hand the file to the browser. */
    async exportVideo() {
      if (!this.projectDir || this.running) return;
      this.el.btnExport.disabled = true;
      this.stream.userMessage('导出成片');
      this.stream.note('开始合成：逐镜头标准化 → 拼接 → 混入背景音乐，完成后自动下载');
      try {
        const data = await api.post('/api/project/export', { project_dir: this.projectDir });
        this._exporting = true;
        this.attachRun(data.job_id);
      } catch (err) {
        this.stream.botMessage('导出失败：' + err.message, { cls: 'reject', tag: '错误' });
        toast('导出失败: ' + err.message, 'err');
        this.el.btnExport.disabled = false;
      }
    }

    _download(path) {
      const a = document.createElement('a');
      a.href = '/api/media?path=' + encodeURIComponent(path) + '&v=' + Date.now();
      a.download = path.split('/').pop() || 'final_video.mp4';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    onClipSelected(track, clip) {
      if (this._playlist) {
        // Playlist mode: clip offsets are timeline-global, but each element
        // only knows its own clip, so seek by jumping to the shot.
        const idx = this._playlist.findIndex((it) => it.shot_id === clip.shot_id);
        if (idx >= 0) {
          this._playIndex = idx;
          this.el.player.src = '/api/media?path=' +
            encodeURIComponent(this._playlist[idx].path);
          this._bgmSeekTo(idx);
          this.el.player.play().catch(() => {});
        }
      } else if (this.el.player.getAttribute('src')) {
        this.el.player.currentTime = clip.start || 0;
      }
      if (clip.shot_id) this.timeline.select(clip.shot_id);
    }
  }

  document.addEventListener('DOMContentLoaded', () => { window.aicut = new App(); });
})();
