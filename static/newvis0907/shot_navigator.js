(function () {
  'use strict';

  const fmtTime = window.AICutFmtTime || function (sec) {
    const value = Number(sec);
    const safe = Number.isFinite(value) && value > 0 ? value : 0;
    return Math.floor(safe / 60) + ':' + String(Math.floor(safe % 60)).padStart(2, '0');
  };

  function num(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : (fallback || 0);
  }

  function text(value) {
    return value === undefined || value === null ? '' : String(value);
  }

  function el(tag, cls, txt) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (txt !== undefined && txt !== null) node.textContent = String(txt);
    return node;
  }

  function empty(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function shotCount(chapter) {
    return Array.isArray(chapter && chapter.model_shot_ids) ? chapter.model_shot_ids.length : 0;
  }

  function mediaUrl(path) {
    const value = text(path).trim();
    return value ? '/api/media?path=' + encodeURIComponent(value) : '';
  }

  function mediaPath(item) {
    return text(item && (item.thumbnail_path || item.poster_path || item.path || item.clip_path)).trim();
  }

  function isVideoPath(path) {
    return /\.(mp4|mov|mkv|webm|avi|m4v)(?:$|\?)/i.test(text(path));
  }

  function firstSubclip(shot) {
    const list = shot && Array.isArray(shot.subclips) ? shot.subclips : [];
    return list[0] || null;
  }

  class AICutShotNavigator {
    constructor(root, options) {
      this.root = root || null;
      this.options = options || {};
      this.payload = null;
      this.hierarchy = null;
      this.quality = null;
      this.chapters = [];
      this.shots = [];
      this.chapterById = {};
      this.shotById = {};
      this.subclipToShot = {};
      this.selectedShotId = '';
      this.selectedSubclipId = '';
      this.activeChapterId = '';
      this.visible = false;
      this.qualityState = 'ready';
      this.qualityStateDetail = '';
      this.qualityOpen = false;
      this.overriddenIssues = {};
      this._buildSkeleton();
      this.clear();
    }

    update(payloadOrHierarchy) {
      const payload = payloadOrHierarchy && payloadOrHierarchy.hierarchy
        ? payloadOrHierarchy
        : { hierarchy: payloadOrHierarchy || null };
      const hierarchy = payload && payload.hierarchy ? payload.hierarchy : null;

      this.payload = payload || null;
      this.hierarchy = hierarchy;
      // Rendered shot clips are the one media path that exists for every shot,
      // so they back the tiles whose footage metadata carries no usable file.
      this.renderedByShot = {};
      (Array.isArray(payload && payload.playlist) ? payload.playlist : []).forEach((item) => {
        const sid = text(item && item.shot_id);
        const path = text(item && item.path);
        if (sid && path) this.renderedByShot[sid] = path;
      });
      this.quality = payload && payload.quality ? payload.quality : null;
      if (this.qualityState !== 'loading') this.qualityState = 'ready';
      this.chapters = Array.isArray(hierarchy && hierarchy.chapters) ? hierarchy.chapters.slice() : [];
      this.shots = Array.isArray(hierarchy && hierarchy.model_shots) ? hierarchy.model_shots.slice() : [];
      this.chapterById = {};
      this.shotById = {};
      this.subclipToShot = {};
      this.chapters.forEach((chapter) => {
        this.chapterById[text(chapter.segment_id)] = chapter;
      });
      this.shots.forEach((shot) => {
        const shotId = text(shot.shot_id);
        this.shotById[shotId] = shot;
        (shot.subclips || []).forEach((subclip) => {
          this.subclipToShot[text(subclip.subclip_id)] = shotId;
        });
      });

      this.visible = this.shots.length > 0;
      if (!this.visible) {
        this.clear();
        return this;
      }
      if (!this.shotById[this.selectedShotId]) {
        const preferred = this.shots.find((shot) => {
          const status = text(shot && shot.quality_status);
          return status && status !== 'ok' && status !== 'unknown';
        });
        this.selectedShotId = text((preferred || this.shots[0]) && (preferred || this.shots[0]).shot_id);
      }
      this._syncChapter();
      this._normalizeSelectedSubclip();
      this._render();
      this._applyVisibility();
      return this;
    }

    selectShot(shotId, options) {
      const opts = options || {};
      const id = text(shotId);
      const shot = this.shotById[id];
      if (!shot) return this;
      this.selectedShotId = id;
      if (opts.subclipId !== undefined) {
        this.selectedSubclipId = text(opts.subclipId);
      }
      this._syncChapter();
      this._normalizeSelectedSubclip();
      this._render();
      if (!opts.silent && typeof this.options.onSeek === 'function') {
        this.options.onSeek({
          kind: 'shot',
          shot: shot,
          subclip: null,
          seconds: num(shot.start),
          options: opts,
        });
      }
      return this;
    }

    selectSubclip(subclipId) {
      const id = text(subclipId);
      const shotId = this.subclipToShot[id];
      if (!shotId) return this;
      const shot = this.shotById[shotId];
      const subclip = (shot && shot.subclips || []).find((item) => text(item.subclip_id) === id);
      if (!shot || !subclip) return this;
      this.selectedShotId = shotId;
      this.selectedSubclipId = id;
      this._syncChapter();
      this._render();
      if (typeof this.options.onSeek === 'function') {
        this.options.onSeek({
          kind: 'subclip',
          shot: shot,
          subclip: subclip,
          seconds: num(shot.start) + num(subclip.start),
          options: {},
        });
      }
      return this;
    }

    syncTime(seconds) {
      const sec = num(seconds, -1);
      if (sec < 0 || !this.shots.length) return this;
      const shot = this._shotForTime(sec);
      if (!shot) return this;
      const rel = Math.max(0, sec - num(shot.start));
      let subclipId = '';
      (shot.subclips || []).some((subclip, index, list) => {
        const start = num(subclip.start);
        const end = num(subclip.end);
        const last = index === list.length - 1;
        const hit = rel >= start && (rel < end || (last && rel <= end + 1e-6));
        if (hit) subclipId = text(subclip.subclip_id);
        return hit;
      });
      const shotId = text(shot.shot_id);
      if (shotId === this.selectedShotId && subclipId === this.selectedSubclipId) {
        return this;
      }
      this.selectShot(text(shot.shot_id), { silent: true, subclipId: subclipId });
      return this;
    }

    clear() {
      this.payload = null;
      this.hierarchy = null;
      this.quality = null;
      this.selectedShotId = '';
      this.selectedSubclipId = '';
      this.activeChapterId = '';
      this.visible = false;
      if (this.root) {
        this.root.dataset.selectedShotId = '';
        this.root.dataset.selectedSubclipId = '';
      }
      if (this.tabsEl) empty(this.tabsEl);
      if (this.shotStripEl) empty(this.shotStripEl);
      if (this.detailEl) empty(this.detailEl);
      this._applyVisibility();
      return this;
    }

    shotIdForSubclip(subclipId) {
      return this.subclipToShot[text(subclipId)] || '';
    }

    _applyVisibility() {
      if (!this.root) return;
      this.root.hidden = !this.visible;
    }

    _buildSkeleton() {
      if (!this.root) return;
      this.root.className = 'shot-navigator';
      this.root.setAttribute('role', 'region');
      this.root.setAttribute('aria-label', '镜头导航');
      this.tabsEl = el('div', 'shot-navigator-tabs');
      this.tabsEl.setAttribute('role', 'tablist');
      this.shotStripEl = el('div', 'shot-navigator-strip');
      this.detailEl = el('section', 'shot-navigator-detail');
      this.detailEl.setAttribute('data-selected-shot-detail', 'true');
      this.root.append(this.tabsEl, this.shotStripEl, this.detailEl);
    }

    _syncChapter() {
      const shot = this.shotById[this.selectedShotId];
      this.activeChapterId = shot ? text(shot.segment_id) : text(this.chapters[0] && this.chapters[0].segment_id);
    }

    _normalizeSelectedSubclip() {
      const shot = this.shotById[this.selectedShotId];
      const list = shot && Array.isArray(shot.subclips) ? shot.subclips : [];
      if (!list.length) {
        this.selectedSubclipId = '';
        return;
      }
      const found = list.some((subclip) => text(subclip.subclip_id) === this.selectedSubclipId);
      if (!found) this.selectedSubclipId = text(list[0].subclip_id);
    }

    _shotForTime(seconds) {
      return this.shots.find((shot, index) => {
        const start = num(shot.start);
        const end = num(shot.end);
        const last = index === this.shots.length - 1;
        return seconds >= start && (seconds < end || (last && seconds <= end + 1e-6));
      }) || null;
    }

    _render() {
      if (!this.root) return;
      this.root.dataset.selectedShotId = this.selectedShotId;
      this.root.dataset.selectedSubclipId = this.selectedSubclipId;
      this._renderTabs();
      this._renderShots();
      this._renderDetail();
    }

    _renderTabs() {
      empty(this.tabsEl);
      const frag = document.createDocumentFragment();
      this.chapters.forEach((chapter) => {
        const button = el('button', 'shot-nav-tab');
        const segmentId = text(chapter.segment_id);
        const active = segmentId === this.activeChapterId;
        button.type = 'button';
        button.setAttribute('role', 'tab');
        button.setAttribute('data-nav-tab', segmentId);
        button.setAttribute('aria-selected', active ? 'true' : 'false');
        if (active) button.classList.add('active');
        button.append(
          el('span', 'shot-nav-tab-index', String(num(chapter.display_index) || 0).padStart(2, '0')),
          el('span', 'shot-nav-tab-title', text(chapter.title) || segmentId),
          el('span', 'shot-nav-tab-meta', shotCount(chapter) + ' 镜')
        );
        button.onclick = () => {
          this.activeChapterId = segmentId;
          const firstShotId = text((chapter.model_shot_ids || [])[0]);
          if (firstShotId) this.selectShot(firstShotId);
          else this._render();
        };
        frag.appendChild(button);
      });
      this.tabsEl.appendChild(frag);
    }

    _renderShots() {
      empty(this.shotStripEl);
      const frag = document.createDocumentFragment();
      const rendered = {};
      this.chapters.forEach((chapter) => {
        const segmentId = text(chapter.segment_id);
        const group = el('section', 'shot-group');
        group.setAttribute('data-shot-group', segmentId);
        if (segmentId === this.activeChapterId) group.classList.add('active');
        const label = el('div', 'shot-group-label');
        label.append(
          el('span', 'shot-group-index', String(num(chapter.display_index) || 0).padStart(2, '0')),
          el('span', 'shot-group-title', text(chapter.title) || segmentId)
        );
        group.appendChild(label);
        (chapter.model_shot_ids || []).forEach((shotId) => {
          const id = text(shotId);
          const shot = this.shotById[id];
          if (!shot || rendered[id]) return;
          rendered[id] = true;
          group.appendChild(this._shotCard(shot));
          if (id === this.selectedShotId && (shot.subclips || []).length) {
            group.appendChild(this._subclipRail(shot));
          }
        });
        if (group.children.length > 1) frag.appendChild(group);
      });
      this.shots.forEach((shot) => {
        const id = text(shot.shot_id);
        if (rendered[id]) return;
        rendered[id] = true;
        frag.appendChild(this._shotCard(shot));
        if (id === this.selectedShotId && (shot.subclips || []).length) {
          frag.appendChild(this._subclipRail(shot));
        }
      });
      this.shotStripEl.appendChild(frag);
    }

    _shotCard(shot) {
      const shotId = text(shot.shot_id);
      const selected = shotId === this.selectedShotId;
      const card = el('section', 'shot-card');
      card.setAttribute('data-shot-id', shotId);
      card.setAttribute('data-segment-id', text(shot.segment_id));
      if (selected) card.classList.add('selected');
      const cover = this._shotCover(shot);
      if (cover) card.appendChild(cover);

      const header = el('button', 'shot-card-main');
      header.type = 'button';
      header.onclick = () => this.selectShot(shotId);
      const lead = firstSubclip(shot);
      const material = text(lead && lead.query) || text(shot.display_subtitle) || '未命名素材';
      header.append(
        el('span', 'shot-card-material', material),
        el('span', 'shot-card-meta', `${fmtTime(num(shot.duration))} · ${(shot.subclips || []).length} 段素材`)
      );
      header.querySelector('.shot-card-material').setAttribute('data-shot-material', 'true');
      card.appendChild(header);

      return card;
    }

    /* The subclips of the selected shot continue the same horizontal run
     * instead of growing the card downwards. */
    _subclipRail(shot) {
      const rail = el('div', 'shot-card-subclips');
      rail.setAttribute('data-subclip-list-open', 'true');
      rail.setAttribute('data-subclip-rail-shot', text(shot.shot_id));
      (shot.subclips || []).forEach((subclip) => {
        rail.appendChild(this._subclipCard(shot, subclip));
      });
      return rail;
    }

    _shotCover(shot) {
      const lead = firstSubclip(shot);
      const fromSubclips = (shot.subclips || []).map(mediaPath).find((value) => value);
      const media = mediaPath(lead) || fromSubclips
        || text((this.renderedByShot || {})[text(shot.shot_id)]);
      const wrap = el('div', 'shot-card-preview');
      wrap.append(
        el('span', 'shot-card-preview-badge', '分镜' + (text(shot.display_number) || text(shot.shot_id))),
        el('span', 'shot-card-preview-time', fmtTime(num(shot.duration)))
      );
      if (!media) {
        wrap.classList.add('empty');
        return wrap;
      }
      // A still frame reads the same for clips and images, so both source kinds
      // land on an identical tile instead of some cards showing nothing.
      const frame = document.createElement(isVideoPath(media) ? 'video' : 'img');
      frame.className = 'shot-card-preview-img';
      if (String(frame.tagName || '').toLowerCase() === 'video') {
        frame.muted = true;
        frame.playsInline = true;
        frame.preload = 'metadata';
      } else {
        frame.alt = (text(shot.display_number) || shot.shot_id || '镜头') + ' 预览';
      }
      // Without a media fragment browsers may hold an empty first paint for a
      // clip, which is what left some tiles blank.
      frame.setAttribute('src', mediaUrl(media) + (isVideoPath(media) ? '#t=0.5' : ''));
      frame.onerror = () => wrap.classList.add('empty');
      wrap.insertBefore(frame, wrap.firstChild);
      return wrap;
    }

    _subclipCard(shot, subclip) {
      const subclipId = text(subclip.subclip_id);
      const selected = subclipId === this.selectedSubclipId;
      const wrap = el('section', 'subclip-card' + (selected ? ' selected' : ''));
      wrap.setAttribute('data-subclip-id', subclipId);

      const jump = el('button', 'subclip-main');
      jump.type = 'button';
      jump.setAttribute('data-subclip-id', subclipId);
      jump.onclick = () => this.selectSubclip(subclipId);

      const rawPath = mediaPath(subclip) || text((this.renderedByShot || {})[text(shot.shot_id)]);
      const thumb = document.createElement(isVideoPath(rawPath) ? 'video' : 'img');
      thumb.className = 'subclip-thumb';
      thumb.alt = text(subclip.display_number) + ' 缩略图';
      if (thumb.tagName && String(thumb.tagName).toLowerCase() === 'video') {
        thumb.muted = true;
        thumb.playsInline = true;
        thumb.preload = 'metadata';
      }
      const src = mediaUrl(rawPath);
      if (src) {
        thumb.setAttribute('src', src);
      } else {
        thumb.classList.add('is-empty');
      }
      jump.appendChild(thumb);

      const body = el('div', 'subclip-copy');
      body.append(
        el('div', 'subclip-index', text(subclip.display_number)),
        el('div', 'subclip-query',
          text(subclip.query) || text(shot.display_subtitle) || text(shot.narration)
            || ('分镜 ' + (text(shot.display_number) || text(shot.shot_id)))),
        el('div', 'subclip-meta',
          `源 ${fmtTime(num(subclip.source_duration))} · 成片 ${fmtTime(num(subclip.final_duration))} · 速率 ${num(subclip.playback_rate, 1).toFixed(2)}x`)
      );
      jump.appendChild(body);
      wrap.appendChild(jump);

      const actions = el('div', 'subclip-actions');
      ['replace', 'duration', 'restore'].forEach((action) => {
        const button = el('button', 'mini-btn');
        button.type = 'button';
        button.textContent = action === 'replace' ? '替换'
          : action === 'duration' ? '时长' : '恢复';
        button.setAttribute('data-subclip-op', action);
        button.setAttribute('data-subclip-id', subclipId);
        button.onclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (typeof this.options.onSubclipAction === 'function') {
            this.options.onSubclipAction(action, shot, subclip);
          }
        };
        actions.appendChild(button);
      });
      wrap.appendChild(actions);
      return wrap;
    }

    _renderDetail() {
      empty(this.detailEl);
      const shot = this.shotById[this.selectedShotId];
      if (!shot) {
        this.detailEl.appendChild(el('p', 'shot-navigator-empty', '暂无镜头数据'));
        return;
      }
      const title = el('div', 'shot-detail-head');
      title.append(
        el('span', 'shot-detail-number', text(shot.display_number)),
        el('span', 'shot-detail-title', text(shot.display_subtitle) || text(shot.narration))
      );
      this.detailEl.appendChild(title);

      const list = el('div', 'shot-detail-grid');
      const currentSubclip = (shot.subclips || []).find((item) => text(item.subclip_id) === this.selectedSubclipId)
        || (shot.subclips || [])[0] || null;
      [
        ['口播', text(shot.narration) || '未提供'],
        ['字幕', text(shot.display_subtitle) || '未提供'],
        ['总时长', fmtTime(num(shot.duration))],
        ['素材段数', String((shot.subclips || []).length)],
        ['当前素材', text(currentSubclip && currentSubclip.query) || '未提供'],
        ['animation_type', text(shot.animation_type) || '未设置'],
      ].forEach(([label, value]) => {
        const row = el('div', 'shot-detail-item');
        row.append(el('span', 'shot-detail-label', label), el('span', 'shot-detail-value', value));
        list.appendChild(row);
      });
      this.detailEl.appendChild(list);
    }

    setQualityState(state, detail) {
      this.qualityState = text(state) || 'ready';
      this.qualityStateDetail = text(detail);
      return this;
    }

    _issueKey(issue) {
      return text(issue && issue.code) + '|' + text(issue && (issue.subclip_id || issue.shot_id || issue.field));
    }

    _allIssues() {
      return Array.isArray(this.quality && this.quality.issues) ? this.quality.issues : [];
    }

    _renderQuality() {}

    _autoFixableShots() {
      const seen = [];
      this._allIssues().forEach((issue) => {
        if (!issue || !issue.auto_fixable) return;
        if (this.overriddenIssues[this._issueKey(issue)]) return;
        const sid = text(issue.shot_id);
        if (sid && seen.indexOf(sid) === -1) seen.push(sid);
      });
      return seen;
    }

    _qualityIssueRow(issue) {
      const key = this._issueKey(issue);
      const overridden = !!this.overriddenIssues[key] || !!(issue && issue.user_override);
      const severity = text(issue && issue.severity) || 'warning';
      const shotId = text(issue && issue.shot_id);
      const row = el('div', 'quality-issue sev-' + severity + (overridden ? ' overridden' : ''));
      row.setAttribute('data-quality-issue', text(issue && issue.code));
      if (shotId) row.setAttribute('data-shot-id', shotId);
      if (issue && issue.subclip_id) row.setAttribute('data-subclip-id', text(issue.subclip_id));

      const locate = el('button', 'quality-issue-main');
      locate.type = 'button';
      locate.append(
        el('span', 'quality-issue-sev', severity),
        el('span', 'quality-issue-msg', text(issue && issue.message))
      );
      locate.onclick = () => {
        if (shotId && this.shotById[shotId]) this.selectShot(shotId);
        if (typeof this.options.onIssueSelect === 'function') {
          this.options.onIssueSelect(issue);
        }
      };
      row.appendChild(locate);

      // fix a single issue's shot (auto-fixable only, honours override).
      if (issue && issue.auto_fixable && !overridden && shotId) {
        const one = el('button', 'quality-issue-fix', '修复');
        one.type = 'button';
        one.setAttribute('data-issue-fix', text(issue.code));
        one.onclick = (event) => {
          event.stopPropagation();
          if (typeof this.options.onQualityFix === 'function') {
            this.options.onQualityFix({ scope: shotId, shot_ids: [shotId], issue: issue });
          }
        };
        row.appendChild(one);
      }

      // user override / restore-auto toggle.
      const toggle = el('button', 'quality-issue-override', overridden ? '恢复自动' : '忽略');
      toggle.type = 'button';
      toggle.setAttribute('data-issue-override', overridden ? 'restore' : 'override');
      toggle.onclick = (event) => {
        event.stopPropagation();
        const next = !overridden;
        this.overriddenIssues[key] = next;
        if (typeof this.options.onIssueOverride === 'function') {
          this.options.onIssueOverride(issue, next);
        }
        this._renderQuality();
      };
      row.appendChild(toggle);
      return row;
    }

    _qualityIssuesForShot(shot) {
      const shotId = text(shot && shot.shot_id);
      const issues = this._allIssues();
      return issues.filter((issue) => {
        const field = text(issue && (issue.shot_id || issue.field));
        return shotId && field.indexOf(shotId) === 0;
      });
    }
  }

  window.AICutShotNavigator = AICutShotNavigator;
})();
