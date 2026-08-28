/* Multi-track timeline.
 *
 * Renders whatever tracks the backend sends, in the order it sends them, so
 * adding or reordering `progress.tracks` in studio_config.yaml needs no
 * frontend change.
 *
 * Clips that exist but have no artifact yet render hollow, which is what makes
 * the view useful mid-run: you can see the shot plan being laid out before any
 * audio or video is produced, and watch lanes fill in.
 */
(function () {
  'use strict';

  const TRACK_COLORS = {
    segments: 'var(--t-segments)',
    shots: 'var(--t-shots)',
    narration: 'var(--t-narration)',
    subtitle: 'var(--t-subtitle)',
    footage: 'var(--t-footage)',
    spotlight: 'var(--t-spotlight)',
    bgm: 'var(--t-bgm)',
    render: 'var(--t-render)',
  };

  function fmtTime(sec) {
    if (!sec || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  class Timeline {
    constructor(opts) {
      this.labelsEl = document.getElementById('tlLabels');
      this.scrollEl = document.getElementById('tlScroll');
      this.canvasEl = document.getElementById('tlCanvas');
      this.rulerEl = document.getElementById('tlRuler');
      this.tracksEl = document.getElementById('tlTracks');
      this.emptyEl = document.getElementById('timelineEmpty');
      this.zoomLabel = document.getElementById('zoomLabel');

      this.onSelect = (opts && opts.onSelect) || function () {};

      this.pxPerSec = 60;
      this.minPx = 8;
      this.maxPx = 240;

      this.order = [];
      this.labels = {};
      this.clips = {};
      this.duration = 0;
      this.dirty = {};
      this.selected = null;

      this._bindZoom();
      window.addEventListener('resize', () => this.render());
    }

    _bindZoom() {
      const zi = document.getElementById('zoomIn');
      const zo = document.getElementById('zoomOut');
      const zf = document.getElementById('zoomFit');
      if (zi) zi.onclick = () => this._zoom(this.pxPerSec * 1.35);
      if (zo) zo.onclick = () => this._zoom(this.pxPerSec / 1.35);
      if (zf) zf.onclick = () => this.fit();
    }

    _zoom(next) {
      const vp = Math.max(1, this.scrollEl.clientWidth);
      const centerSec = (this.scrollEl.scrollLeft + vp / 2) / this.pxPerSec;
      this.pxPerSec = Math.max(this.minPx, Math.min(this.maxPx, next));
      this.render();
      // keep the viewport centred on the same instant across a zoom
      this.scrollEl.scrollLeft = Math.max(0, centerSec * this.pxPerSec - vp / 2);
    }

    fit() {
      if (this.duration > 0) {
        const avail = Math.max(1, this.scrollEl.clientWidth - 8);
        this.pxPerSec = Math.max(this.minPx, Math.min(this.maxPx, avail / this.duration));
      }
      this.render();
      this.scrollEl.scrollLeft = 0;
    }

    setTracks(trackDefs) {
      this.order = trackDefs.map((t) => t.key);
      trackDefs.forEach((t) => { this.labels[t.key] = t.label; });
      this._renderLabels();
    }

    update(progress) {
      if (!progress) return;
      if (Array.isArray(progress.track_order) && progress.track_order.length) {
        const changed = progress.track_order.join('|') !== this.order.join('|');
        this.order = progress.track_order;
        if (progress.labels) Object.assign(this.labels, progress.labels);
        if (changed) this._renderLabels();
      }
      this.clips = progress.clips || {};
      this.duration = progress.total_duration || 0;
      this.render();
    }

    setDirty(dirtyMap) {
      this.dirty = dirtyMap || {};
      this.render();
    }

    select(shotId) {
      this.selected = shotId;
      this.tracksEl.querySelectorAll('.tl-clip').forEach((el) => {
        el.classList.toggle('selected', el.dataset.shotId === shotId && !!shotId);
      });
    }

    _renderLabels() {
      this.labelsEl.innerHTML = '';
      const spacer = document.createElement('div');
      spacer.className = 'tl-label';
      this.labelsEl.appendChild(spacer);

      this.order.forEach((key) => {
        const el = document.createElement('div');
        el.className = 'tl-label';
        el.style.setProperty('--tc', TRACK_COLORS[key] || 'var(--text-3)');
        const sw = document.createElement('span');
        sw.className = 'tl-swatch';
        el.appendChild(sw);
        el.appendChild(document.createTextNode(this.labels[key] || key));
        this.labelsEl.appendChild(el);
      });
    }

    render() {
      const hasContent = this.order.some((k) => (this.clips[k] || []).length > 0);
      if (this.emptyEl) this.emptyEl.hidden = hasContent;

      const vp = Math.max(1, this.scrollEl.clientWidth);
      const width = Math.max(Math.ceil(this.duration * this.pxPerSec) + 2, vp);
      this.canvasEl.style.width = width + 'px';
      if (this.zoomLabel) this.zoomLabel.textContent = Math.round(this.pxPerSec) + ' px/s';

      this._renderRuler(width);
      this._renderTracks(width);

      // keep lane labels vertically aligned with lanes while scrolling
      this.scrollEl.onscroll = () => {
        this.labelsEl.style.transform = 'translateY(' + -this.scrollEl.scrollTop + 'px)';
      };
    }

    _renderRuler(width) {
      const step = this._tickStep();
      const frag = document.createDocumentFragment();
      for (let t = 0; t * this.pxPerSec <= width; t += step) {
        const tick = document.createElement('div');
        tick.className = 'tl-tick';
        tick.style.left = (t * this.pxPerSec) + 'px';
        const lbl = document.createElement('span');
        lbl.textContent = fmtTime(t);
        tick.appendChild(lbl);
        frag.appendChild(tick);
      }
      this.rulerEl.innerHTML = '';
      this.rulerEl.appendChild(frag);
    }

    _tickStep() {
      const targetPx = 68;
      const raw = targetPx / this.pxPerSec;
      const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
      for (const s of steps) if (s >= raw) return s;
      return 900;
    }

    _renderTracks(width) {
      const frag = document.createDocumentFragment();

      this.order.forEach((key) => {
        const lane = document.createElement('div');
        lane.className = 'tl-track';
        lane.dataset.track = key;
        lane.style.width = width + 'px';

        (this.clips[key] || []).forEach((clip, idx) => {
          lane.appendChild(this._clipEl(key, clip, idx));
        });
        frag.appendChild(lane);
      });

      this.tracksEl.innerHTML = '';
      this.tracksEl.appendChild(frag);
    }

    _clipEl(track, clip, idx) {
      const el = document.createElement('div');
      el.className = 'tl-clip';
      el.style.setProperty('--tc', TRACK_COLORS[track] || 'var(--text-3)');

      const start = clip.start || 0;
      const end = clip.end || start;
      el.style.left = (start * this.pxPerSec) + 'px';
      el.style.width = Math.max((end - start) * this.pxPerSec, 3) + 'px';

      if (clip.ready === false) el.classList.add('pending');

      const shotId = clip.shot_id || '';
      if (shotId) {
        el.dataset.shotId = shotId;
        if (this.dirty[shotId]) el.classList.add('dirty');
        if (this.selected === shotId) el.classList.add('selected');
      }
      el.dataset.track = track;
      el.dataset.index = idx;

      el.textContent = this._label(track, clip);
      el.title = this._tooltip(track, clip);

      el.onclick = () => this.onSelect(track, clip);
      return el;
    }

    _label(track, clip) {
      switch (track) {
        case 'segments': return (clip.title || clip.segment_id || '') + ' · ' + (clip.shot_count || 0) + '镜';
        case 'shots': return clip.title || clip.shot_id || '';
        case 'narration':
        case 'subtitle': return clip.text || clip.shot_id || '';
        case 'footage': return clip.ready ? '素材' : (clip.source === 'retrieval' ? '待检索' : '待生成');
        case 'spotlight': return clip.structure_id || (clip.ready ? '装饰' : '待装饰');
        case 'bgm': return clip.name || clip.track_id || 'BGM';
        case 'render': return clip.ready ? '已渲染' : '待渲染';
        default: return clip.shot_id || '';
      }
    }

    _tooltip(track, clip) {
      const lines = [];
      const id = clip.shot_id || clip.segment_id || clip.track_id || '';
      lines.push(`${id}  ${fmtTime(clip.start)} – ${fmtTime(clip.end)}  (${(clip.duration || 0).toFixed(2)}s)`);
      if (clip.title) lines.push(clip.title);
      if (clip.summary) lines.push(clip.summary);
      if (clip.text) lines.push('文字: ' + clip.text);
      if (clip.shot_ids && clip.shot_ids.length) lines.push('镜头: ' + clip.shot_ids.join(', '));
      if (track === 'spotlight') {
        const t = [clip.structure_id, clip.motion_id, clip.visual_id].filter(Boolean).join(' / ');
        if (t) lines.push('模板: ' + t);
        if (clip.slots) {
          Object.entries(clip.slots).forEach(([k, v]) => lines.push(`  ${k}: ${v}`));
        }
      }
      if (clip.clip_path) lines.push('素材: ' + clip.clip_path);
      if (clip.audio_path) lines.push('音频: ' + clip.audio_path);
      if (clip.rendered_path) lines.push('渲染: ' + clip.rendered_path);
      if (clip.ready === false) lines.push('(尚未生成)');
      return lines.join('\n');
    }
  }

  class Rail {
    constructor() {
      this.el = document.getElementById('trackRail');
      this.built = '';
    }

    update(progress) {
      if (!progress || !progress.stats) return;
      const order = progress.track_order || [];
      const labels = progress.labels || {};
      const key = order.join('|');

      if (key !== this.built) {
        this.el.innerHTML = '';
        order.forEach((t) => {
          const item = document.createElement('div');
          item.className = 'rail-item';
          item.dataset.track = t;
          item.style.setProperty('--tc', TRACK_COLORS[t] || 'var(--text-3)');
          item.innerHTML =
            '<div class="rail-top">' +
              '<span class="rail-label"></span>' +
              '<span class="rail-count"></span>' +
            '</div>' +
            '<div class="rail-bar"><div class="rail-fill"></div></div>';
          item.querySelector('.rail-label').textContent = labels[t] || t;
          this.el.appendChild(item);
        });
        this.built = key;
      }

      order.forEach((t) => {
        const item = this.el.querySelector(`.rail-item[data-track="${t}"]`);
        if (!item) return;
        const s = progress.stats[t] || { done: 0, total: 0, pct: 0 };
        item.querySelector('.rail-count').textContent = `${s.done}/${s.total}`;
        item.querySelector('.rail-fill').style.width = (s.pct || 0) + '%';
        item.classList.toggle('active', s.pct > 0 && s.pct < 100);
      });
    }
  }

  window.AICutTimeline = Timeline;
  window.AICutRail = Rail;
  window.AICutFmtTime = fmtTime;
})();
