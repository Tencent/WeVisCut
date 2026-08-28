/* Gold Template · Viewer — 三层自由重组 · 实时预览
 * 服务器根 = Gold_Template/，本页在 /viewer/。
 * 生成产物在 generated/ 下（structures / motions / visuals / templates）。 */
(function () {
  "use strict";

  var BASE = location.pathname.replace(/\/viewer\/?[^/]*$/, "/");
  if (BASE === location.pathname) BASE = "/";
  var ABS = location.origin + BASE;
  var GEN = "generated/"; // 生成产物前缀
  var SFX_CACHE = "?v=20260603-source1";

  // ── 字体 ──
  var FONTS = [
    ["NotoSansSC-Black.ttf",   "Noto Sans SC", 900, "truetype"],
    ["NotoSansSC-Bold.ttf",    "Noto Sans SC", 700, "truetype"],
    ["NotoSansSC-Medium.ttf",  "Noto Sans SC", 500, "truetype"],
    ["NotoSansSC-Regular.ttf", "Noto Sans SC", 400, "truetype"],
    ["NotoSerifSC-Bold.otf",   "Noto Serif SC", 700, "opentype"]
  ];
  var FONT_FACES = FONTS.map(function (f) {
    return '@font-face{font-family:"' + f[1] + '";font-weight:' + f[2] +
      ';font-style:normal;font-display:block;src:url("' + ABS + "fonts/" + f[0] +
      '") format("' + f[3] + '");}';
  }).join("\n");

  var RESET_CSS = [
    "*{margin:0;padding:0;box-sizing:border-box;}",
    "html,body{width:1920px;height:1080px;overflow:hidden;}",
    "#root{position:relative;width:100%;height:100%;}",
    "#scene{display:flex;align-items:center;justify-content:center;width:100%;height:100%;}",
    ".stage-panel{position:relative;overflow:hidden;width:calc(100% - 320px);height:calc(100% - 200px);" +
      "background:var(--stage-bg);border-radius:var(--radius-stage);box-shadow:var(--shadow-soft);" +
      "-webkit-backdrop-filter:blur(var(--stage-blur,0px));backdrop-filter:blur(var(--stage-blur,0px));}"
  ].join("\n");

  var HF_VAR_TYPES = ["string", "number", "color", "boolean", "enum"];

  // ── DOM ──
  var $ = function (id) { return document.getElementById(id); };
  var selS = $("selStructure"), selM = $("selMotion"), selV = $("selVisual");
  var selBgMode = $("selBgMode");
  var inpD = $("inpDuration"), inpBgVideo = $("inpBgVideo"), inpBgClipLen = $("inpBgClipLen");
  var inpDecorStart = $("inpDecorStart"), inpDecorDuration = $("inpDecorDuration");
  var slotFields = $("slotFields");
  var frameBox = $("frameBox"), frame = $("previewFrame"), emptyHint = $("emptyHint");
  var previewArea = $("previewArea"), statusEl = $("status"), comboId = $("comboId");
  var sfxbar = $("sfxbar"), sfxNone = $("sfxNone"), auditor = $("auditor");
  var compatWarn = $("compatWarn");
  frame.setAttribute("allow", "autoplay");

  var state = {
    schema: null,
    sfxReg: { sfx: [], track_index_base: 102 },
    cues: [],
    sfxTimers: [],
    sceneStart: 0,
    compatPairs: {},
    avail: { structures: {}, motions: {}, visuals: {} },
    structMeta: {},
    catName: {},
    muted: false,
    paused: false,
    applyTimer: null
  };

  function setStatus(msg, kind) {
    statusEl.textContent = msg;
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  function fetchText(path) {
    return fetch(BASE + path).then(function (r) {
      if (!r.ok) throw new Error(path + " → " + r.status);
      return r.text();
    });
  }
  function fetchJSON(path) { return fetchText(path).then(function (t) { return JSON.parse(t); }); }
  function exists(path) {
    return fetch(BASE + path, { method: "HEAD" }).then(function (r) { return r.ok; }).catch(function () { return false; });
  }
  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

  function extractStructure(raw) {
    var cssM = raw.match(/<style[^>]*data-hf=["']structure-css["'][^>]*>([\s\S]*?)<\/style>/i);
    var scriptM = raw.match(/<script[^>]*data-hf=["']structure-script["'][^>]*>([\s\S]*?)<\/script>/i);
    var cleaned = scriptM ? raw.replace(/<script[^>]*data-hf=["']structure-script["'][^>]*>[\s\S]*?<\/script>/i, "") : raw;
    var domM = cleaned.match(/<[^>]*data-hf=["']structure-dom["'][^>]*>[\s\S]*<\/[^>]+>/i);
    if (!domM) throw new Error("structure.html 缺少 data-hf=\"structure-dom\" 块");
    return { css: cssM ? cssM[1].trim() : "", dom: domM[0].trim(), script: scriptM ? scriptM[1].trim() : "" };
  }

  function computeVars(schema, userSlots) {
    var eff = {}; var slots = (schema && schema.slots) || {};
    Object.keys(slots).forEach(function (id) {
      var spec = slots[id];
      var raw = (userSlots && Object.prototype.hasOwnProperty.call(userSlots, id)) ? userSlots[id] : spec["default"];
      var t = spec.type || "string";
      if (HF_VAR_TYPES.indexOf(t) < 0) {
        eff[id] = (typeof raw === "string") ? raw : JSON.stringify(raw);
      } else {
        eff[id] = raw;
      }
    });
    return eff;
  }

  function defaultSlotsObj(schema) {
    var o = {}; var slots = (schema && schema.slots) || {};
    Object.keys(slots).forEach(function (id) { o[id] = slots[id]["default"]; });
    return o;
  }

  // ── 槽位表单 ──
  function renderSlotUI(schema) {
    slotFields.innerHTML = "";
    if (!schema || !schema.slots) return;
    var slots = schema.slots;
    Object.keys(slots).forEach(function (id) {
      var spec = slots[id];
      var isJson = (spec.type || "string") === "json" || HF_VAR_TYPES.indexOf(spec.type || "string") < 0;
      var wrap = document.createElement("div");
      wrap.className = "slot-field";

      var label = document.createElement("label");
      label.textContent = (spec.label || id);
      if (spec.max_chars) {
        var tag = document.createElement("span");
        tag.className = "slot-tag";
        tag.textContent = "\u2264" + spec.max_chars + "\u5B57";
        label.appendChild(tag);
      }
      wrap.appendChild(label);

      var input;
      if (isJson) {
        input = document.createElement("textarea");
        input.className = "slot-json";
        var def = spec["default"];
        input.value = (typeof def === "string") ? def : JSON.stringify(def, null, 2);
      } else {
        input = document.createElement("input");
        input.type = "text";
        input.placeholder = spec.label || id;
        input.value = (spec["default"] !== undefined && spec["default"] !== null) ? String(spec["default"]) : "";
      }
      input.dataset.slotId = id;
      input.dataset.slotType = isJson ? "json" : (spec.type || "string");
      input.addEventListener("input", scheduleApply);
      wrap.appendChild(input);
      slotFields.appendChild(wrap);
    });
  }

  function collectSlots() {
    var obj = {};
    var fields = slotFields.querySelectorAll("[data-slot-id]");
    for (var i = 0; i < fields.length; i++) {
      var el = fields[i];
      var id = el.dataset.slotId;
      if (el.dataset.slotType === "json") {
        try { obj[id] = JSON.parse(el.value); } catch (e) { obj[id] = el.value; }
      } else {
        obj[id] = el.value;
      }
    }
    return obj;
  }

  function resetSlotDefaults() {
    if (!state.schema || !state.schema.slots) return;
    var slots = state.schema.slots;
    var fields = slotFields.querySelectorAll("[data-slot-id]");
    for (var i = 0; i < fields.length; i++) {
      var el = fields[i];
      var spec = slots[el.dataset.slotId];
      if (!spec) continue;
      if (el.dataset.slotType === "json") {
        var def = spec["default"];
        el.value = (typeof def === "string") ? def : JSON.stringify(def, null, 2);
      } else {
        el.value = (spec["default"] !== undefined && spec["default"] !== null) ? String(spec["default"]) : "";
      }
    }
    scheduleApply();
  }

  function renderFootageChain(src, total, clipLen) {
    var safeTotal = Math.max(0.1, total);
    var segLen = clipLen > 0 ? clipLen : safeTotal;
    var count = clipLen > 0 ? Math.max(1, Math.ceil(safeTotal / segLen)) : 1;
    var style = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;";
    var tags = [];
    for (var i = 0; i < count; i++) {
      var start = +(i * segLen).toFixed(3);
      var dur = +Math.min(segLen, safeTotal - start).toFixed(3);
      if (dur <= 0) break;
      tags.push('<video id="footage-' + i + '" data-start="' + start + '" data-duration="' + dur +
        '" data-track-index="0" src="' + ABS + src + '" muted playsinline style="' + style + '"></video>');
    }
    return (
      '<div class="bg-footage" style="position:absolute;inset:0;z-index:0;overflow:hidden;background:#000;">' +
      tags.join("\n") +
      '<div class="footage-scrim" style="position:absolute;inset:0;z-index:1;pointer-events:none;' +
      'background:linear-gradient(180deg,rgba(0,0,0,.10) 0%,rgba(0,0,0,.28) 100%);"></div></div>'
    );
  }

  function runtimeJS(motionId, totalDuration, sceneStart, sceneDuration, isFootage, skipMotion) {
    var lines = [
      'var V=window.__hyperframes.getVariables();',
      'document.querySelectorAll("[data-slot]").forEach(function(el){var k=el.getAttribute("data-slot");if(V[k]!==undefined&&V[k]!==null)el.textContent=V[k];});',
      'var root=document.getElementById("root");var scene=document.getElementById("scene");',
      'var tl=gsap.timeline({paused:true});',
      'if(typeof window.__hf_structure_init==="function"){window.__hf_structure_init(tl,{gsap:gsap,root:root,duration:' + sceneDuration + '});}'
    ];
    if (!skipMotion) {
      if (isFootage) {
        var fadeIn = Math.min(0.4, sceneDuration * 0.15);
        var fadeOut = Math.min(0.4, sceneDuration * 0.15);
        var fadeOutAt = Math.max(sceneStart + fadeIn + 0.1, sceneStart + sceneDuration - fadeOut);
        lines.push('scene.style.opacity=0;');
        lines.push('tl.to(scene,{opacity:1,duration:' + fadeIn + ',ease:"power2.out"},' + sceneStart + ');');
        lines.push('tl.to(scene,{opacity:0,duration:' + fadeOut + ',ease:"power2.in"},' + fadeOutAt + ');');
        lines.push(
          'if(window.__hf_motions&&window.__hf_motions["' + motionId + '"]){' +
          'var decor=gsap.timeline();' +
          'window.__hf_motions["' + motionId + '"](decor,{gsap:gsap,root:root,duration:' + sceneDuration + ',isFinal:false});' +
          'tl.add(decor,' + sceneStart + ');}'
        );
      } else {
        lines.push(
          'if(window.__hf_motions&&window.__hf_motions["' + motionId + '"]){' +
          'window.__hf_motions["' + motionId + '"](tl,{gsap:gsap,root:root,duration:' + sceneDuration + ',isFinal:false});}'
        );
      }
    } else if (isFootage) {
      var fadeIn2 = Math.min(0.4, sceneDuration * 0.15);
      var fadeOut2 = Math.min(0.4, sceneDuration * 0.15);
      var fadeOutAt2 = Math.max(sceneStart + fadeIn2 + 0.1, sceneStart + sceneDuration - fadeOut2);
      lines.push('scene.style.opacity=0;');
      lines.push('tl.to(scene,{opacity:1,duration:' + fadeIn2 + ',ease:"power2.out"},' + sceneStart + ');');
      lines.push('tl.to(scene,{opacity:0,duration:' + fadeOut2 + ',ease:"power2.in"},' + fadeOutAt2 + ');');
    }
    lines.push('window.__timelines=window.__timelines||{};window.__timelines["root"]=tl;');
    return lines.join("");
  }

  function controllerJS(duration) {
    return [
      '(function(){var tl=window.__timelines&&window.__timelines.root;if(!tl){return;}',
      'var DUR=' + duration + ';var GAP=0.7;var TOTAL=DUR+GAP;',
      'var audios=[].slice.call(document.querySelectorAll("audio[data-start]"));',
      'var videos=[].slice.call(document.querySelectorAll("video[data-start]"));',
      'videos.forEach(function(v){v.muted=true;v.playsInline=true;v.style.opacity=0;});',
      'var paused=false,muted=false,base=performance.now(),lastT=0;',
      'function resetFlags(){audios.forEach(function(a){a._f=false;});}',
      'window.addEventListener("message",function(e){var d=e.data||{};',
      ' if(d.t==="pause"){paused=true;}',
      ' else if(d.t==="play"){paused=false;base=performance.now()-lastT*1000;}',
      ' else if(d.t==="replay"){base=performance.now();lastT=0;resetFlags();',
      '  audios.forEach(function(a){try{a.pause();a.currentTime=0;}catch(_){}});',
      '  videos.forEach(function(v){try{v.pause();v.currentTime=0;}catch(_){}});}',
      ' else if(d.t==="mute"){muted=d.v;audios.forEach(function(a){a.muted=muted;});}});',
      'function frame(now){',
      ' if(!paused){var t=((now-base)/1000)%TOTAL;',
      '  if(t<lastT){resetFlags();}lastT=t;',
      '  tl.time(Math.min(t,DUR));',
      '  videos.forEach(function(v){',
      '   var vs=parseFloat(v.getAttribute("data-start"))||0;',
      '   var vd=parseFloat(v.getAttribute("data-duration"))||0;',
      '   var on=t>=vs&&t<vs+vd;',
      '   v.style.opacity=on?1:0;',
      '   if(on){var want=t-vs;if(Math.abs((v.currentTime||0)-want)>0.08){try{v.currentTime=want;}catch(_){}}',
      '    try{var p=v.play();if(p&&p.catch){p.catch(function(){});}}catch(_){}}',
      '   else{try{v.pause();}catch(_){}}});',
      '  audios.forEach(function(a){var at=parseFloat(a.getAttribute("data-start"))||0;',
      '   if(!a._f&&t>=at&&t<=at+0.18){',
      '    try{a.currentTime=0;a.muted=muted;a.volume=parseFloat(a.getAttribute("data-volume"))||1;',
      '    var p=a.play();if(p&&p.catch){p.catch(function(){});}}catch(_){}a._f=true;}});',
      ' }requestAnimationFrame(frame);',
      '}requestAnimationFrame(frame);})();'
    ].join("");
  }

  function buildAudioTags(cues, sceneStart, sceneDuration) {
    var enabled = cues.filter(function (c) { return c.enabled; });
    if (!enabled.length) return "";
    var base = state.sfxReg.track_index_base || 102;
    var map = {}; (state.sfxReg.sfx || []).forEach(function (s) { map[s.id] = s; });
    return enabled.map(function (c, i) {
      var meta = map[c.sfx] || {};
      var file = meta.file || (c.sfx + ".mp3");
      var at = clamp(sceneStart + (parseFloat(c.at) || 0), sceneStart, sceneStart + sceneDuration);
      var vol = (c.volume != null) ? c.volume : (meta.default_volume != null ? meta.default_volume : 0.35);
      var dur = (c.dur != null) ? c.dur : (meta.default_dur != null ? meta.default_dur : 1.0);
      return '<audio id="sfx-' + i + '" data-start="' + at + '" data-duration="' + dur +
        '" data-track-index="' + (base + i) + '" src="' + ABS + "sfx/" + file +
        SFX_CACHE + '" data-volume="' + vol + '"></audio>';
    }).join("\n");
  }

  function sfxFileFor(cue) {
    var map = {}; (state.sfxReg.sfx || []).forEach(function (s) { map[s.id] = s; });
    var meta = map[cue.sfx] || {};
    return {
      src: ABS + "sfx/" + (meta.file || (cue.sfx + ".mp3")) + SFX_CACHE,
      volume: (cue.volume != null) ? cue.volume : (meta.default_volume != null ? meta.default_volume : 0.6)
    };
  }

  function clearParentSfxTimers() {
    state.sfxTimers.forEach(function (id) { clearTimeout(id); });
    state.sfxTimers = [];
  }

  function playCue(cue) {
    if (state.muted || !cue.enabled) return;
    var sfx = sfxFileFor(cue);
    var a = new Audio(sfx.src);
    a.volume = sfx.volume;
    a.play().catch(function () {});
  }

  function scheduleParentSfx() {
    clearParentSfxTimers();
    if (state.muted) return;
    state.cues.filter(function (c) { return c.enabled; }).forEach(function (cue) {
      var at = state.sceneStart + (parseFloat(cue.at) || 0);
      var timer = setTimeout(function () { playCue(cue); }, Math.max(0, at * 1000));
      state.sfxTimers.push(timer);
    });
  }

  function buildDoc(p) {
    var varsAttr = JSON.stringify(
      Object.keys((p.schema && p.schema.slots) || {}).map(function (id) {
        var spec = p.schema.slots[id];
        var t = spec.type || "string";
        var hf = HF_VAR_TYPES.indexOf(t) >= 0;
        return { id: id, type: hf ? t : "string", label: spec.label || id, "default": p.eff[id] };
      })
    ).replace(/'/g, "&#39;");

    var out = [];
    out.push('<!doctype html><html lang="zh-CN" data-composition-variables=\'', varsAttr, '\'>');
    out.push('<head><meta charset="UTF-8"/>');
    out.push('<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"><\/script>');
    out.push("<style>", FONT_FACES, "\n", RESET_CSS, "\n/* tokens */\n", p.tokensCss, "\n/* structure */\n", p.structureCss, "</style>");
    out.push("</head><body class=\"stage\">");
    out.push('<div id="root" data-composition-id="root" data-start="0" data-duration="', String(p.duration), '" data-width="1920" data-height="1080">');
    if (p.bgMode === "video") {
      out.push(renderFootageChain(p.bgVideo, p.duration, p.bgClipLen));
      out.push('<div id="scene" class="clip" data-start="', String(p.sceneStart), '" data-duration="', String(p.sceneDuration),
        '" data-track-index="2" style="position:relative;z-index:2;"><div class="stage-panel">');
    } else {
      out.push('<div class="bg-stage" data-anim-role="bg"></div>');
      out.push('<div id="scene" class="clip" data-start="0" data-duration="', String(p.duration), '" data-track-index="1"><div class="stage-panel">');
    }
    out.push(p.structureDom);
    out.push("</div></div>");
    out.push(p.audioTags || "");
    out.push("</div>");
    out.push("<script>window.__hyperframes={getVariables:function(){return ", JSON.stringify(p.eff), ";}};<\/script>");
    var hasStructScript = !!(p.structureScript);
    if (hasStructScript) { out.push("<script>", p.structureScript, "<\/script>"); }
    if (!hasStructScript) { out.push("<script>", p.motionJs, "<\/script>"); }
    out.push("<script>", runtimeJS(p.motionId, p.duration, p.sceneStart, p.sceneDuration, p.bgMode === "video", hasStructScript), "<\/script>");
    out.push("<script>", controllerJS(p.duration), "<\/script>");
    out.push("</body></html>");
    return out.join("");
  }

  // ── 兼容性提示 ──
  function updateCompatWarn() {
    var sid = selS.value, mid = selM.value;
    if (!sid || !mid) { compatWarn.style.display = "none"; return; }
    var key = sid + "::" + mid;
    if (state.compatPairs[key]) {
      compatWarn.style.display = "none";
    } else if (Object.keys(state.compatPairs).length > 0) {
      compatWarn.style.display = "";
      compatWarn.textContent = "⚠ 此 structure×motion 未经验证组合。dataviz motion 搭非匹配结构时图形区可能为空（不报错）。";
    } else {
      compatWarn.style.display = "none";
    }
  }

  // ── SFX bar ──
  function renderSfxBar() {
    [].slice.call(sfxbar.querySelectorAll(".sfx-chip")).forEach(function (n) { n.remove(); });
    if (!state.cues.length) { sfxNone.style.display = ""; return; }
    sfxNone.style.display = "none";
    state.cues.forEach(function (c) {
      var chip = document.createElement("span"); chip.className = "sfx-chip";
      var cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!c.enabled;
      cb.addEventListener("change", function () { c.enabled = cb.checked; scheduleApply(); });
      var name = document.createElement("span"); name.className = "name"; name.textContent = c.sfx;
      var meta = document.createElement("span"); meta.className = "meta"; meta.textContent = "@" + c.at + "s";
      var play = document.createElement("button"); play.className = "play sm"; play.textContent = "▶";
      play.addEventListener("click", function () { audition(c); });
      chip.appendChild(cb); chip.appendChild(name); chip.appendChild(meta); chip.appendChild(play);
      sfxbar.appendChild(chip);
    });
  }

  function audition(cue) {
    var sfx = sfxFileFor(cue);
    auditor.src = sfx.src;
    auditor.volume = sfx.volume;
    auditor.currentTime = 0;
    auditor.play().catch(function () {});
  }

  // ── debounced auto-apply ──
  function scheduleApply() {
    if (state.applyTimer) clearTimeout(state.applyTimer);
    state.applyTimer = setTimeout(apply, 300);
  }

  // ── 合成预览 ──
  function apply() {
    if (state.applyTimer) { clearTimeout(state.applyTimer); state.applyTimer = null; }
    clearParentSfxTimers();
    var sid = selS.value, mid = selM.value, vid = selV.value;
    if (!sid || !vid) return;
    var duration = parseFloat(inpD.value) || 6.5;
    var bgMode = selBgMode.value || "css";
    var sceneStart = bgMode === "video" ? (parseFloat(inpDecorStart.value) || 0) : 0;
    sceneStart = clamp(sceneStart, 0, duration);
    var sceneDuration = bgMode === "video" ? (parseFloat(inpDecorDuration.value) || (duration - sceneStart)) : duration;
    sceneDuration = clamp(sceneDuration, 0.1, Math.max(0.1, duration - sceneStart));
    var bgVideo = (inpBgVideo.value || "").trim();
    var bgClipLen = parseFloat(inpBgClipLen.value) || 0;
    var meta = state.structMeta[sid] || {};
    var parentMeta = (meta.variant_of && state.structMeta[meta.variant_of]) ? state.structMeta[meta.variant_of] : null;
    var tcode = meta.template_code || (parentMeta && parentMeta.template_code) || "";
    var name = parentMeta
      ? ((parentMeta.display_name || parentMeta.id) + " › " + (meta.variant_label || meta.display_name || meta.id))
      : (meta.display_name || sid);
    var tag = (tcode ? tcode + " · " : "") + name;
    if (meta.category && state.catName[meta.category]) tag += "（" + state.catName[meta.category] + "）";
    comboId.textContent = tag + " · " + sid + " × " + mid + " × " + vid + (bgMode === "video" ? " · video" : "");

    if (!state.avail.structures[sid] || !state.avail.visuals[vid]) {
      setStatus("所选层尚未生成（文件缺失）", "err"); return;
    }
    if (!state.avail.motions[mid] && !state.hasStructScript) {
      setStatus("所选运动层未生成且结构无内嵌动画", "err"); return;
    }
    if (bgMode === "video" && !bgVideo) {
      setStatus("视频背景路径为空", "err"); return;
    }
    var userSlots = collectSlots();
    state.sceneStart = sceneStart;

    setStatus("合成中…");
    var fetches = [
      fetchText(GEN + "structures/" + sid + "/structure.html"),
      (state.avail.motions[mid] ? fetchText(GEN + "motions/" + mid + "/motion.js") : Promise.resolve("")),
      fetchText(GEN + "visuals/" + vid + "/tokens.css")
    ];
    Promise.all(fetches).then(function (res) {
      var st = extractStructure(res[0]);
      var hasScript = !!(st.script);
      state.hasStructScript = hasScript;

      selM.disabled = hasScript;
      selM.style.opacity = hasScript ? "0.45" : "1";
      if (hasScript) {
        compatWarn.style.display = "";
        compatWarn.textContent = "\u2714 \u6b64\u7ed3\u6784\u542b\u5185\u5d4c\u52a8\u753b\uff0c\u5df2\u81ea\u52a8\u8df3\u8fc7\u901a\u7528\u8fd0\u52a8";
      } else {
        updateCompatWarn();
      }

      var eff = computeVars(state.schema, userSlots);
      var audioTags = buildAudioTags(state.cues, sceneStart, sceneDuration);
      var doc = buildDoc({
        schema: state.schema, eff: eff,
        tokensCss: res[2], structureCss: st.css, structureDom: st.dom,
        structureScript: st.script,
        motionJs: res[1], motionId: mid, duration: duration,
        sceneStart: sceneStart, sceneDuration: sceneDuration,
        bgMode: bgMode, bgVideo: bgVideo, bgClipLen: bgClipLen,
        audioTags: audioTags
      });
      frame.srcdoc = doc;
      emptyHint.style.display = "none";
      frameBox.style.display = "";
      state.paused = false;
      $("btnPlay").textContent = "\u23f8 \u6682\u505c";
      frame.onload = function () { fitScale(); sendFrame({ t: "mute", v: state.muted }); scheduleParentSfx(); };
      var label = hasScript ? sid + " \u00d7 " + vid + " \u00b7 \u5185\u5d4c\u52a8\u753b" : sid + "\u00d7" + mid + "\u00d7" + vid;
      setStatus("\u5c31\u7eea \u00b7 " + label, "ok");
    }).catch(function (err) {
      setStatus("\u5408\u6210\u5931\u8d25\uff1a" + err.message, "err");
    });
  }

  function sendFrame(msg) { try { frame.contentWindow.postMessage(msg, "*"); } catch (e) {} }

  function fitScale() {
    var pad = 32;
    var availW = previewArea.clientWidth - pad;
    var availH = previewArea.clientHeight - pad;
    var s = Math.max(0.08, Math.min(availW / 1920, availH / 1080));
    frame.style.transform = "scale(" + s + ")";
    frameBox.style.width = (1920 * s) + "px";
    frameBox.style.height = (1080 * s) + "px";
  }

  // ── 选择结构 → 载入 slot_schema ──
  function onStructureChange() {
    var sid = selS.value;
    updateCompatWarn();
    if (!state.avail.structures[sid]) { state.schema = null; slotFields.innerHTML = ""; return Promise.resolve(); }
    return fetchJSON(GEN + "structures/" + sid + "/slot_schema.json").then(function (schema) {
      state.schema = schema;
      renderSlotUI(schema);
      if (schema.duration_default) {
        inpD.value = schema.duration_default;
        inpBgClipLen.value = schema.duration_default;
        inpDecorDuration.value = schema.duration_default;
      }
      return loadCues();
    }).then(function () { renderSfxBar(); scheduleApply(); })
      .catch(function (e) { setStatus("读 slot_schema 失败：" + e.message, "err"); });
  }

  function loadCues() {
    var mid = selM.value, sid = selS.value;
    var jobs = [];
    if (state.avail.motions[mid]) {
      jobs.push(fetchJSON(GEN + "motions/" + mid + "/motion_card.json").catch(function () { return {}; }));
    } else {
      jobs.push(Promise.resolve({}));
    }
    if (state.avail.structures[sid]) {
      jobs.push(fetchJSON(GEN + "structures/" + sid + "/structure_card.json").catch(function () { return {}; }));
    } else {
      jobs.push(Promise.resolve({}));
    }
    return Promise.all(jobs).then(function (cards) {
      var cues = [];
      (cards[0].sfx_cues || []).forEach(function (c) { cues.push(c); });
      (cards[1].sfx_cues || []).forEach(function (c) { cues.push(c); });
      state.cues = cues.map(function (c) {
        return { at: c.at, sfx: c.sfx, volume: c.volume, dur: c.dur, enabled: true };
      });
    });
  }

  function onMotionChange() {
    updateCompatWarn();
    loadCues().then(function () { renderSfxBar(); scheduleApply(); });
  }

  function updateBgMode() {
    var video = selBgMode.value === "video";
    var f = $("bgVideoField");
    if (f) f.classList.toggle("hidden", !video);
    inpBgClipLen.disabled = !video;
    inpDecorStart.disabled = !video;
  }

  function buildOptions(sel, items, kind, labelFn) {
    sel.innerHTML = "";
    items.forEach(function (it) {
      var ok = state.avail[kind][it.id];
      var opt = document.createElement("option");
      opt.value = it.id;
      opt.textContent = labelFn(it) + (ok ? "" : " · 未生成");
      if (!ok) opt.disabled = true;
      sel.appendChild(opt);
    });
  }

  // 结构下拉：按 5 大类分组（optgroup）
  function buildStructureOptions(sel, structures, categories) {
    sel.innerHTML = "";
    var catList = (categories && categories.length) ? categories.slice() : [];
    if (!catList.length) {
      // 无分类元信息时回退到扁平
      buildOptions(sel, structures, "structures",
        function (s) { return s.id + " · " + (s.family || "narrative"); });
      return;
    }
    var order = {}; catList.forEach(function (c, i) { order[c.id] = i; });
    // 收集未归类的结构归入「其他」
    var hasOther = structures.some(function (s) { return !s.category || order[s.category] === undefined; });
    var groups = catList.map(function (c) {
      return { id: c.id, label: c.id + " · " + c.name, items: [] };
    });
    if (hasOther) groups.push({ id: "_other", label: "其他", items: [] });

    structures.forEach(function (s) {
      var gi = (order[s.category] !== undefined) ? order[s.category] : (groups.length - 1);
      groups[gi].items.push(s);
    });

    // 建立父→变体映射（variant_of 指向父结构 id）
    var byId = {}; structures.forEach(function (s) { byId[s.id] = s; });

    function optionFor(s, isVariant) {
      var ok = state.avail.structures[s.id];
      var opt = document.createElement("option");
      opt.value = s.id;
      var label;
      if (isVariant) {
        // 变体：缩进 + 变体中文名，归到父结构下
        label = "    └ " + (s.variant_label || s.display_name || s.id);
      } else {
        var prefix = s.template_code ? (s.template_code + " · ") : "";
        label = prefix + (s.display_name || s.id);
        if (s.bonus) label += " · bonus";
      }
      if (!ok) label += " · 未生成";
      opt.textContent = label;
      if (!ok) opt.disabled = true;
      return opt;
    }

    groups.forEach(function (g) {
      if (!g.items.length) return;
      // 只把“父结构/独立结构”放进顶层排序；变体跟随父结构
      var parents = g.items.filter(function (s) { return !s.variant_of || !byId[s.variant_of]; });
      var variantsByParent = {};
      g.items.forEach(function (s) {
        if (s.variant_of && byId[s.variant_of]) {
          (variantsByParent[s.variant_of] = variantsByParent[s.variant_of] || []).push(s);
        }
      });
      // 组内排序：先 T-code（规范模版），bonus 排后
      parents.sort(function (a, b) {
        var ta = a.template_code || "Z" + (a.display_name || a.id);
        var tb = b.template_code || "Z" + (b.display_name || b.id);
        return ta < tb ? -1 : (ta > tb ? 1 : 0);
      });
      var og = document.createElement("optgroup");
      og.label = g.label;
      parents.forEach(function (s) {
        og.appendChild(optionFor(s, false));
        (variantsByParent[s.id] || []).forEach(function (v) {
          og.appendChild(optionFor(v, true));
        });
      });
      sel.appendChild(og);
    });
  }
  function pick(sel, preferred) {
    var opts = [].slice.call(sel.options);
    var av = opts.filter(function (o) { return !o.disabled; });
    var hit = av.filter(function (o) { return o.value === preferred; })[0];
    if (hit || av[0] || opts[0]) sel.value = (hit || av[0] || opts[0]).value;
  }

  function init() {
    setStatus("载入注册表…");
    Promise.all([
      fetchJSON("registry/structures.json"),
      fetchJSON("registry/motions.json"),
      fetchJSON("registry/visuals.json"),
      fetchJSON("registry/sfx.json").catch(function () { return { sfx: [], track_index_base: 102 }; }),
      fetchJSON("registry/templates.json").catch(function () { return { templates: [] }; })
    ]).then(function (reg) {
      var structures = reg[0].structures || [];
      var categories = reg[0].categories || [];
      var motions = reg[1].motions || [];
      var visuals = reg[2].visuals || [];
      state.sfxReg = reg[3];
      structures.forEach(function (s) { state.structMeta[s.id] = s; });
      categories.forEach(function (c) { state.catName[c.id] = c.name; });

      (reg[4].templates || []).forEach(function (t) {
        if (t.structure && t.motion) state.compatPairs[t.structure + "::" + t.motion] = true;
      });

      var probes = [];
      structures.forEach(function (s) {
        probes.push(exists(GEN + "structures/" + s.id + "/structure.html")
          .then(function (ok) { state.avail.structures[s.id] = ok; }));
      });
      motions.forEach(function (m) {
        probes.push(exists(GEN + "motions/" + m.id + "/motion.js")
          .then(function (ok) { state.avail.motions[m.id] = ok; }));
      });
      visuals.forEach(function (v) {
        probes.push(exists(GEN + "visuals/" + v.id + "/tokens.css")
          .then(function (ok) { state.avail.visuals[v.id] = ok; }));
      });

      return Promise.all(probes).then(function () {
        buildStructureOptions(selS, structures, categories);
        buildOptions(selM, motions, "motions",
          function (m) { return m.id + " · " + (m.family || "generic") + " · " + (m.energy || ""); });
        buildOptions(selV, visuals, "visuals",
          function (v) { return v.id; });

        pick(selS, structures.length ? structures[0].id : "");
        pick(selM, motions.length ? motions[0].id : "");
        pick(selV, visuals.length ? visuals[0].id : "");

        selS.addEventListener("change", function () { onStructureChange(); });
        selM.addEventListener("change", function () { onMotionChange(); });
        selV.addEventListener("change", function () { scheduleApply(); });
        selBgMode.addEventListener("change", function () { updateBgMode(); scheduleApply(); });
        [inpD, inpBgClipLen, inpDecorStart, inpDecorDuration].forEach(function (el) {
          el.addEventListener("input", scheduleApply);
          el.addEventListener("change", scheduleApply);
        });
        inpBgVideo.addEventListener("change", scheduleApply);

        updateBgMode();
        updateCompatWarn();
        return onStructureChange();
      });
    }).then(function () {
      setStatus("就绪");
    }).catch(function (e) {
      setStatus("初始化失败：" + e.message, "err");
    });
  }

  // ── transport ──
  $("btnResetSlots").addEventListener("click", function () {
    resetSlotDefaults();
  });
  $("btnPlay").addEventListener("click", function () {
    state.paused = !state.paused;
    this.textContent = state.paused ? "▶ 播放" : "⏸ 暂停";
    sendFrame({ t: state.paused ? "pause" : "play" });
  });
  $("btnReplay").addEventListener("click", function () {
    sendFrame({ t: "replay" });
    scheduleParentSfx();
  });
  $("btnMute").addEventListener("click", function () {
    state.muted = !state.muted;
    this.textContent = state.muted ? "🔈 取消静音" : "🔇 静音";
    sendFrame({ t: "mute", v: state.muted });
    if (state.muted) clearParentSfxTimers();
  });
  window.addEventListener("resize", function () { if (frameBox.style.display !== "none") fitScale(); });

  init();
})();
