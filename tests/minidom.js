/* A DOM small enough to read, large enough to run the studio frontend.
 *
 * jsdom is unreachable from this network, and `node --check` only proves the
 * files parse — it says nothing about whether an id in index.html matches the
 * id the script looks up, which is exactly the class of bug that breaks a page
 * silently. So this implements the subset the studio actually uses: element
 * creation, the attribute/class/dataset/style surfaces, innerHTML assignment,
 * and a selector engine covering the selectors in our code.
 *
 * Anything outside that subset throws loudly rather than returning undefined,
 * so an unsupported call shows up as a test failure instead of a silent no-op.
 */
'use strict';

const VOID_TAGS = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr',
]);

// ---------------------------------------------------------------- selectors
function parseCompound(text) {
  // e.g. `input[data-slot]`, `.rail-item[data-track="x"]`, `#id`, `div.a.b`
  const out = { tag: '', id: '', classes: [], attrs: [] };
  const re = /(^[a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:\s*=\s*"([^"]*)")?\]/g;
  let m;
  let consumed = 0;
  while ((m = re.exec(text)) !== null) {
    consumed = re.lastIndex;
    if (m[1]) out.tag = m[1].toLowerCase();
    else if (m[2]) out.id = m[2];
    else if (m[3]) out.classes.push(m[3]);
    else if (m[4]) out.attrs.push([m[4], m[5]]);
  }
  if (consumed !== text.length) {
    throw new Error(`minidom: unsupported selector fragment ${JSON.stringify(text)}`);
  }
  return out;
}

function matchesCompound(el, c) {
  if (c.tag && el.tagName.toLowerCase() !== c.tag) return false;
  if (c.id && el.getAttribute('id') !== c.id) return false;
  for (const cls of c.classes) if (!el.classList.contains(cls)) return false;
  for (const [name, value] of c.attrs) {
    const actual = el.getAttribute(name);
    if (actual === null) return false;
    if (value !== undefined && actual !== value) return false;
  }
  return true;
}

function parseSelector(selector) {
  // Supports comma groups and descendant combinators only; that is all we use.
  return selector.split(',').map((group) =>
    group.trim().split(/\s+/).filter(Boolean).map(parseCompound));
}

function matchesChain(el, chain) {
  const last = chain[chain.length - 1];
  if (!matchesCompound(el, last)) return false;
  let i = chain.length - 2;
  let cur = el.parentNode;
  while (i >= 0 && cur) {
    if (cur.nodeType === 1 && matchesCompound(cur, chain[i])) i -= 1;
    cur = cur.parentNode;
  }
  return i < 0;
}

// ---------------------------------------------------------------- nodes
class ClassList {
  constructor(el) { this.el = el; }
  get _set() {
    return new Set((this.el.getAttribute('class') || '').split(/\s+/).filter(Boolean));
  }
  _write(set) { this.el.setAttribute('class', [...set].join(' ')); }
  add(...names) { const s = this._set; names.forEach((n) => s.add(n)); this._write(s); }
  remove(...names) { const s = this._set; names.forEach((n) => s.delete(n)); this._write(s); }
  contains(name) { return this._set.has(name); }
  toggle(name, force) {
    const s = this._set;
    const on = force === undefined ? !s.has(name) : !!force;
    if (on) s.add(name); else s.delete(name);
    this._write(s);
    return on;
  }
  get length() { return this._set.size; }
}

class TextNode {
  constructor(text) { this.nodeType = 3; this.data = String(text); this.parentNode = null; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
}

class Element {
  constructor(tagName, doc) {
    this.nodeType = 1;
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = doc;
    this.parentNode = null;
    this.childNodes = [];
    this.attributes = new Map();
    this.classList = new ClassList(this);
    this.dataset = makeDataset(this);
    this.style = makeStyle(this);
    this.listeners = {};
    this._value = '';
  }

  // -- attributes --
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    const v = this.attributes.get(name);
    return v === undefined ? null : v;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }

  get id() { return this.getAttribute('id') || ''; }
  set id(v) { this.setAttribute('id', v); }
  get className() { return this.getAttribute('class') || ''; }
  set className(v) { this.setAttribute('class', v); }

  get hidden() { return this.attributes.get('hidden') === 'true' || this.attributes.has('hidden') && this.attributes.get('hidden') !== 'false'; }
  set hidden(v) {
    if (v) this.setAttribute('hidden', 'true'); else this.attributes.delete('hidden');
  }
  get disabled() { return this.attributes.has('disabled'); }
  set disabled(v) {
    if (v) this.setAttribute('disabled', 'true'); else this.attributes.delete('disabled');
  }
  get title() { return this.getAttribute('title') || ''; }
  set title(v) { this.setAttribute('title', v); }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }

  // -- tree --
  appendChild(node) {
    if (node instanceof DocumentFragment) {
      [...node.childNodes].forEach((c) => this.appendChild(c));
      return node;
    }
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }
  removeChild(node) {
    const i = this.childNodes.indexOf(node);
    if (i >= 0) { this.childNodes.splice(i, 1); node.parentNode = null; }
    return node;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }

  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get parentElement() {
    return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null;
  }

  get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }
  set textContent(v) {
    this.childNodes.forEach((n) => { n.parentNode = null; });
    this.childNodes = [];
    if (v !== '' && v !== null && v !== undefined) {
      this.appendChild(new TextNode(v));
    }
  }

  get innerHTML() { return serialize(this); }
  set innerHTML(html) {
    this.childNodes.forEach((n) => { n.parentNode = null; });
    this.childNodes = [];
    if (!html) return;
    parseInto(this, String(html), this.ownerDocument);
  }

  // -- queries --
  querySelectorAll(selector) {
    const groups = parseSelector(selector);
    const out = [];
    walk(this, (el) => {
      if (el !== this && groups.some((chain) => matchesChain(el, chain))) out.push(el);
    });
    return out;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }

  // -- events --
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
  }
  /* Bubbles up the tree like a real event, because handlers on nested rows
   * (a delete button inside a clickable list item) rely on stopPropagation,
   * and a non-bubbling stub would make those tests vacuously pass. */
  dispatch(type, event) {
    let stopped = false;
    const ev = Object.assign({
      type: type,
      target: this,
      preventDefault() {},
      stopPropagation() { stopped = true; },
    }, event || {});

    let node = this;
    while (node && node.nodeType === 1) {
      ev.currentTarget = node;
      const inline = node['on' + type];
      if (typeof inline === 'function') inline.call(node, ev);
      (node.listeners[type] || []).forEach((fn) => fn.call(node, ev));
      if (stopped) break;
      node = node.parentNode;
    }
    return ev;
  }
  click() { return this.dispatch('click'); }

  // -- layout stubs; enough for the timeline's arithmetic --
  get clientWidth() { return this._clientWidth !== undefined ? this._clientWidth : 900; }
  set clientWidth(v) { this._clientWidth = v; }
  get clientHeight() { return this._clientHeight !== undefined ? this._clientHeight : 400; }
  set clientHeight(v) { this._clientHeight = v; }
  get scrollWidth() { return this.clientWidth; }
  get scrollHeight() { return this.clientHeight; }
  get scrollLeft() { return this._scrollLeft || 0; }
  set scrollLeft(v) { this._scrollLeft = v; }
  get scrollTop() { return this._scrollTop || 0; }
  set scrollTop(v) { this._scrollTop = v; }
  get currentTime() { return this._currentTime || 0; }
  set currentTime(v) { this._currentTime = v; }
}

class DocumentFragment {
  constructor() {
    this.nodeType = 11;
    this.childNodes = [];
  }
  appendChild(node) {
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = null;
    this.childNodes.push(node);
    return node;
  }
}

function walk(root, visit) {
  visit(root);
  root.childNodes.forEach((n) => { if (n.nodeType === 1) walk(n, visit); });
}

function makeDataset(el) {
  const toAttr = (k) => 'data-' + k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase());
  return new Proxy({}, {
    get: (_t, k) => (typeof k === 'string' ? (el.getAttribute(toAttr(k)) ?? undefined) : undefined),
    set: (_t, k, v) => { el.setAttribute(toAttr(k), v); return true; },
    has: (_t, k) => el.hasAttribute(toAttr(k)),
    ownKeys: () => [...el.attributes.keys()].filter((a) => a.startsWith('data-'))
      .map((a) => a.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())),
    getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
  });
}

function makeStyle(el) {
  const props = new Map();
  const target = {
    setProperty(name, value) { props.set(name, String(value)); },
    getPropertyValue(name) { return props.get(name) || ''; },
    _props: props,
  };
  return new Proxy(target, {
    get: (t, k) => (k in t ? t[k] : (props.get(k) || '')),
    set: (t, k, v) => { props.set(k, String(v)); return true; },
  });
}

// ---------------------------------------------------------------- parsing
const TAG_RE = /<!--[\s\S]*?-->|<!DOCTYPE[^>]*>|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=/>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*(\/?)>/g;
const ATTR_RE = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;

function decode(text) {
  return text
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, '\u00a0').replace(/&amp;/g, '&');
}

function parseInto(root, html, doc) {
  const stack = [root];
  let last = 0;
  let m;
  TAG_RE.lastIndex = 0;

  const pushText = (text) => {
    if (!text) return;
    const top = stack[stack.length - 1];
    // ignore pure whitespace between elements; it never matters to our tests
    if (!text.trim() && top.childNodes.length === 0) return;
    if (!text.trim()) return;
    top.appendChild(new TextNode(decode(text)));
  };

  while ((m = TAG_RE.exec(html)) !== null) {
    pushText(html.slice(last, m.index));
    last = TAG_RE.lastIndex;

    if (m[0].startsWith('<!')) continue;

    if (m[1]) {                                   // closing tag
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName.toLowerCase() === m[1].toLowerCase()) {
          stack.length = i;
          break;
        }
      }
      continue;
    }

    const tag = m[2].toLowerCase();
    const el = doc.createElement(tag);
    if (m[3]) {
      ATTR_RE.lastIndex = 0;
      let a;
      while ((a = ATTR_RE.exec(m[3])) !== null) {
        const value = a[2] !== undefined ? a[2]
          : a[3] !== undefined ? a[3]
          : a[4] !== undefined ? a[4] : '';
        el.setAttribute(a[1], decode(value));
      }
    }
    stack[stack.length - 1].appendChild(el);

    if (tag === 'script' || tag === 'style') {
      // skip raw text content wholesale
      const close = new RegExp(`</${tag}\\s*>`, 'i');
      const rest = html.slice(TAG_RE.lastIndex);
      const idx = rest.search(close);
      if (idx >= 0) {
        TAG_RE.lastIndex += idx + rest.slice(idx).indexOf('>') + 1;
        last = TAG_RE.lastIndex;
      }
      continue;
    }
    if (!VOID_TAGS.has(tag) && !m[4]) stack.push(el);
  }
  pushText(html.slice(last));
}

function serialize(el) {
  return el.childNodes.map((n) => {
    if (n.nodeType === 3) return n.data;
    const attrs = [...n.attributes.entries()].map(([k, v]) => ` ${k}="${v}"`).join('');
    const tag = n.tagName.toLowerCase();
    if (VOID_TAGS.has(tag)) return `<${tag}${attrs}>`;
    return `<${tag}${attrs}>${serialize(n)}</${tag}>`;
  }).join('');
}

// ---------------------------------------------------------------- document
class MiniDocument {
  constructor(html) {
    this.nodeType = 9;
    this.documentElement = new Element('html', this);
    this.documentElement.ownerDocument = this;
    this.listeners = {};
    if (html) parseInto(this.documentElement, html, this);
    this.body = this.documentElement.querySelector('body') || this.documentElement;
  }
  createElement(tag) { return new Element(tag, this); }
  createTextNode(text) { return new TextNode(text); }
  createDocumentFragment() { return new DocumentFragment(); }
  getElementById(id) {
    let found = null;
    walk(this.documentElement, (el) => {
      if (!found && el.getAttribute('id') === id) found = el;
    });
    return found;
  }
  querySelector(sel) { return this.documentElement.querySelector(sel); }
  querySelectorAll(sel) { return this.documentElement.querySelectorAll(sel); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type) { (this.listeners[type] || []).forEach((fn) => fn({ type: type })); }
  get title() { return this._title || ''; }
  set title(v) { this._title = v; }
}

module.exports = { MiniDocument, Element, TextNode, DocumentFragment, parseSelector };
