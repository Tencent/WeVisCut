window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["scale_pop_hold"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function (role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };

  if ($("eyebrow").length) {
    tl.from($("eyebrow"), { y: 30, opacity: 0, duration: 0.5, ease: "power3.out" }, 0.15);
  }
  if ($("title").length) {
    tl.from($("title"), { scale: 0.82, opacity: 0, duration: 0.7, ease: "back.out(1.4)" }, 0.30);
  }
  if ($("subtitle").length) {
    tl.from($("subtitle"), { y: 24, opacity: 0, duration: 0.55, ease: "expo.out" }, 0.62);
  }
  if ($("stat").length) {
    tl.from($("stat"), { scale: 0.75, opacity: 0, duration: 0.7, ease: "back.out(1.4)" }, 0.30);
  }
  if ($("unit").length) {
    tl.from($("unit"), { scale: 0.7, opacity: 0, duration: 0.45, ease: "back.out(1.2)" }, 0.65);
  }
  if ($("card").length) {
    tl.from($("card"), { scale: 0.92, opacity: 0, duration: 0.55, ease: "back.out(1.3)", stagger: 0.15 }, 0.40);
  }
  if ($("list-item").length) {
    tl.from($("list-item"), { y: 20, opacity: 0, duration: 0.4, ease: "power2.out", stagger: 0.10 }, 0.70);
  }
  if ($("media").length) {
    tl.from($("media"), { scale: 0.94, opacity: 0, duration: 0.6, ease: "power2.out" }, 0.35);
  }
  if ($("caption").length) {
    tl.from($("caption"), { y: 16, opacity: 0, duration: 0.45, ease: "power2.out" }, 0.80);
  }
  if ($("accent-line").length) {
    tl.from($("accent-line"), { scaleX: 0, opacity: 0, duration: 0.5, ease: "power2.out", transformOrigin: "left center" }, 0.85);
  }
  if ($("badge").length) {
    tl.from($("badge"), { y: 16, opacity: 0, duration: 0.4, ease: "power2.out" }, 0.90);
  }
  if ($("cta").length) {
    tl.from($("cta"), { scale: 0.9, opacity: 0, duration: 0.45, ease: "back.out(1.2)" }, 1.0);
  }
};
