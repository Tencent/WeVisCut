window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["mask_reveal"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function (role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };

  if ($("eyebrow").length) {
    tl.from($("eyebrow"), { y: 20, opacity: 0, duration: 0.45, ease: "power2.out" }, 0.10);
  }
  if ($("title").length) {
    tl.from($("title"), { clipPath: "inset(0 100% 0 0)", duration: 0.7, ease: "power4.out" }, 0.20);
  }
  if ($("subtitle").length) {
    tl.from($("subtitle"), { clipPath: "inset(0 100% 0 0)", duration: 0.55, ease: "power3.out" }, 0.65);
  }
  if ($("stat").length) {
    tl.from($("stat"), { clipPath: "inset(0 100% 0 0)", duration: 0.7, ease: "power4.out" }, 0.20);
  }
  if ($("unit").length) {
    tl.from($("unit"), { opacity: 0, x: -10, duration: 0.4, ease: "power2.out" }, 0.70);
  }
  if ($("card").length) {
    tl.from($("card"), { clipPath: "inset(0 100% 0 0)", duration: 0.6, ease: "power3.out", stagger: 0.18 }, 0.35);
  }
  if ($("list-item").length) {
    tl.from($("list-item"), { clipPath: "inset(0 100% 0 0)", duration: 0.5, ease: "power3.out", stagger: 0.12 }, 0.55);
  }
  if ($("media").length) {
    tl.from($("media"), { clipPath: "inset(0 100% 0 0)", duration: 0.7, ease: "expo.out" }, 0.30);
  }
  if ($("caption").length) {
    tl.from($("caption"), { clipPath: "inset(0 100% 0 0)", duration: 0.5, ease: "power3.out" }, 0.85);
  }
  if ($("accent-line").length) {
    tl.from($("accent-line"), { scaleX: 0, opacity: 0, duration: 0.6, ease: "expo.out", transformOrigin: "left center" }, 0.55);
  }
  if ($("badge").length) {
    tl.from($("badge"), { y: 16, opacity: 0, duration: 0.4, ease: "power2.out" }, 1.04);
  }
  if ($("cta").length) {
    tl.from($("cta"), { clipPath: "inset(0 100% 0 0)", duration: 0.5, ease: "power3.out" }, 1.10);
  }
};
