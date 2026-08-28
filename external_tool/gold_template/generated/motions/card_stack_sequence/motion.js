window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["card_stack_sequence"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function (role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };

  if ($("eyebrow").length) {
    tl.from($("eyebrow"), { y: 20, opacity: 0, duration: 0.4, ease: "power2.out" }, 0.10);
  }
  if ($("title").length) {
    tl.from($("title"), { y: 30, opacity: 0, duration: 0.55, ease: "power3.out" }, 0.15);
  }
  if ($("subtitle").length) {
    tl.from($("subtitle"), { y: 18, opacity: 0, duration: 0.45, ease: "expo.out" }, 0.50);
  }
  if ($("stat").length) {
    tl.from($("stat"), { y: 40, opacity: 0, scale: 0.96, duration: 0.5, ease: "back.out(1.3)" }, 0.50);
  }
  if ($("unit").length) {
    tl.from($("unit"), { opacity: 0, duration: 0.35, ease: "power2.out" }, 0.75);
  }
  if ($("accent-line").length) {
    tl.from($("accent-line"), { scaleX: 0, opacity: 0, duration: 0.7, ease: "expo.out", transformOrigin: "left center" }, 0.30);
  }
  if ($("card").length) {
    tl.from($("card"), { y: 40, opacity: 0, scale: 0.96, duration: 0.5, ease: "back.out(1.3)", stagger: 0.22 }, 0.70);
  }
  if ($("list-item").length) {
    tl.from($("list-item"), { y: 16, opacity: 0, duration: 0.4, ease: "power2.out", stagger: 0.18 }, 1.10);
  }
  if ($("media").length) {
    tl.from($("media"), { y: 30, opacity: 0, scale: 0.96, duration: 0.5, ease: "back.out(1.2)" }, 0.55);
  }
  if ($("caption").length) {
    tl.from($("caption"), { y: 16, opacity: 0, duration: 0.4, ease: "power2.out" }, 1.20);
  }
  if ($("badge").length) {
    tl.from($("badge"), { y: -12, opacity: 0, duration: 0.35, ease: "power2.out" }, 1.30);
  }
  if ($("cta").length) {
    tl.from($("cta"), { y: 20, opacity: 0, scale: 0.96, duration: 0.45, ease: "back.out(1.2)" }, 1.40);
  }
};
