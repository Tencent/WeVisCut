window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["parallax_pan"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function (role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };

  if ($("bg").length) {
    tl.from($("bg"), { x: -20, duration: 6.8, ease: "sine.inOut" }, 0);
  }
  if ($("media").length) {
    tl.from($("media"), { scale: 1.06, duration: 0.8, ease: "power2.out" }, 0.15);
  }
  if ($("eyebrow").length) {
    tl.from($("eyebrow"), { y: 18, opacity: 0, duration: 0.45, ease: "power2.out" }, 0.10);
  }
  if ($("title").length) {
    tl.from($("title"), { y: 30, opacity: 0, duration: 0.6, ease: "power3.out" }, 0.15);
  }
  if ($("subtitle").length) {
    tl.from($("subtitle"), { y: 20, opacity: 0, duration: 0.5, ease: "expo.out" }, 0.50);
  }
  if ($("stat").length) {
    tl.from($("stat"), { y: 24, opacity: 0, duration: 0.55, ease: "power3.out" }, 0.30);
  }
  if ($("unit").length) {
    tl.from($("unit"), { opacity: 0, duration: 0.35, ease: "power2.out" }, 0.55);
  }
  if ($("card").length) {
    tl.from($("card"), { y: 24, opacity: 0, duration: 0.5, ease: "power2.out", stagger: 0.14 }, 0.40);
  }
  if ($("list-item").length) {
    tl.from($("list-item"), { x: -24, opacity: 0, duration: 0.5, ease: "expo.out", stagger: 0.14 }, 0.60);
  }
  if ($("caption").length) {
    tl.from($("caption"), { y: 16, opacity: 0, duration: 0.45, ease: "power2.out" }, 0.70);
  }
  if ($("badge").length) {
    tl.from($("badge"), { y: -12, opacity: 0, duration: 0.35, ease: "power2.out" }, 0.75);
  }
  if ($("cta").length) {
    tl.from($("cta"), { y: 16, opacity: 0, duration: 0.4, ease: "back.out(1.2)" }, 0.80);
  }
  if ($("accent-line").length) {
    tl.from($("accent-line"), { scaleX: 0, duration: 0.5, ease: "expo.out", transformOrigin: "left center" }, 0.40);
  }
};
