window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["slide_in_hold"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function (role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };

  if ($("eyebrow").length) {
    tl.from($("eyebrow"), { y: 24, opacity: 0, duration: 0.5, ease: "power3.out" }, 0.12);
  }
  if ($("title").length) {
    tl.from($("title"), { y: 30, opacity: 0, duration: 0.55, ease: "power3.out" }, 0.15);
  }
  if ($("subtitle").length) {
    tl.from($("subtitle"), { y: 20, opacity: 0, duration: 0.5, ease: "expo.out" }, 0.50);
  }
  if ($("stat").length) {
    tl.from($("stat"), { x: -40, opacity: 0, duration: 0.55, ease: "expo.out" }, 0.40);
  }
  if ($("unit").length) {
    tl.from($("unit"), { x: 20, opacity: 0, duration: 0.4, ease: "power2.out" }, 0.60);
  }
  if ($("card").length) {
    var cards = $("card");
    for (var i = 0; i < cards.length; i++) {
      var xOff = (i % 2 === 0) ? -60 : 60;
      tl.from(cards[i], { x: xOff, opacity: 0, duration: 0.6, ease: "expo.out" }, 0.35 + i * 0.12);
    }
  }
  if ($("list-item").length) {
    tl.from($("list-item"), { y: 14, opacity: 0, duration: 0.4, ease: "power2.out", stagger: 0.12 }, 0.70);
  }
  if ($("media").length) {
    tl.from($("media"), { x: 40, opacity: 0, duration: 0.6, ease: "expo.out" }, 0.30);
  }
  if ($("caption").length) {
    tl.from($("caption"), { y: 16, opacity: 0, duration: 0.45, ease: "power2.out" }, 0.75);
  }
  if ($("badge").length) {
    tl.from($("badge"), { y: -16, opacity: 0, duration: 0.4, ease: "power2.out" }, 0.80);
  }
  if ($("cta").length) {
    tl.from($("cta"), { y: 20, opacity: 0, duration: 0.45, ease: "back.out(1.2)" }, 0.85);
  }
  if ($("accent-line").length) {
    tl.from($("accent-line"), { scaleX: 0, opacity: 0, duration: 0.5, ease: "power2.out", transformOrigin: "left center" }, 0.55);
  }
};
