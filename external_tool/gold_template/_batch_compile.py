#!/usr/bin/env python3
"""Batch compile all templates that are missing index.html."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from assembler.assemble import assemble

ROOT = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(ROOT, "generated")

MOTION_DEFAULT = "scale_pop_hold"
VISUAL_DEFAULT = "dark_tech"

TEMPLATE_CONFIG = {
    "T28_question_hook":         {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 6.0},
    "T29_fun_fact":              {"motion": "scale_pop_hold",     "visual": "neon_cyber",   "dur": 5.5},
    "T30_analogy_card":          {"motion": "slide_in_hold",      "visual": "warm_editorial","dur": 6.0},
    "T31_emoji_concept":         {"motion": "scale_pop_hold",     "visual": "clean_saas",   "dur": 5.0},
    "T32_cause_effect_chain":    {"motion": "card_stack_sequence", "visual": "dark_tech",   "dur": 7.5},
    "T33_myth_busting":          {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 6.5},
    "T34_scale_comparison":      {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 6.0},
    "T35_ranking_list":          {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 7.5},
    "T36_pros_cons":             {"motion": "slide_in_hold",      "visual": "clean_saas",   "dur": 6.5},
    "T37_historical_moment":     {"motion": "slide_in_hold",      "visual": "warm_editorial","dur": 6.0},
    "T38_question_pause":        {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 5.5},
    "T39_recap_checklist":       {"motion": "card_stack_sequence", "visual": "clean_saas",  "dur": 7.0},
    "T40_open_question":         {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 5.5},
    "T41_venn_compare":          {"motion": "scale_pop_hold",     "visual": "dark_tech",    "dur": 6.0},
}


def get_config(template_id):
    for prefix, cfg in TEMPLATE_CONFIG.items():
        if template_id.startswith(prefix):
            return cfg
    return {"motion": MOTION_DEFAULT, "visual": VISUAL_DEFAULT, "dur": 6.0}


def main():
    tpl_dir = os.path.join(GEN, "templates")
    entries = sorted(os.listdir(tpl_dir))
    compiled = 0
    failed = []
    
    for entry in entries:
        idx_path = os.path.join(tpl_dir, entry, "index.html")
        if os.path.isfile(idx_path):
            continue
        
        parts = entry.split("_", 1)
        if len(parts) < 2:
            continue
        struct_id = parts[1]
        
        struct_html = os.path.join(GEN, "structures", struct_id, "structure.html")
        if not os.path.isfile(struct_html):
            failed.append(f"{entry}: missing structure {struct_id}")
            continue
        
        ei_path = os.path.join(tpl_dir, entry, "example_inputs.json")
        slots = {}
        if os.path.isfile(ei_path):
            with open(ei_path) as f:
                slots = json.load(f)
        
        cfg = get_config(entry)
        
        try:
            html = assemble(
                structure_id=struct_id,
                motion_id=cfg["motion"],
                visual_id=cfg["visual"],
                slots=slots,
                duration=cfg["dur"],
            )
            os.makedirs(os.path.dirname(idx_path), exist_ok=True)
            with open(idx_path, "w", encoding="utf-8") as f:
                f.write(html)
            compiled += 1
            print(f"  OK  {entry} ({struct_id} x {cfg['motion']} x {cfg['visual']})")
        except Exception as e:
            failed.append(f"{entry}: {e}")
            print(f"  FAIL {entry}: {e}")
    
    print(f"\nCompiled: {compiled}, Failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
