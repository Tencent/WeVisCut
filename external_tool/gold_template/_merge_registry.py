#!/usr/bin/env python3
"""把 generated/structures/* 合并进 registry/structures.json：
- 基结构（registry 已有同 id）：去掉 generated:false，补 bound（若 card 标注）。
- 变体（id = 基id_后缀）：新增条目，带 variant_of + variant_label（继承父 category，不占 template_code）。
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REG = ROOT / "registry" / "structures.json"
GEN = ROOT / "generated" / "structures"

SUFFIX_LABEL = {
    "scatter": "散布布局", "split": "左右分栏", "fullbleed": "全屏铺满",
    "grid": "规整网格", "timeline": "时间线", "center": "居中聚焦",
    "side": "侧边竖排", "float": "浮动卡片", "diagonal": "对角错落",
    "stack": "堆叠层级", "exploded": "分解视图", "iso": "等距视图",
    "corner": "角落总览", "zigzag": "错列推进", "radial": "中心放射",
    "cards": "卡片式", "masonry": "错落瀑布", "multi": "多画面",
    "tunnel": "推进隧道", "wall": "屏幕墙",
}

reg = json.loads(REG.read_text(encoding="utf-8"))
structs = reg["structures"]
by_id = {e["id"]: e for e in structs}
base_ids = sorted(by_id.keys(), key=len, reverse=True)  # 最长前缀优先

def find_base(vid):
    for b in base_ids:
        if vid == b:
            return None  # 自身就是基
        if vid.startswith(b + "_"):
            return b
    return None

dirs = sorted(d.name for d in GEN.iterdir() if d.is_dir())
updated, added = [], []
new_entries = []
for sid in dirs:
    cardp = GEN / sid / "structure_card.json"
    card = json.loads(cardp.read_text(encoding="utf-8")) if cardp.exists() else {}
    if sid in by_id:
        e = by_id[sid]
        if e.pop("generated", None) is False:
            updated.append(sid + " (-generated:false)")
        if card.get("bound") and not e.get("bound"):
            e["bound"] = True
        continue
    base = find_base(sid)
    if not base:
        # 未知基，跳过（理论不发生）
        continue
    suffix = sid[len(base) + 1:]
    label = SUFFIX_LABEL.get(suffix, suffix)
    be = by_id[base]
    entry = {
        "id": sid,
        "owner_prompt": be.get("owner_prompt"),
        "category": be.get("category"),
        "display_name": be.get("display_name", base) + " · " + label,
        "family": card.get("family", be.get("family")),
        "variant_of": base,
        "variant_label": label,
        "use_for": card.get("use_for", be.get("use_for", [])),
        "spotlight_level": card.get("spotlight_level", be.get("spotlight_level", "mid")),
        "duration_range": card.get("duration_range", be.get("duration_range", [5.0, 8.0])),
        "anim_roles": card.get("anim_roles", be.get("anim_roles", [])),
        "slots": card.get("slots", be.get("slots", [])),
    }
    if be.get("bonus"):
        entry["bonus"] = True
    if card.get("bound"):
        entry["bound"] = True
    if card.get("data_slots"):
        entry["data_slots"] = card["data_slots"]
    new_entries.append((base, entry))
    added.append(sid + " (variant_of=" + base + ", " + label + ")")

# 把变体插入到其父结构之后，便于 viewer 缩进归组
for base, entry in new_entries:
    idx = next((i for i, e in enumerate(structs) if e["id"] == base), len(structs) - 1)
    # 找到该 base 已有变体的最后位置
    j = idx
    while j + 1 < len(structs) and structs[j + 1].get("variant_of") == base:
        j += 1
    structs.insert(j + 1, entry)
    by_id[entry["id"]] = entry

REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("base updated:", len(updated))
for u in updated: print("  ", u)
print("variants added:", len(added))
for a in added: print("  ", a)
print("total structures in registry:", len(structs))
