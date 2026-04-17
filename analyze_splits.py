"""
analyze_splits.py — v2 (CHANGES V2)
Reads split-aware manifests (already split; no simulation needed).
Reports split source, actual counts, and actual ratio per dataset.
NEW: Shows OK PREDETERMINED vs CHANGES SPLIT 8:1:1 distinction clearly.
"""
import json
import os
from pathlib import Path

MANIFEST_DIR = "processed/manifests"

LANG_GROUPS = {
    'id':  ['id_cv', 'id_fleurs', 'id_librivox', 'id_titml',
            'id_indocsc', 'id_sindodsc'],
    'ar':  ['ar_cv', 'ar_fleurs', 'ar_clartts'],
    'en':  ['en_librispeech', 'en_fleurs', 'en_cv_spon'],
    'cs':  ['cs_escwa', 'cs_hari', 'cs_homostoria'],
}

SPLIT_LABELS = {
    'predetermined_full':           'OK  PREDETERMINED (full train/dev/test)',
    'predetermined_partial_carved': 'WARNING  PREDETERMINED PARTIAL (dev carved)',
    '8_1_1':                        'CHANGES  APPLIED 8:1:1 (no original splits)',
}


def load_manifest(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def hbar(val: float, width: int = 24) -> str:
    filled = int(width * val)
    return "█" * filled + "░" * (width - filled)


def fmt_ratio_bars(ratio: dict, counts: dict) -> list:
    lines = []
    for s in ['train', 'dev', 'test']:
        r = ratio.get(s, 0.0)
        c = counts.get(s, 0)
        lines.append(
            f"    {s:<6} {hbar(r)} {r*100:5.1f}%  ({c:,} utts)"
        )
    return lines


def print_section(title: str, lines: list, out: list) -> None:
    hdr = f"\n{'─'*68}\n  {title}\n{'─'*68}"
    print(hdr)
    out.append(hdr)
    for ln in lines:
        print(ln)
        out.append(ln)


def main():
    if not os.path.isdir(MANIFEST_DIR):
        print(f"ERROR: {MANIFEST_DIR} not found. Run run_preprocessing.py first.")
        return

    out_lines = [
        "TrilangSR — Data Split Report (CHANGES V2)",
        "=" * 68,
        "Split source key:",
        "  OK PREDETERMINED (full)   — original train/dev/test preserved",
        "  WARNING PREDETERMINED PARTIAL  — had train+test only; dev carved",
        "  CHANGES APPLIED 8:2:1          — no original splits; speaker-indep. split",
        "=" * 68,
    ]

    lang_totals = {}  # {lang: {split: {h, n}}}

    for lang, keys in LANG_GROUPS.items():
        lang_lines = []
        lang_split_hours  = {'train': 0.0, 'dev': 0.0, 'test': 0.0}
        lang_split_counts = {'train': 0,   'dev': 0,   'test': 0}

        for key in keys:
            mf_path = os.path.join(MANIFEST_DIR, f"{key}_manifest.json")
            if not os.path.exists(mf_path):
                lang_lines.append(f"  {key:<22}  [NOT FOUND]")
                continue

            mf = load_manifest(mf_path)
            src    = mf.get('split_source', 'unknown')
            note   = mf.get('split_note', '')
            counts = mf.get('counts', {})
            ratio  = mf.get('actual_ratio', {})
            total  = mf.get('total_utts', 0)

            label = SPLIT_LABELS.get(src, f'  {src}')

            # Compute hours from records
            split_hours = {}
            for s in ['train', 'dev', 'test']:
                recs = mf.get(s, [])
                h    = sum(r.get('duration', 0.0) for r in recs) / 3600
                split_hours[s] = h
                lang_split_hours[s]  += h
                lang_split_counts[s] += counts.get(s, 0)

            lang_lines.append(f"\n  ── {key}")
            lang_lines.append(f"     {label}")
            lang_lines.append(f"     {note}")
            lang_lines.append(f"     Total: {total:,} utts  |  "
                               f"{sum(split_hours.values()):.2f} h")
            lang_lines.extend(fmt_ratio_bars(ratio, counts))
            lang_lines.append(
                "     Hours: " +
                " | ".join(f"{s}={split_hours[s]:.2f}h" for s in ['train','dev','test'])
            )

        # Language summary
        total_h = sum(lang_split_hours.values())
        total_n = sum(lang_split_counts.values())
        lang_lines.append(f"\n  ── {lang.upper()} TOTAL")
        lang_lines.append(f"     {total_n:,} utts  |  {total_h:.2f} h")
        for s in ['train', 'dev', 'test']:
            h = lang_split_hours[s]
            n = lang_split_counts[s]
            r = n / max(total_n, 1)
            lang_lines.append(f"     {s:<6} {hbar(r)} {r*100:5.1f}%  "
                               f"({n:,} utts, {h:.2f} h)")

        lang_totals[lang] = {
            'hours': dict(lang_split_hours),
            'counts': dict(lang_split_counts),
        }
        print_section(f"Language: {lang.upper()}", lang_lines, out_lines)

    # ── Trilingual combined summary ──────────────────────────────────────────
    tri_lines = []
    tri_lines.append("  Combined: ID + AR + EN (no CS)")
    tri_lines.append(f"  {'Split':<8} {'Utts':>10}  {'Hours':>8}  {'Ratio':>7}")
    tri_lines.append(f"  {'─'*8} {'─'*10}  {'─'*8}  {'─'*7}")
    tri_total_n = 0
    tri_total_h = 0.0
    for s in ['train', 'dev', 'test']:
        n = sum(lang_totals[l]['counts'][s] for l in ['id', 'ar', 'en'])
        h = sum(lang_totals[l]['hours'][s]  for l in ['id', 'ar', 'en'])
        tri_total_n += n
        tri_total_h += h
        tri_lines.append(f"  {s:<8} {n:>10,}  {h:>8.2f}  "
                         f"{n/max(sum(lang_totals[l]['counts'][s2] for l in ['id','ar','en'] for s2 in ['train','dev','test']),1)*100:6.1f}%")
    tri_lines.append(f"\n  TOTAL    {tri_total_n:>10,}  {tri_total_h:>8.2f}")
    print_section("TRILINGUAL COMBINED (ID + AR + EN)", tri_lines, out_lines)

    # ── Thesis summary table ─────────────────────────────────────────────────
    tbl = []
    tbl.append(f"  {'Lang':<6} {'Split Source Distribution':>30}  "
               f"{'Total h':>8}  {'Train h':>8}  {'Dev h':>7}  {'Test h':>7}")
    tbl.append("  " + "─" * 68)

    for lang in ['id', 'ar', 'en', 'cs']:
        keys_for_lang = LANG_GROUPS.get(lang, [])
        for key in keys_for_lang:
            mf_path = os.path.join(MANIFEST_DIR, f"{key}_manifest.json")
            if not os.path.exists(mf_path):
                continue
            mf  = load_manifest(mf_path)
            src = mf.get('split_source', 'unknown')
            src_short = {
                'predetermined_full':           'PRED-FULL',
                'predetermined_partial_carved': 'PRED-PART',
                '8_1_1':                        '8:1:1    ',
            }.get(src, src[:9])
            h_total = sum(
                sum(r.get('duration', 0.0) for r in mf.get(s, [])) / 3600
                for s in ['train', 'dev', 'test']
            )
            h = {s: sum(r.get('duration', 0.0) for r in mf.get(s, [])) / 3600
                 for s in ['train', 'dev', 'test']}
            tbl.append(f"  {key:<22} [{src_short}]  "
                       f"{h_total:>8.2f}  {h['train']:>8.2f}  "
                       f"{h['dev']:>7.2f}  {h['test']:>7.2f}")
        tbl.append("")

    print_section("THESIS SUMMARY TABLE", tbl, out_lines)

    # Save report
    os.makedirs("processed", exist_ok=True)
    report_path = "processed/splits_report_v2.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()