"""Model-fit tables in booktabs style, data vs model."""
import json

from paths import CLEAN, TABLES
from smm_joint import load_targets

targets = load_targets()
M = json.loads((CLEAN / 'wage_gaps.json').read_text())['model_fit']
D = dict(lfp=targets['lfp'], nu=targets['nu'], en=targets['en'], eu=targets['eu'],
         ue=targets['ue'], unemp=targets['unemp'], gpg=float(targets['gpg']))

# Row order: NC Men, NC Women, C Men, C Women, each (no child, young child).
ROWS = [  # (type_idx, skill, gender, children)
    (4, 'Non-college', 'Men',   'no child'),    (0, '', '', 'young child'),
    (5, 'Non-college', 'Women', 'no child'),     (1, '', '', 'young child'),
    (6, 'College',     'Men',   'no child'),     (2, '', '', 'young child'),
    (7, 'College',     'Women', 'no child'),      (3, '', '', 'young child'),
]
GROUPS = [('Non-college', 'Men', 0, 4), ('Non-college', 'Women', 1, 5),
          ('College', 'Men', 2, 6), ('College', 'Women', 3, 7)]


def pct(v):
    return f"{100 * v:.1f}"


def targeted():
    L = [r"\begin{tabular}{lll cc cc cc cc}", r"\toprule",
         r"& & & \multicolumn{2}{c}{LFP} & \multicolumn{2}{c}{$N\!\to\!U$} "
         r"& \multicolumn{2}{c}{$E\!\to\!N$} & \multicolumn{2}{c}{$E\!\to\!U$} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
         r"Skill & Gender & Children & Data & Model & Data & Model & Data & Model & Data & Model \\",
         r"\midrule"]
    for idx, sk, ge, ch in ROWS:
        cells = [pct(D['lfp'][idx]), pct(M['lfp'][idx]), pct(D['nu'][idx]), pct(M['nu'][idx]),
                 pct(D['en'][idx]), pct(M['en'][idx]), pct(D['eu'][idx]), pct(M['eu'][idx])]
        L.append(f"{sk} & {ge} & {ch} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{11}{l}{\textit{Aggregate targeted moments:}}\\")
    ueg_d = [0.5 * (D['ue'][a] + D['ue'][b]) for _, _, a, b in GROUPS]
    ueg_m = [0.5 * (M['ue'][a] + M['ue'][b]) for _, _, a, b in GROUPS]
    ue_str = ", ".join(f"{'Un' if 'Non' in g[0] else 'Sk'}{'M' if g[1] == 'Men' else 'F'}: "
                       f"{pct(d)}/{pct(m)}" for g, d, m in zip(GROUPS, ueg_d, ueg_m))
    L.append(r"\multicolumn{3}{l}{Job-finding $U\!\to\!E$ (data/model, \%)} & "
             r"\multicolumn{8}{l}{" + ue_str + r"} \\")
    L.append(r"\multicolumn{3}{l}{Gender pay gap (male/female)} & "
             r"\multicolumn{8}{l}{Data " + f"{D['gpg']:.3f}" + r" \quad Model "
             + f"{M['gpg']:.3f}" + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def untargeted():
    L = [r"\begin{tabular}{lll cc cc}", r"\toprule",
         r"& & & \multicolumn{2}{c}{Unemp.\ rate} & \multicolumn{2}{c}{$U\!\to\!E$ (type)} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Skill & Gender & Children & Data & Model & Data & Model \\",
         r"\midrule"]
    for idx, sk, ge, ch in ROWS:
        cells = [pct(D['unemp'][idx]), pct(M['unemp'][idx]), pct(D['ue'][idx]), pct(M['ue'][idx])]
        L.append(f"{sk} & {ge} & {ch} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


(TABLES / 'tab_fit_targeted.tex').write_text(targeted() + "\n")
(TABLES / 'tab_fit_untargeted.tex').write_text(untargeted() + "\n")
print("wrote output/tables/tab_fit_targeted.tex and tab_fit_untargeted.tex")
print("\n=== TARGETED ===\n" + targeted())
print("\n=== UNTARGETED ===\n" + untargeted())
