"""LaTeX tables generated from the result files (nothing typed by hand)."""
import json
from paths import CALIBRATION_RESULT, CLEAN, TABLES
TYPE_COLUMNS = ['UnM', 'UnF', 'SkM', 'SkF']          # type order is [child x4, no-child x4]


def write_calibration_table():
    result = json.loads(CALIBRATION_RESULT.read_text())
    lam, kappa, mu_vec, delta = (result[key] for key in ('lambda_grid', 'K', 'mu_vec', 'delta'))

    def row(label, description, values, fmt):
        return f"{label} & {description} & " + " & ".join(format(v, fmt) for v in values) + r" \\"

    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             "Parameter & Description & " + " & ".join(TYPE_COLUMNS) + r" \\", r"\midrule",
             row(r"$\lambda$ (child)", "Opp.-cost volatility, child-present", lam[:4], '.4f'),
             row(r"$\lambda$ (no child)", "Opp.-cost volatility, no child", lam[4:], '.4f'),
             row(r"$K$ (child)", "GPD shape, child-present", kappa[:4], '.3f'),
             row(r"$K$ (no child)", "GPD shape, no child", kappa[4:], '.3f'),
             row(r"$\mu$", "Matching efficiency (by skill market)", [mu_vec[0], mu_vec[1], mu_vec[2], mu_vec[3]], '.3f'),
             r"\midrule",
             r"\multicolumn{6}{l}{\textit{Externally fixed --- separation rate $\delta$ ($E\!\to\!U$), by type:}} \\",
             row(r"$\delta$ (child)", "Separation rate, child-present", delta[:4], '.4f'),
             row(r"$\delta$ (no child)", "Separation rate, no child", delta[4:], '.4f'),
             r"\bottomrule", r"\end{tabular}"]
    (TABLES / 'tab_calib_params.tex').write_text("\n".join(lines) + "\n")


def write_counterfactual_table():
    rows = json.loads((CLEAN / 'counterfactuals.json').read_text())['rows']
    base = {r['skill']: r for r in rows if r['scenario'] == 'Baseline'}
    lines = [r"\begin{tabular}{ll rr rr r r}", r"\toprule",
             r"Scenario & Skill & \multicolumn{2}{c}{Mothers' LFP} & \multicolumn{2}{c}{Mothers' employment}"
             r" & Earnings & Child penalty \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & & level & $\Delta$pp & level & $\Delta$pp & $\Delta$\% & (earnings, \%) \\", r"\midrule"]
    scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
    for scenario in scenarios:
        for r in (x for x in rows if x['scenario'] == scenario):
            b = base[r['skill']]
            d_lfp, d_emp = 100 * (r['lfp'] - b['lfp']), 100 * (r['emp'] - b['emp'])
            if scenario == 'Baseline':
                delta_lfp = delta_emp = delta_earn = '---'
            else:
                delta_lfp, delta_emp = f"{d_lfp:+.1f}", f"{d_emp:+.1f}"
                delta_earn = f"{r['earn_pct_chg']:+.1f}"
            lines.append(f"{scenario} & {r['skill']} & {r['lfp']:.3f} & {delta_lfp} & {r['emp']:.3f} & {delta_emp}"
                         f" & {delta_earn} & {r['child_penalty_earn']:.1f} \\\\")
        if scenario != scenarios[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / 'tab_counterfactuals.tex').write_text("\n".join(lines) + "\n")


def write_ge_table():
    data = json.loads((CLEAN / 'decompose_ge.json').read_text())
    lines = [r"\begin{tabular}{ll r rr}", r"\toprule",
             r"Skill & Outcome & Baseline gap & Mechanical & GE \\", r"\midrule"]
    for mech, ge in zip(data['mechanical'], data['ge']):
        unit = 'pp' if mech['unit'] == 'pp' else r'\%'
        lines.append(f"{mech['skill']} & {mech['outcome']} & {mech['gap']:.1f} {unit} & "
                     f"{mech['share']:.0f}\\% & {ge['share']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / 'tab_decomposition_ge.tex').write_text("\n".join(lines) + "\n")


if __name__ == '__main__':
    write_calibration_table(); write_counterfactual_table(); write_ge_table()
    print("wrote output/tables/tab_calib_params.tex, tab_counterfactuals.tex, tab_decomposition_ge.tex")
