"""Write layouts/deck.slides.json: one slide per '# slide: <id>' cell, setup cells skipped, speaker notes attached.
Run after editing deck.py (cell order changes):  python3 deck/make_layout.py

--if-empty: only write when the layout file is missing or has no cell data. Older marimo editors (e.g. 0.17)
save the slides layout as {"data": {}}, dropping speaker notes; CI uses this flag to restore them without
overwriting notes edited in a newer editor."""
import json
import re
import sys
from pathlib import Path

DECK = Path(__file__).resolve().parent

NOTES = {
    "title": "20–25 minutes. Plan: thesis → theory in 7 steps → two scatterplots → replication → extension → discussion.",
    "thesis": (
        "Figure 1 is the unweighted mean across the 169 countries (OECD: 35), rebuilt from the UN 2015 data. "
        "It matches the paper: All 0.38 → 0.92 and OECD 0.51 → 1.35 by 2050. The checkbox overlays the 2024 UN revision, "
        "which is nearly identical (OECD 2050: 1.36). Mechanism wording: the cost of automating, C(θ), does not depend on K. "
        "Abundant capital raises the PAYOFF to each automated task (the gap D)."),
    "theory-1": (
        "Two layers: a monopolist supplies every intermediate q(i) at marginal cost 1 and also picks θ; competitive firms "
        "produce tasks; tasks aggregate into Y. Spending on q is a share η of each task's revenue, so the monopolist's revenue is ηY "
        "and its profit is ηχY."),
    "theory-2": (
        "Stress the q(i)^η in BOTH equations: the monopolist earns a share of all output, automated or not. In automated tasks, "
        "k and l are perfect substitutes, so firms use whichever factor is cheaper."),
    "theory-3": (
        "Assumptions: C increasing, strictly convex, C′(0) = 0, C′ → ∞ as θ → 1. The Inada conditions are on the SLOPE; "
        "C itself can stay finite. The functional form −θ − ln(1−θ) is ours, for illustration."),
    "theory-4": (
        "r/w = [θ/K]/[(1−θ)/L] up to the same constant, so (4) ⇔ r < w. Under (4), automated tasks use only capital, and "
        "∂ln Y/∂θ = D > 0. Slider: with K/L = 3, (4) fails once θ > K/(K+L) = 0.75."),
    "theory-5": (
        "Without automation costs, the monopolist would pick θ = K/(K+L). Diminishing returns, not cost, rule out full "
        "automation there. With costs, θ stops while D > 0, which is why the indirect effect is first-order (no envelope theorem: "
        "the firm maximizes profit, not Y). χ is the Lerner index (p−1)/p; the paper's word 'markup' is loose."),
    "theory-6": (
        "Implicit differentiation of D(θ, L) + Γ′(θ) = 0: ∂D/∂θ = −1/(θ(1−θ)), ∂D/∂ln L = −1, so dθ/dln L = 1/(Γ″ − 1/(θ(1−θ))) < 0. "
        "In the illustration, θ* rises from 0.674 (L = 1) to 0.787 (L = 0.4)."),
    "theory-7": (
        "Derivation of the θ > 0.8 bound. At the optimum D = −Γ′ = C′/(ηχ − C) =: x. ε_Γ = θΓ″/Γ′ = θ(C″/C′ + x). "
        "So (1−θ)ε_Γ + 1/D = (1−θ)θC″/C′ + (1−θ)θx + 1/x ≥ 2√(θ(1−θ)) by AM–GM, since C″ > 0. Reversal needs "
        "θ > 2√(θ(1−θ)) ⇔ θ > 0.8. Numerically, the smallest θ* with a reversal we found was 0.86. The paper's text says a large "
        "K–L gap and small ε suffice; it does not state that θ must already be large. Condition (4) gives only D > 0."),
    "evidence-1": (
        "Column 1 of Table 1: 0.335 (0.210), not significant. r = 0.11 (paper data), 0.14 (new data, 1990–2023). "
        "Per-capita vs aggregate: the model's Y is total output; a smaller L mechanically raises output per worker even "
        "without automation if the population is fixed."),
    "evidence-2": (
        "Points recovered from the PDF's vector paths: 49 markers, 31 OECD. The x-values match our aging variable exactly, "
        "and OLS on the recovered points gives slope 0.479 / intercept −0.027, identical to the line drawn in the figure. "
        "r = 0.53 (0.51 without Korea)."),
    "transition": "Each control removes one alternative story. The IV uses birthrates from 1960–90, fixed before the 1990–2015 outcome.",
    "rep-t1": (
        "All 30 cells of Tables 1–2 match: coefficients, SEs, first-stage F, overidentification p-values, N. "
        "Col 4 without Equatorial Guinea gives 0.615 (0.290), as in footnote 4."),
    "rep-t2": (
        "Table 2 instruments are period-specific: three for 1965–90 (1950–65 births), six for 1990–2015. The 1960–65 birthrate "
        "appears in both periods and is absorbed by country fixed effects, leaving 8 effective instruments. "
        "The SE conventions match ivreg2 (no small-sample correction) and reg, robust (HC1)."),
    "ext-data": (
        "Rebuilt aging variables correlate 0.97–0.99 with the originals in levels, but only 0.95 in 1990–2015 changes (the regressor). "
        "Singapore drives much of it (0.47 → 0.18). Mali's 1990–2014 growth changed most across PWT vintages (+1.23 log points)."),
    "ext-results": (
        "Wald: the paper's method. AR (6 instruments): when J is near rejection, the set can be narrower than Wald; it once "
        "excluded zero for the OECD. AR (1 instrument = mean birthrate): pure test of β; every new-data OECD set includes zero. "
        "Panel C col 8 (not shown) gives empty 6-instrument AR sets even in the original data: the instruments disagree."),
    "ext-implications": (
        "The OECD single-instrument point estimates stay around 1.0–1.5: the result loses precision, not size. "
        "Full sample 2023: single-instrument AR set [−0.10, 2.89] (original IVs), [−0.04, 2.70] (shifted). "
        "Not shown (by choice): moderator tests also find the positive association concentrated in capital-poor countries."),
    "sw": "Starter bullets; fill in your own.",
    "thanks": "",
}


def main():
    target = DECK / "layouts" / "deck.slides.json"
    if "--if-empty" in sys.argv and target.exists():
        try:
            if json.loads(target.read_text()).get("data", {}).get("cells"):
                print("layout already has cell data; leaving it as is")
                return
        except json.JSONDecodeError:
            pass
    src = (DECK / "deck.py").read_text()
    cells = re.split(r"\n@app\.cell", src)[1:]
    out = []
    for c in cells:
        m = re.search(r"#\s*slide:\s*([\w-]+)", c)
        if not m:
            out.append({"type": "skip"})
            continue
        sid = m.group(1)
        assert sid in NOTES, f"no notes for slide {sid}"
        out.append({"type": "slide", "speakerNotes": NOTES[sid], "showCode": False})
    (DECK / "layouts").mkdir(exist_ok=True)
    layout = {"type": "slides", "data": {"cells": out, "deck": {"transition": "fade", "verticalAlign": "top"}}}
    (DECK / "layouts" / "deck.slides.json").write_text(json.dumps(layout, indent=1, ensure_ascii=False))
    print(f"{len(out)} cells, {sum(c['type'] == 'slide' for c in out)} slides")


if __name__ == "__main__":
    main()
