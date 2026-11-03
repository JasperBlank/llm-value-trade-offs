"""The figures for pilots 6, 7 and 8, drawn straight from the committed result files.

  python figures/make_figures.py          writes figures/fig*.png

fig1_readable_vs_used.png   pilot 6: the value probe reads A's value but carries almost none of the decision
fig2_generic_stage.png      pilot 6: value, apples and weight reach the answer position at the same layer
fig3_route_qwen.png         pilot 7: where option A's (and B's) information sits, layer by layer (Qwen3-14B)
fig3b_route_llama.png       pilot 7: the same for Llama 3.1 8B
fig4_monkey_test.png        pilot 7: what one value direction carries, before and after option B (Qwen3-14B)
fig5_price_list.png         pilot 8: the price of each of 24 intrinsic values in trade-offs, and its stated rating
fig6_sentences_not_values.png  pilot 8 (exploratory): prices are stable per sentence, not per value
fig7_pilot8_tests.png       pilot 8: the pre-registered tests against their bars
fig8_vibe_check.png         pilot 9: the direct 'how good' rating of each sentence against its menu price
fig9_full_range.png         pilot 10: rating against menu over the full range of goodness, and the log-scale readout
fig10_vibe_direction.png    pilot 11: where the price direction can be read, and what pushing along it does
fig11_recipes.png           pilot 12: the disagreement is readable from the sentence, and pushing it moves the rating beyond the choice
fig12_recipe_in_words.png   pilot 13: the two recipes in named features, how well each predictor prices outcomes, the pushes against random ones
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
P6 = ROOT / "pilot6-value-subspace/results/real"
P7 = ROOT / "pilot7-value-route/results/real"
P8 = ROOT / "pilot8-intrinsic-values/results/real"
P9 = ROOT / "pilot9-vibe-check/results/real"
P10 = ROOT / "pilot10-full-range/results/real"
P11 = ROOT / "pilot11-vibe-direction/results/real"
P12 = ROOT / "pilot12-recipes/results/real"
P13 = ROOT / "pilot13-recipe-in-words/results/real"

# Reference palette (dataviz skill, light mode); slots 1-3 validated as a set.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9ec5f4"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9.5,
})


def load(p):
    return json.load(open(p))


def suptitle(fig, title, subtitle):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def note(fig, text):
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=8, color=MUTED)


def hbars(ax, labels, values, colors, xmax, fmt="{:.2f}", ref=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.56, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(max(v, 0) + xmax * 0.02, yi, fmt.format(v), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def fig1():
    das = load(P6 / "das.json")["layers"]["28"]
    dr = load(P6 / "directions6.json")
    st = load(P6 / "steer6.json")
    dc = st["decision_check"]
    labels = ["Value probe", "Learned"]
    cols = [ORANGE, BLUE]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.6))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.72, bottom=0.2, wspace=0.62)
    hbars(axs[0], labels, [dr["28|probe"]["corr_with_va_heldout"], dr["28|whole_k1"]["corr_with_va_heldout"]], cols, 1.15)
    axs[0].set_title("Tracks option A's value\n(r, held-out prompts)")
    hbars(axs[1], labels, [dc["probe_corr_with_models_logit_diff"], dc["learned_corr_with_models_logit_diff"]], cols, 1.15)
    axs[1].set_title("Tracks the model's own\nA-vs-B lean (r)")
    l3 = labels + ["Random"]
    hbars(axs[2], l3, [das["probe_swap"]["of_whole_swap"]["value"], das["runs"]["whole_k1"]["of_whole_swap"]["value"],
                       max(das["random_swap"]["of_whole_swap"]["value"], 0)], cols + [MUTED], 1.15)
    axs[2].set_title("Share of the decision carried\nwhen swapped between prompts")
    d = st["directions"]
    hbars(axs[3], l3, [d["probe"]["slope_logit_diff"], d["learned"]["slope_logit_diff"], max(d["random"]["slope_logit_diff"], 0)],
          cols + [MUTED], 0.65)
    axs[3].set_title("Push per value unit\n(logit difference)")
    suptitle(fig, "Readable is not the same as used",
             "Qwen3-14B, answer position, layer 28 (pilot 6). The probe reads A's value best, but the model decides along another direction.")
    note(fig, "The learned direction was trained to reproduce the effect of swapping the whole state (distributed alignment search); "
              "all numbers on held-out wordings. Natural sensitivity: 3.22 logit difference per value unit.")
    fig.savefig(OUT / "fig1_readable_vs_used.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig2():
    C = load(P6 / "control.json")["tasks"]
    tasks = [("value_test", "Which outcome should happen? (value)", BLUE, "o"),
             ("quantity", "Which pile has more apples?", ORANGE, "s"),
             ("weight", "Which is heavier?", AQUA, "^")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.17, wspace=0.45)
    for key, lab, col, mk in tasks:
        cond = C[key]["conditions"]
        Ls = sorted(int(k.split("@")[1]) for k in cond if k.startswith("resid@"))
        v = [cond[f"resid@{L}"]["value"] for L in Ls]
        a1.plot(Ls, v, color=col, linewidth=2, marker=mk, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
    a1.set_xticks(Ls)
    a1.set_ylim(-0.03, 1.08)
    a1.set_xlabel("Layer (of 40)")
    a1.set_ylabel("Share carried by the answer position")
    a1.grid(axis="y", color=GRID, linewidth=0.6)
    a1.axvspan(24, 28, color=GRID, alpha=0.5, linewidth=0)
    a1.text(26, 0.42, "arrives\nhere", ha="center", fontsize=8.5, color=INK2)
    a1.legend(loc="upper left")
    a1.set_title("Swapping the whole answer-position state")
    lab2 = ["Value", "Apples", "Weight", "20 random pieces\n(value task)"]
    vals = [C["value_test"]["conditions"]["top20"]["value"], C["quantity"]["conditions"]["top20"]["value"],
            C["weight"]["conditions"]["top20"]["value"], max(C["value_test"]["conditions"]["random20"]["value"], 0)]
    hbars(a2, lab2, vals, [BLUE, ORANGE, AQUA, MUTED], 1.0)
    a2.set_title("Swapping only pilot 5's 20 value pieces\n(blocks 24-27)")
    suptitle(fig, "The decision stage is shared with non-value comparisons",
             "Qwen3-14B (pilot 6). Value, apples and weight reach the answer position at the same layer, through the same pieces.")
    note(fig, "Share = slope of the change in the logit difference on the full effect of the swapped option (0 = none, 1 = all); 1,440 / 216 / 261 pairs.")
    fig.savefig(OUT / "fig2_generic_stage.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
SITES = [("A_end", "A's last token"), ("A_nl", "line break after A"), ("B_line", "B's line"),
         ("B_nl", "line break after B"), ("reply", "\"Reply with ...\" line"), ("template", "chat template"),
         ("last", "answer position")]


def route(model, swaps, fname, title, subtitle):
    mp = load(P7 / model / "map.json")
    Ls = mp["layers"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axs = plt.subplots(1, len(swaps), figsize=(3.3 * len(swaps) + 1.6, 4.3))
    fig.subplots_adjust(left=0.13, right=0.93, top=0.76, bottom=0.14, wspace=0.08)
    axs = np.atleast_1d(axs)
    for ax, (sw, lab) in zip(axs, swaps):
        T = mp["swaps"][sw]["transfer"]
        M = np.array([[max(T[s][str(L)]["value"], 0) for L in Ls] for s, _ in SITES])
        im = ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.25:
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if M[i, j] >= 0.55 else INK)
        ax.set_xticks(range(len(Ls)), Ls, fontsize=8)
        ax.set_yticks(range(len(SITES)), [s[1] for s in SITES] if ax is axs[0] else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(Ls)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITES)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(lab)
        ax.set_xlabel("Layer")
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("share of the swap's effect", fontsize=8.5, color=INK2)
    suptitle(fig, title, subtitle)
    note(fig, "Each cell: the donor's states put into one part of the prompt at one layer; share of the donor's full effect on the answer "
              "(128 held-out pairs per panel). Values below 0.25 unlabelled.")
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig4():
    das = load(P7 / "qwen/das.json")
    tests = [("good_test", "A's good\n(trained on)"), ("number_test", "A's number\n(never seen)"),
             ("bnumber_test", "B's number"), ("quantity", "Apples"), ("weight", "Weight")]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.2, wspace=0.12)
    for ax, (role, ttl) in zip(axs, [("a_site", "Before B: A's last token, layer 12"),
                                     ("b_site", "After B: the text after B's line, layer 24")]):
        ev = das["sites"][role]["eval"]["good_trained"]
        x = np.arange(len(tests))
        site = [max(ev[t]["site_swap_of_donor"]["value"], 0) for t, _ in tests]
        one = [max(ev[t]["of_donor"]["value"], 0) for t, _ in tests]
        w = 0.36
        ax.bar(x - w / 2, site, w, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, label="the whole site swapped")
        ax.bar(x + w / 2, one, w, color=BLUE, edgecolor=SURFACE, linewidth=2, label="only the one value direction swapped")
        ax.set_xlim(-0.6, len(tests) - 0.4)
        for xi, (s, o) in enumerate(zip(site, one)):
            ax.text(xi - w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8, color=INK2)
            ax.text(xi + w / 2, o + 0.02, f"{o:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [t[1] for t in tests], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(ttl)
    axs[0].set_ylabel("Share of the swap's effect on the answer")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.45, 1.0))
    suptitle(fig, "One value direction appears only after the model has read both options",
             "Qwen3-14B (pilot 7). The direction was trained on swaps of A's good only, then tested on changes it never saw.")
    note(fig, "Before B, A's last token holds the good but not the number (0.00). After B, the direction carries good and number changes and B's "
              "value (opposite sign: a comparison), but little of apples or weight.")
    fig.savefig(OUT / "fig4_monkey_test.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Pilot 8: trade-offs between 24 intrinsic values (Qwen3-14B)
# --------------------------------------------------------------------------
def fig5():
    beh = load(P8 / "qwen/behave.json")
    b = beh["b"]
    names = sorted(beh["worth"], key=lambda v: beh["worth"][v]["u"])
    ppl = [beh["worth"][v]["people_vs_mean"] for v in names]
    lo = [np.exp(beh["worth"][v]["ci95"][0] / b) for v in names]
    hi = [np.exp(beh["worth"][v]["ci95"][1] / b) for v in names]
    rate = [beh["ratings"][v] for v in names]
    y = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(11, 7.4), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.12, wspace=0.08)
    ax = axs[0]
    ax.hlines(y, lo, hi, color=BLUE_LIGHT, linewidth=3)
    ax.plot(ppl, y, "o", color=BLUE, markersize=6)
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3], ["¼", "½", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_xlim(0.25, 3.6)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("In trade-offs: one person counts as … people of an average value")
    ax2 = axs[1]
    ax2.barh(y, rate, color=ORANGE, height=0.56, edgecolor=SURFACE, linewidth=1.5)
    for yi, r in zip(y, rate):
        ax2.text(r + 0.1, yi, f"{r:.1f}", va="center", fontsize=8.5, color=INK2)
    ax2.set_xlim(0, 9)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("Asked directly: how much it matters (0-9)")
    suptitle(fig, "Qwen3-14B's price list for 24 intrinsic values",
             f"Pilot 8. Left: fitted from 13,248 choices ('A) 2 people each …  B) 8 people each …'), 95% intervals. "
             f"Right: stated rating. They agree only r = {beh['ratings_vs_worth_r']:.2f}.")
    note(fig, "Values from the Clearer Thinking intrinsic-values list, two sentences each. The two sentences written for one value agree\n"
              f"only r = {beh['phrasing_reliability_r']:.2f} (see the next figure), so read these as prices of the sentences used, not of the value in general.")
    fig.savefig(OUT / "fig5_price_list.png", dpi=200)
    plt.close(fig)


def fig6():
    ex = load(P8 / "exploratory_sentences.json")
    S = ex["sentences"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.1)
    lim = (-4.2, 3.9)
    ax = axs[0]
    ax.plot(lim, lim, color=AXIS, linewidth=1)
    ax.plot([s["u_wording0"] for s in S], [s["u_wording1"] for s in S], "o", color=BLUE, markersize=5, alpha=0.85)
    ax.set_title(f"Same sentence, two question wordings\nr = {ex['same_sentence_across_wordings_r']:.2f}")
    ax.set_xlabel("Price, asked 'Which … would you rather happen?'")
    ax.set_ylabel("Price, asked 'Which … do you prefer?'")
    ax = axs[1]
    ax.plot(lim, lim, color=AXIS, linewidth=1)
    by = {}
    for s in S:
        by.setdefault(s["value"], {})[s["phrasing"]] = s["u"]
    xs = np.array([by[v][0] for v in by])
    ys = np.array([by[v][1] for v in by])
    ax.plot(xs, ys, "o", color=ORANGE, markersize=5.5)
    gap = sorted(by, key=lambda v: -abs(by[v][0] - by[v][1]))[:5]
    for v in gap:
        ax.annotate(v, (by[v][0], by[v][1]), xytext=(5, 3), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_title(f"Two sentences written for the same value\nr = {ex['two_sentences_of_a_value_r']:.2f}")
    ax.set_xlabel("Price of the value's first sentence")
    ax.set_ylabel("Price of its second sentence")
    for a in axs:
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_aspect("equal")
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    suptitle(fig, "The model has stable prices for situations, not for value categories",
             "Pilot 8, exploratory (after the results). Price = worth per person in log-odds, fitted per sentence from the same 13,248 choices.")
    note(fig, "Example: Diversity 'each become part of a community made up of many different kinds of people' +2.2, but "
              "'each get to know someone very different from themselves' −2.6.")
    fig.savefig(OUT / "fig6_sentences_not_values.png", dpi=200)
    plt.close(fig)


def fig7():
    V = load(P8 / "verdicts8.json")
    beh = load(P8 / "qwen/behave.json")
    src = load(P8 / "qwen/source.json")
    rat = load(P8 / "qwen/rating.json")
    rows = [
        ("Q2b  One comparison direction (after option B, layer 24)\n       carries trade-offs between 12 values it never saw", V["Q2b"]["share"], 0.5, "share"),
        ("Q2a  A worth direction on option A's own words (layer 20)\n       orders 12 values it never saw", V["Q2a"]["r"], 0.7, "r"),
        ("Q4b  The same direction, in 'how much does it matter?'\n       prompts, orders those 12 values", V["Q4b"]["r"], 0.7, "r"),
        ("Q4c  Swapping only that direction carries the rating", V["Q4c"]["share"], 0.3, "share"),
        ("Q4a  Stated ratings match the trade-off prices", V["Q4a"]["r"], 0.7, "r"),
        ("Q3   Prices from one question wording predict the other", beh["one_scale"]["heldout_wording_r"], 0.8, "r"),
        ("Q3   Choices are transitive (share of triads not cyclic)", 1 - beh["transitivity"]["share"], 0.95, "share"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0.43, right=0.86, top=0.84, bottom=0.15)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, bar, unit) in zip(y, rows):
        ok = v >= bar
        ax.barh(yi, v, color=BLUE if ok else ORANGE, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.plot([bar, bar], [yi - 0.38, yi + 0.38], color=INK, linewidth=1.6)
        ax.text(1.03, yi, f"{unit} {v:.2f}  {'pass' if ok else 'fail'}", va="center", fontsize=9, color=INK,
                transform=ax.get_yaxis_transform(), fontweight="bold" if ok else "normal")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    suptitle(fig, "A shared comparison step, but no per-value worth code before it",
             "Pilot 8, Qwen3-14B: the pre-registered tests. Black tick = the bar set before the run; blue = passed, orange = failed.")
    note(fig, f"Not shown: adding a third option moved the A:B odds by a median factor of {beh['menu']['median_factor']:.0f} (bar 1.5), likely a "
              f"position effect of the 3-option format.\nOption A's own words carry only {src['site_swap_share']:.2f} of a value swap; "
              f"the rating prompt's site {rat['site_swap_of_donor']['value']:.2f} of a rating difference.")
    fig.savefig(OUT / "fig7_pilot8_tests.png", dpi=200)
    plt.close(fig)


def fig8(p9=None):
    """Pilot 9: the direct 'how good would it be' rating of each sentence against its menu price."""
    p9 = Path(p9) if p9 else P9
    R = load(p9 / "ratings9.json")
    V = load(p9 / "verdicts9.json")
    ex = load(P8 / "exploratory_sentences.json")
    sent = {(s["value"], s["phrasing"]): s for s in ex["sentences"]}
    rate = {(v, p): x for (v, p, n, w), x in zip(R["keys"], R["rating"]) if n == 4 and w == 0}
    names = sorted({k[0] for k in rate})
    t = V["per_wording"]["W0 how good"]
    at7 = np.mean([abs(x - 7) < 0.1 for (v, p, n, w), x in zip(R["keys"], R["rating"]) if w == 0])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.065, right=0.98, top=0.8, bottom=0.15, wspace=0.22)
    # left: every sentence
    ax = axs[0]
    ax.axhspan(6.9, 7.1, color=BLUE_LIGHT, alpha=0.35, linewidth=0, zorder=0)
    ax.text(-3.9, 7.2, f"'7': {at7:.0%} of all answers", fontsize=8.5, color=INK2)
    x = np.array([sent[k]["u"] for k in rate])
    y = np.array([rate[k] for k in rate])
    ax.plot(x, y, "o", color=BLUE, markersize=6, zorder=2)
    labels = {("Longevity", 0): "live a year longer", ("Protection", 0): "kept safe from a danger",
              ("Respect", 1): "receive the recognition they deserve", ("Happiness", 0): "feel happier for a week"}
    for k, lab in labels.items():
        right = sent[k]["u"] > 1.5 or k == ("Respect", 1)
        ax.annotate(lab, (sent[k]["u"], rate[k]), xytext=(-7 if right else 7, -3 if k[0] != "Respect" else 4),
                    textcoords="offset points", fontsize=8.5, color=INK2, ha="right" if right else "left",
                    va="top" if k[0] != "Respect" else "center")
    ax.set_xlim(-4, 3.5)
    ax.set_xlabel("Menu price of the sentence (log-odds per person, pilot 8)")
    ax.set_ylabel("'How good would it be if …?' (0-9, 4 people)")
    ax.set_title(f"Each of the 48 sentences: r = {t['t1_r']:.2f}")
    # right: the two sentences of each value
    ax = axs[1]
    du = np.array([sent[(v, 0)]["u"] - sent[(v, 1)]["u"] for v in names])
    dr = np.array([rate[(v, 0)] - rate[(v, 1)] for v in names])
    lim_u, lim_r = 5.2, 2.0
    ax.fill_between([0, lim_u], 0, lim_r, color=AQUA, alpha=0.08, linewidth=0)
    ax.fill_between([-lim_u, 0], -lim_r, 0, color=AQUA, alpha=0.08, linewidth=0)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    agree = np.sign(du) == np.sign(dr)
    ax.plot(du[agree], dr[agree], "o", color=BLUE, markersize=6.5)
    ax.plot(du[~agree], dr[~agree], "o", color=ORANGE, markersize=6.5)
    for v in ("Respect", "Beauty", "Nurturing", "Longevity", "Non-suffering", "Diversity"):
        i = names.index(v)
        ns = v == "Non-suffering"
        ax.annotate(v + (" (the pain pair)" if ns else ""), (du[i], dr[i]), xytext=(-6 if ns else 6, 6 if ns else 4),
                    textcoords="offset points", fontsize=8.5, color=INK2, ha="right" if ns else "left")
    ax.set_xlim(-lim_u, lim_u)
    ax.set_ylim(-lim_r, lim_r)
    ax.set_xlabel("Menu: price of a value's 1st sentence minus its 2nd")
    ax.set_ylabel("Rating: 1st sentence minus 2nd")
    ax.set_title(f"Within each value: the same sentence wins in {t['t2_same']} of 24")
    for a in axs:
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    suptitle(fig, "Asked directly, the model shares the menu's wording quirks, but its ratings are squashed towards 7",
             "Pilot 9, Qwen3-14B. Pre-registered: T1 failed (r 0.55, bar 0.7); T2 passed (20 of 24, bar 17); "
             "8 people rated only +0.43 above 2 people.")
    note(fig, "Right: shaded quarters = rating and menu prefer the same sentence (blue); orange = they disagree. "
              "Many within-value rating differences are small because most answers sit at 7.")
    fig.savefig(OUT / "fig8_vibe_check.png", dpi=200)
    plt.close(fig)

def fig9(p10=None, title=None, subtitle=None):
    """Pilot 10: the 'how good' rating against the menu price over the full range, and pilot 9's sentences on a log scale."""
    p10 = Path(p10) if p10 else P10
    V = load(p10 / "verdicts10.json")
    prices = load(p10 / "prices10.json")
    R = load(p10 / "ratings10.json")
    S = load(ROOT / "pilot10-full-range/sentences10.json")["sentences"]
    ex = load(P8 / "exploratory_sentences.json")
    p8u = {(s["value"], s["phrasing"]): s["u"] for s in ex["sentences"]}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.065, right=0.98, top=0.8, bottom=0.15, wspace=0.22)
    ax = axs[0]
    col = {"small": AQUA, "medium": BLUE, "large": ORANGE}
    for lev in ("small", "medium", "large"):
        pts = [d for d in prices if d["level"] == lev]
        ax.plot([d["u"] for d in pts], [d["rating_W0_8"] for d in pts], "o", color=col[lev], markersize=6,
                label={"small": "small version", "medium": "pilot 8's sentences", "large": "large version"}[lev])
    ax.set_xlabel("Menu price (log-odds per person, 45,600 choices)")
    ax.set_ylabel("'How good would it be if …?' (0-9, 8 people)")
    ax.set_title(f"All 96 sentences: r = {V['F1']['r']:.2f}")
    ax.legend(loc="upper left")
    ax = axs[1]
    sc = {(k[0], k[1], k[2]): x for k, x in zip(R["keys"], R["score"])}
    mid = [i for i, s in enumerate(S) if s["level"] == "medium"]
    x = [p8u[(S[i]["value"], S[i]["p8"])] for i in mid]
    y = [sc[(i, 4, 0)] for i in mid]
    ax.plot(x, y, "o", color=BLUE, markersize=6)
    ax.set_xlabel("Pilot 8's menu price of the sentence")
    ax.set_ylabel("Log-scale score of 'How good …' (4 people)")
    ax.set_title(f"Pilot 9's 48 sentences on a log scale: r = {V['F2']['r']:.2f}")
    for a in axs:
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    suptitle(fig, title or "Rating and menu over the full range of goodness",
             subtitle or f"Pilot 10, Qwen3-14B. F1 r = {V['F1']['r']:.2f} (bar 0.8); F2 r = {V['F2']['r']:.2f} (bar 0.7); "
             f"32 people rated {V['F3']['gain_2_to_32']:+.2f} above 2 people.")
    note(fig, "Right: the log-scale score keeps the ratios between small probabilities (0.1% against 0.4% on '8') that the expected rating flattens.")
    fig.savefig(OUT / "fig9_full_range.png", dpi=200)
    plt.close(fig)


def fig10(p11=None):
    """Pilot 11: where the vibe direction can be read (held out), and what pushing along it does, layer by layer."""
    p11 = Path(p11) if p11 else P11
    rep = open(p11 / "report.md").read().splitlines()
    rows = [l.split("|")[1:-1] for l in rep if l.startswith("| ") and l.split("|")[1].strip().isdigit()]
    read = [[float(x) for x in r] for r in rows if len(r) == 10]
    steer = [r for r in rows if len(r) == 6]
    L = [int(r[0]) for r in read]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.78, bottom=0.15, wspace=0.22)
    ax = axs[0]
    ax.plot(L, [r[1] for r in read], "-o", color=BLUE, markersize=4, label="menu price, read as option A (before B)")
    ax.plot(L, [r[3] for r in read], "-o", color=AQUA, markersize=4, label="menu price, read in the rating prompt")
    ax.plot(L, [r[4] for r in read], "-o", color=ORANGE, markersize=4, label="the rating, read in the rating prompt")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out r (one value left out)")
    ax.set_title("Reading the sentence: the price is there, in both prompts")
    ax.legend(loc="lower right")
    ax = axs[1]
    vib = [(int(r[0]), float(r[2].split("(")[0]), float(r[3])) for r in steer if r[1].strip() == "vibe"]
    rnd = [(int(r[0]), float(r[2].split("(")[0]), float(r[3])) for r in steer if r[1].strip() == "random"]
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--")
    ax.plot([v[0] for v in vib], [v[1] for v in vib], "-o", color=BLUE, markersize=4, label="the choice (menu)")
    ax.plot([v[0] for v in vib], [v[2] for v in vib], "-o", color=ORANGE, markersize=4, label="the stated rating")
    for v in rnd:
        ax.plot([v[0]], [v[1]], "x", color=INK2, markersize=7)
        ax.annotate("random direction", (v[0], v[1]), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("Layer pushed")
    ax.set_ylabel("Share of the predicted change")
    ax.set_title("Pushing along it: the choice moves, the rating barely")
    ax.legend(loc="upper right")
    for a in axs:
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    suptitle(fig, "A price direction on the sentence: readable, partly used for the choice, hardly for the rating",
             "Pilot 11, Qwen3-14B, 96 sentences. Direction = difference of means between the highest- and lowest-priced thirds; pushed over the sentence's words by ±2 price units.")
    note(fig, "Dashed line: the pre-registered bar (0.5). After layer 24 pushing the sentence does nothing: by then its information has moved on to later positions.")
    fig.savefig(OUT / "fig10_vibe_direction.png", dpi=200)
    plt.close(fig)


def fig11(p12=None):
    """Pilot 12: the disagreement between rating and price can be read from the sentence, and pushing along it moves
    the rating beyond what one shared variable would allow."""
    p12 = Path(p12) if p12 else P12
    V = load(p12 / "verdicts12.json")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=0.065, right=0.98, top=0.78, bottom=0.15, wspace=0.25)
    ax = axs[0]
    q1 = V["Q1"]
    L = sorted(int(k) for k in q1["r_by_layer_menu"])
    ax.axhspan(-0.1, q1["null_95"], color=GRID, alpha=0.6, linewidth=0)
    ax.text(L[-1], q1["null_95"] - 0.012, "95% of runs with a shuffled target fall in this band", fontsize=8.5, color=INK2, va="top", ha="right")
    ax.plot(L, [q1["r_by_layer_menu"][str(x)] for x in L], "-o", color=BLUE, markersize=4, label="read as option A (no rating question)")
    ax.plot(L, [q1["r_by_layer_rating_ctx"][str(x)] for x in L], "-o", color=AQUA, markersize=4, label="read in the rating prompt")
    ax.set_ylim(-0.1, 0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out r (one value left out)")
    ax.set_title("Where the rating disagrees with the price\ncan be read from the sentence")
    ax.legend(loc="upper left")
    ax = axs[1]
    P = V["pushes"]
    sr = V["Q2"]["rating_per_logit_price_direction"]
    xmax = 0.9
    ax.plot([0, xmax], [0, sr * xmax], color=AXIS, linewidth=1.2, linestyle="--")
    ax.text(0.42, sr * 0.42 - 0.012, "one shared vibe:\nevery push on this line", fontsize=8.5, color=INK2, ha="left", va="top")
    col = {"price": BLUE, "rating": AQUA, "disagreement": ORANGE, "random": INK2}
    lab = {"price": "price direction", "rating": "rating direction", "disagreement": "disagreement direction", "random": "random direction"}
    for k, v in P.items():
        x, y = v["choice_logit_per_2_price_units"], v["rating_points_per_2_price_units"]
        ax.plot([x], [y], "o", color=col[k], markersize=8)
        right = x > 0.6
        ax.annotate(lab[k], (x, y), xytext=(-9 if right else 8, 4 if k != "random" else -12), textcoords="offset points",
                    fontsize=9, color=INK, ha="right" if right else "left")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(-0.05, xmax)
    ax.set_ylim(-0.03, 0.16)
    ax.set_xlabel("Change in the choice (logit of picking the sentence), per push")
    ax.set_ylabel("Change in the stated rating (points), per push")
    ax.set_title("Pushing along the disagreement moves the rating\nfar more than its small effect on the choice would")
    for a in axs:
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    suptitle(fig, "Shared ingredients, different recipes",
             f"Pilot 12, Qwen3-14B, 96 sentences. Q1: held-out r {q1['r']:.2f} at layer {q1['layer']}, permutation p {q1['p']:.3f}. "
             f"Q2: the disagreement push lies above the shared-vibe line (95% interval above 0).")
    note(fig, "Pushes at layer 20, all by the same distance. The random direction also falls off the line (below it), so the line is a rough benchmark.")
    fig.savefig(OUT / "fig11_recipes.png", dpi=200)
    plt.close(fig)


LABELS13 = {"severity": "how serious the need", "duration": "how long it lasts", "health_safety": "health and safety",
            "moral": "righting a moral wrong", "relationships": "relationships and care", "meaning": "meaning and purpose",
            "pleasure": "pleasure and fun", "legacy": "lasting for others", "concreteness": "concrete and certain", "size": "size (F1)"}


def lovo_ols(F, y, groups):
    """Held-out least-squares predictions, one value left out at a time (as pilot13.py)."""
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        A = np.column_stack([np.ones(tr.sum()), F[tr]])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum()), F[te]]) @ beta
    return pred


def fig12(p13=None):
    """Pilot 13: the choice recipe in words. The two recipes (features rated blind by Claude; exploratory), how well
    each predictor prices Qwen's outcomes, and the pushes at layer 20 against 20 random directions."""
    p13 = Path(p13) if p13 else P13
    V = load(p13 / "verdicts13.json")
    E = load(p13 / "external_ratings.json")
    D = load(p13 / "features13.json")
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.8), gridspec_kw={"width_ratios": [1.0, 1.05, 1.1]})
    fig.subplots_adjust(left=0.105, right=0.985, top=0.77, bottom=0.15, wspace=0.55)
    # A: the two recipes, features rated blind by Claude
    ax = axs[0]
    W = E["claude"]["weights"]
    words = list(W)
    y = np.arange(len(words))[::-1]
    for yi, n in zip(y, words):
        w = W[n]
        ax.plot([w["price"], w["rating"]], [yi, yi], color=AXIS, linewidth=1.6, zorder=1)
        ax.plot([w["price"]], [yi], "o", color=BLUE, markersize=7, zorder=2, label="the choice (price)" if yi == y[0] else None)
        ax.plot([w["rating"]], [yi], "o", color=ORANGE, markersize=7, zorder=2, label="the stated rating" if yi == y[0] else None)
        lo, hi = w["diff_ci95"]
        if lo > 0 or hi < 0:
            ax.text(max(w["price"], w["rating"]) + 0.03, yi, "*", va="center", fontsize=12, color=INK)
    ax.set_yticks(y, [LABELS13[n] for n in words])
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_ylim(-0.6, len(words) + 0.5)
    ax.set_xlabel("Standardised weight (nine features fitted together)")
    ax.set_title("The two recipes\n(features rated blind by Claude)")
    ax.legend(loc="upper center", ncol=2, handletextpad=0.2, columnspacing=1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    # B: how well each predictor prices Qwen's outcomes
    ax = axs[1]
    old = [d for d in D if d["set"] == "old"]
    r_rating_old = float(np.corrcoef([d["Rbar"] for d in old], [d["u"] for d in old])[0, 1])
    o = V["P4"]["other_predictors_r"]
    rows = [("nine features, Qwen's own answers", V["P1"]["r"], V["P4"]["r"]),
            ("nine features, rated blind by Claude", E["claude"]["P1_r"], E["claude"]["P4_r"]),
            ("one question: how big a difference", V["P1"]["r_F1_alone"], o["F1_alone"]),
            ("Qwen's states at layer 20 (ridge)", V["P4"]["ridge_heldout_r_on_old"], o[[k for k in o if k.startswith("states_ridge")][0]]),
            ("Qwen's stated 'how good' rating", r_rating_old, o["how_good_rating"])]
    yb = np.arange(len(rows))[::-1] * 1.0
    h = 0.36
    ax.barh(yb + h / 2, [r[1] for r in rows], height=h, color=BLUE, label=f"pilot 10's {len(old)} (held out by value)")
    ax.barh(yb - h / 2, [r[2] for r in rows], height=h, color=ORANGE, label=f"the {V['n_new']} new sentences")
    for yi, r in zip(yb, rows):
        ax.text(r[1] + 0.01, yi + h / 2, f"{r[1]:.2f}", va="center", fontsize=9, color=INK)
        ax.text(r[2] + 0.01, yi - h / 2, f"{r[2]:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(yb, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(rows) - 0.3 + 0.9)
    ax.set_xlabel("r with Qwen's measured price")
    ax.set_title("Every predictor gets about 0.8 on the old\nsentences and about 0.7 on new ones")
    ax.legend(loc="upper center", ncol=2, handletextpad=0.3, columnspacing=1.0, fontsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    # C: pushes along each direction against 20 random ones
    ax = axs[2]
    P = V["pushes"]
    sr = V["rating_per_logit_price_direction"]
    xs = [v["choice_logit_per_push"] for v in P.values()]
    x0, x1 = min(xs + [0]) - 0.05, max(xs) * 1.12 + 0.05
    ax.plot([x0, x1], [sr * x0, sr * x1], color=AXIS, linewidth=1.2, linestyle="--")
    rnd = [v for k, v in P.items() if k.startswith("random:")]
    ax.plot([v["choice_logit_per_push"] for v in rnd], [v["rating_points_per_push"] for v in rnd], "x", color=MUTED, markersize=6,
            label="20 random directions")
    feats = {k: v for k, v in P.items() if k.startswith("feature:")}
    ax.plot([v["choice_logit_per_push"] for v in feats.values()], [v["rating_points_per_push"] for v in feats.values()], "o",
            color=AQUA, markersize=5, label="the ten features' directions")
    for k, off in (("feature:concreteness", (-6, -3)), ("feature:health_safety", (5, -11))):
        v = feats[k]
        ax.annotate(LABELS13[k.split(":")[1]], (v["choice_logit_per_push"], v["rating_points_per_push"]), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK2, ha="right" if off[0] < 0 else "left")
    for k, col, lab, off in (("price", BLUE, "price", (6, -13)), ("rating", INK2, "rating", (-8, 5)),
                             ("disagreement", ORANGE, "disagreement (pilot 12)", (8, 4))):
        ax.plot([P[k]["choice_logit_per_push"]], [P[k]["rating_points_per_push"]], "o", color=col, markersize=8)
        ax.annotate(lab, (P[k]["choice_logit_per_push"], P[k]["rating_points_per_push"]), xytext=off, textcoords="offset points",
                    fontsize=9, color=INK, ha="right" if off[0] < 0 else "left")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(x0, x1)
    ax.set_xlabel("Change in the choice (logit), per push")
    ax.set_ylabel("Change in the stated rating (points), per push")
    ax.set_title("Pushes at layer 20, all by the same distance")
    ax.legend(loc="upper left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    q = E["qwen"]["halo_mean_r_among_F2_F10"]
    c = E["claude"]["halo_mean_r_among_F2_F10"]
    suptitle(fig, "Why the model picks what it picks, in words",
             f"Pilot 13, Qwen3-14B. Pre-registered: P1 r {V['P1']['r']:.2f} (bar 0.8, fail); P2 fail; P3 pass; P4 r {V['P4']['r']:.2f} (bar 0.7, pass). "
             f"Qwen's own feature answers are entangled (mean r between features {q:.2f}; Claude's blind ratings {c:.2f}),\n"
             "so panel A uses the blind ratings (exploratory, planned before the results).")
    note(fig, "* the stated rating and the choice weigh this feature differently (95% bootstrap interval excludes 0). Middle: the rating's r on the "
              "old sentences is a plain correlation (nothing fitted). Right: a push that changes only the shared price lands on the dashed line.")
    fig.savefig(OUT / "fig12_recipe_in_words.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    route("qwen", [("good", "A's good changes"), ("number", "A's number of people changes"),
                   ("bnumber", "B's number of people changes"), ("quantity", "Apples (non-value)")],
          "fig3_route_qwen.png", "Where the options' information sits on its way to the answer",
          "Qwen3-14B (pilot 7). Good and number take different paths, meet at the line breaks around layers 20-24, then jump to the answer at 28.")
    route("llama", [("good", "A's good changes"), ("number", "A's number of people changes"),
                    ("bnumber", "B's number of people changes"), ("quantity", "Apples (non-value)")],
          "fig3b_route_llama.png", "The same route in Llama 3.1 8B, earlier",
          "Llama 3.1 8B (pilot 7). Line breaks at layers 8-12, answer position from layer 16 (of 32).")
    fig4()
    fig5()
    fig6()
    fig7()
    if (P9 / "verdicts9.json").exists():
        fig8()
    if (P10 / "verdicts10.json").exists():
        fig9()
    if (P11 / "report.md").exists():
        fig10()
    if (P12 / "verdicts12.json").exists():
        fig11()
    if (P13 / "verdicts13.json").exists():
        fig12()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
