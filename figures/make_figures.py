"""The figures for pilots 6 and 7, drawn straight from the committed result files.

  python figures/make_figures.py          writes figures/fig*.png

fig1_readable_vs_used.png   pilot 6: the value probe reads A's value but carries almost none of the decision
fig2_generic_stage.png      pilot 6: value, apples and weight reach the answer position at the same layer
fig3_route_qwen.png         pilot 7: where option A's (and B's) information sits, layer by layer (Qwen3-14B)
fig3b_route_llama.png       pilot 7: the same for Llama 3.1 8B
fig4_monkey_test.png        pilot 7: what one value direction carries, before and after option B (Qwen3-14B)
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
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
