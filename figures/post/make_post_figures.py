"""Figures for the LessWrong post, drawn from the committed result files.

  python figures/post/make_post_figures.py            all figures, and CAPTIONS.md
  python figures/post/make_post_figures.py hero_x     just one

writes figures/post/*.png (2x), the HTML each one is rendered from (figures/post/html/), and CAPTIONS.md, which holds
the details kept out of the images. Charts are matplotlib SVG inside an HTML frame; the diagrams are plain HTML.
Rendering uses Playwright's Chromium.

Design rules (deliberate; see README.md):
- one message per figure, in a title of about eight words at most, and at most one short line under it
- no kicker labels, emoji, cards, shadows, pills or stat blocks; a white background; space and hairlines for grouping
- Source Serif 4 for titles, Libre Franklin for everything else (fonts/, SIL Open Font License)
- colour only where it means something: blue = option A or outcomes, orange = option B or apples, aqua = weight,
  red = the deleted word
- method details live in CAPTIONS.md, not in the image

hero_x.png           the main figure, 1600 x 900 (for X and the top of the post)
f1_same_value.png    pilot 10: outcomes of one value get very different prices
f2_vibe_check.png    pilot 13: the "how good would it be?" rating tracks the price about as well as anything else
f3_one_word.png      pilot 14: single words move the choice the way their meaning says
f4_cycle.png         pilot 16: a preference cycle that no fixed ranking allows
f5_apples.png        pilot 17: outcomes, apples and weights are decided at the same depth, by the same parts
f6_price_timing.png  pilot 17: a rough price after the first option, finished after the second
"""
import html
import json
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FONTS = HERE / "fonts"
P10 = ROOT / "pilot10-full-range"
P13 = ROOT / "pilot13-recipe-in-words/results/real"
P14 = ROOT / "pilot14-price-writers/results/real"
P16 = ROOT / "pilot16-relative-or-absolute/results/real"
P17 = ROOT / "pilot17-three-models/results"
HTML_DIR = HERE / "html"
for _f in sorted(FONTS.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_f))

# The menu as the models saw it (pilot 10's build_prompts10.py).
CONTEXT = "The people involved are strangers in your town whom you will never meet."
QUESTION = "Which of these two outcomes would you rather happen?"

INK, INK2, MUTED, RULE, AXIS = "#141414", "#555555", "#8c8c8c", "#e6e6e6", "#bdbdbd"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d1453b"
NEUTRAL = "#f1f1f1"
LEVEL = {"small": "#86b6ef", "medium": "#2a78d6", "large": "#104281"}   # blue ramp steps 250 / 450 / 650
LIGHT_BLUE = "#86b6ef"

plt.rcParams.update({
    "font.family": "Libre Franklin", "svg.fonttype": "none", "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8, "legend.frameon": False,
    "legend.fontsize": 11.5, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load(p):
    return json.load(open(p))


def esc(s):
    return html.escape(s, quote=True)


def sig(x):
    return 1 / (1 + np.exp(-x))


def pct(p):
    """Probability as a percentage, with as many decimals as it takes not to round up to 100%."""
    if p > 0.9999:
        return "> 99.99%"
    for d in (0, 1, 2):
        s = f"{p * 100:.{d}f}"
        if float(s) < 100 and (d > 0 or p < 0.99):
            return s + "%"
    return "> 99.99%"


# ---------------------------------------------------------------- page frame
def font_faces():
    faces = [("Libre Franklin", "libre-franklin", w, s) for w, s in
             [(400, "normal"), (500, "normal"), (600, "normal"), (700, "normal"), (400, "italic")]]
    faces += [("Source Serif 4", "source-serif-4", w, s) for w, s in
              [(400, "normal"), (600, "normal"), (700, "normal"), (400, "italic")]]
    return "".join(f"@font-face {{ font-family:'{fam}'; src:url('../fonts/{stem}-{w}-{st}.ttf'); font-weight:{w}; "
                   f"font-style:{st}; }}\n" for fam, stem, w, st in faces)


CSS = font_faces() + """
:root { --ink:#141414; --ink2:#555555; --muted:#8c8c8c; --rule:#e6e6e6; --blue:#2a78d6; --orange:#eb6834;
        --aqua:#1baf7a; --red:#d1453b; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#fff; font-family:'Libre Franklin', sans-serif; color:var(--ink); -webkit-font-smoothing:antialiased; }
#fig { background:#fff; padding:44px 52px 34px; }
h1 { font-family:'Source Serif 4', serif; font-weight:600; font-size:38px; line-height:1.12; letter-spacing:-.006em; }
.sub { font-size:18px; color:var(--ink2); margin-top:8px; line-height:1.4; }
.chart { margin-top:18px; }
.chart svg { display:block; }
.src { font-size:13.5px; color:var(--muted); margin-top:10px; }
"""


def page(body, width=1200, extra_css=""):
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + extra_css + "</style></head><body>"
            f"<div id='fig' style='width:{width}px'>" + body + "</div></body></html>")


def header(title, sub=None):
    return f"<h1>{title}</h1>" + (f"<div class='sub'>{sub}</div>" if sub else "")


def svg_of(fig):
    buf = StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def dot_legend(ax, colors, labels, **kw):
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=8.5, color=c, markeredgecolor="white") for c in colors]
    return ax.legend(hs, labels, handletextpad=0.2, columnspacing=1.2, **kw)


# ---------------------------------------------------------------- F1: one value, many prices
F1_ROWS = ["Respect", "Nurturing", "Happiness", "Justice"]
F1_LABEL = {  # each value's cheapest and dearest outcome, as worded after "8 people each"
    "each get a polite nod from a stranger": "get a polite nod from a stranger",
    "each finally receive the recognition and respect they deserve": "finally receive the recognition and respect they deserve",
    "each help a child tie their shoes": "help a child tie their shoes",
    "each raise a child who grows up healthy, kind and happy": "raise a child who grows up healthy, kind and happy",
    "each smile at a funny joke": "smile at a funny joke",
    "each become deeply happy for the rest of their lives": "become deeply happy for the rest of their lives",
    "each see someone who cut in line sent to the back": "see someone who cut in line sent to the back",
    "each see a person who harmed them face fair consequences": "see a person who harmed them face fair consequences",
}


def f1_data():
    P = load(P10 / "results/real/prices10.json")
    byv = {}
    for d in P:
        byv.setdefault(d["value"], []).append(d)
    means = [np.mean([d["u"] for d in L]) for L in byv.values()]
    spans = [max(d["u"] for d in L) - min(d["u"] for d in L) for L in byv.values()]
    return P, byv, max(means) - min(means), float(np.median(spans)), max(spans)


def fig_f1():
    P, byv, *_ = f1_data()
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.92, bottom=0.12)
    ys = np.arange(len(F1_ROWS) + 1) * 1.0
    u_all = np.array([d["u"] for d in P])
    ax.scatter(u_all, np.zeros_like(u_all), s=24, color=AXIS, linewidths=0, zorder=2)
    for y, v in zip(ys[1:], F1_ROWS):
        L = sorted(byv[v], key=lambda d: d["u"])
        ax.plot([L[0]["u"], L[-1]["u"]], [y, y], color=RULE, lw=3, zorder=1, solid_capstyle="round")
        for d in L:
            ax.scatter([d["u"]], [y], s=105, color=LEVEL[d["level"]], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(L[0]["u"] - 0.1, y + 0.33, F1_LABEL[L[0]["text"]], ha="left", va="center", fontsize=11.5, color=INK2)
        ax.text(L[-1]["u"] + 0.1, y - 0.33, F1_LABEL[L[-1]["text"]], ha="right", va="center", fontsize=11.5, color=INK2)
    ax.set_yticks(ys, ["all 96"] + F1_ROWS)
    for t in ax.get_yticklabels()[1:]:
        t.set_color(INK)
        t.set_fontsize(12.5)
        t.set_fontweight("medium")
    ax.get_yticklabels()[0].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(F1_ROWS) + 0.6, -0.6)
    ax.set_xlim(-10.0, 7.0)
    ax.set_xticks([-8, -4, 0, 4])
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("price in forced choices", labelpad=6)
    dot_legend(ax, list(LEVEL.values()), ["small", "medium", "large version"], loc="lower right", ncol=3,
               bbox_to_anchor=(1.0, 0.98))
    body = header("Same value, wildly different prices") + f"<div class='chart'>{svg_of(fig)}</div>" + \
        "<div class='src'>Qwen3-14B · 96 outcomes, four for each of 24 values</div>"
    return page(body)


# ---------------------------------------------------------------- F2: the vibe check
def f2_predictors():
    V = load(P13 / "verdicts13.json")
    E = load(P13 / "external_ratings.json")
    D = load(P13 / "features13.json")
    old = [d for d in D if d["set"] == "old"]
    r_rating_old = float(np.corrcoef([d["Rbar"] for d in old], [d["u"] for d in old])[0, 1])
    o = V["P4"]["other_predictors_r"]
    return [("its own “how good?” rating", r_rating_old, o["how_good_rating"]),
            ("nine named features", E["claude"]["P1_r"], E["claude"]["P4_r"]),
            ("its internal state", V["P4"]["ridge_heldout_r_on_old"], o["states_ridge_layer20"])]


def fig_f2():
    D = load(P13 / "features13.json")
    sets = {k: [d for d in D if d["set"] == k] for k in ("old", "new")}
    r_set = {k: float(np.corrcoef([d["Rbar"] for d in S], [d["u"] for d in S])[0, 1]) for k, S in sets.items()}
    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.07, right=0.985, top=0.9, bottom=0.14, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    for k, col in (("old", BLUE), ("new", ORANGE)):
        S = sets[k]
        ax.scatter([d["Rbar"] for d in S], [d["u"] for d in S], s=36, color=col, alpha=0.9, edgecolors="white",
                   linewidths=0.6, zorder=3)
    ax.text(4.1, 6.9, f"96 outcomes, r = {r_set['old']:.2f}", color=BLUE, fontsize=12, fontweight="semibold", va="top")
    ax.text(4.1, 5.6, f"24 new ones, r = {r_set['new']:.2f}", color=ORANGE, fontsize=12, fontweight="semibold", va="top")
    T = {d["text"]: d for d in D}
    for t, lab, dx, dy, ha in [("each spend a week travelling somewhere they have never been", "a week travelling", 14, -2,
                                "left"),
                               ("each get to decide where to go for lunch", "deciding on lunch", -14, 16, "right")]:
        d = T[t]
        ax.scatter([d["Rbar"]], [d["u"]], s=110, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        ax.annotate(lab, (d["Rbar"], d["u"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=11.5, color=INK, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=2, shrinkB=6))
    ax.set_xlim(4.0, 9.25)
    ax.set_ylim(-10, 7.5)
    ax.set_xlabel("its own “how good would it be?” rating")
    ax.set_ylabel("price in forced choices")
    ax.grid(color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax = fig.add_subplot(gs[1])
    pred = f2_predictors()
    yy = np.arange(len(pred))[::-1] * 1.0
    for yi, (lab, r_old, r_new) in zip(yy, pred):
        ax.text(0.5, yi + 0.26, lab, ha="left", va="bottom", fontsize=12, color=INK)
        ax.plot([r_new, r_old], [yi, yi], color=RULE, lw=3, zorder=1)
        ax.scatter([r_old], [yi], s=80, color=BLUE, edgecolors="white", linewidths=1.2, zorder=3)
        ax.scatter([r_new], [yi], s=80, color=ORANGE, edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(r_old + 0.012, yi, f"{r_old:.2f}", va="center", ha="left", fontsize=11.5, color=INK2)
        ax.text(r_new - 0.012, yi, f"{r_new:.2f}", va="center", ha="right", fontsize=11.5, color=INK2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 0.92)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(-0.5, len(pred) - 0.25)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("how well it predicts the price (r)")
    ax.set_title("Nothing does clearly better", fontsize=13, pad=10)
    body = header("The price tracks a vibe check") + f"<div class='chart'>{svg_of(fig)}</div>" + \
        "<div class='src'>Qwen3-14B</div>"
    return page(body)


# ---------------------------------------------------------------- F3: one word
F3_ROWS = ["are each saved from being killed in a fire", "each feel a brief sense of awe while looking at the stars",
           "are each relieved of a painful condition for a week", "each help a child tie their shoes"]


def chip_color(c, cap=12.0):
    """Diverging blue (raises) / red (lowers) around a light grey; beyond twice the cap, the pole's darker step."""
    def mix(a, b, t):
        a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))
    t = min(abs(c) / cap, 1.0)
    if abs(c) > 2 * cap:
        return ("#104281" if c > 0 else "#9c2f27"), True
    return mix(NEUTRAL, BLUE if c > 0 else RED, t ** 0.85), t > 0.62


def fig_f3():
    W = {s["text"]: s for s in load(P14 / "words14.json")}
    rows = []
    for text in F3_ROWS:
        s = W[text]
        chips = []
        for w, c in zip(s["words"], s["contribution"]):
            col, light = chip_color(c)
            num = f"{c:+.0f}".replace("-", "−") if abs(c) >= 3 else ""
            chips.append(f"<span class='w'><span class='chip' style='background:{col};color:{'#fff' if light else INK}'>"
                         f"{esc(w)}</span><span class='n'>{num}</span></span>")
        rows.append(f"<div class='row'>{''.join(chips)}</div>")
    stops = ", ".join(f"{chip_color(c)[0]} {(c + 12) / 24 * 100:.0f}%" for c in np.linspace(-12, 12, 9))
    legend = (f"<div class='legend'><span>less likely</span><span class='ramp' style='background:linear-gradient(90deg,"
              f"{stops})'></span><span>more likely to be chosen</span></div>")
    css = """
    .rows { margin-top:30px; display:flex; flex-direction:column; gap:20px; }
    .row { display:flex; flex-wrap:wrap; gap:4px 5px; }
    .w { display:inline-flex; flex-direction:column; align-items:center; }
    .chip { font-size:23px; padding:3px 7px 4px; border-radius:3px; line-height:1.25; }
    .n { font-size:13.5px; font-weight:500; color:var(--ink2); height:18px; margin-top:3px; font-variant-numeric:tabular-nums; }
    .legend { display:flex; align-items:center; gap:10px; margin-top:26px; font-size:14px; color:var(--ink2); }
    .ramp { display:inline-block; width:200px; height:10px; border-radius:2px; }
    """
    body = header("One word can flip the choice", "What each word adds to the chance that Qwen3-14B picks the outcome") + \
        f"<div class='rows'>{''.join(rows)}</div>{legend}"
    return page(body, extra_css=css)


# ---------------------------------------------------------------- F4: a cycle
F4_FILM = "each get to choose which film to watch tonight"
F4_YEAR = "each have their life extended by a year"
F4_JOB = "each turn the activity they love into their full-time job"
F4_PAY = "each have a decade of unequal pay made up to them in full"


def cycle_sets(L, gap=2.0):
    """Sets of two outcomes and two alternatives where each alternative clearly beats one outcome and loses to the
    other (every margin above `gap`): a cycle when the two alternatives split the pair in opposite directions."""
    win, lose = L > gap, L < -gap
    cyc = could = 0
    for k in range(L.shape[1]):
        for l in range(k + 1, L.shape[1]):
            up_k = win[:, k][:, None] & lose[:, k][None, :]      # outcome i beats k, outcome j loses to k
            up_l = win[:, l][:, None] & lose[:, l][None, :]
            same, opp = int((up_k & up_l).sum()), int((up_k & up_l.T).sum())
            cyc, could = cyc + opp, could + same + opp
    return cyc, could


def f4_data():
    Z = np.load(P16 / "effects16.npz")
    L = Z["logits_intact"]
    T = [s["text"] for s in load(P10 / "sentences10.json")["sentences"]]
    price = {d["text"]: d["u"] for d in load(P10 / "results/real/prices10.json")}
    assert np.allclose(Z["u"], [price[t] for t in T])     # pilot 16's rows are pilot 10's sentences in this order
    B = load(P16 / "verdicts16.json")["B"]
    f, y, j, p = T.index(F4_FILM), T.index(F4_YEAR), B.index(F4_JOB), B.index(F4_PAY)
    e = {"film>job": sig(L[f, j]), "job>year": sig(-L[y, j]), "year>pay": sig(L[y, p]), "pay>film": sig(-L[f, p])}
    assert min(e.values()) > 0.5
    V = {}
    for m, sub in [("Qwen3-14B", "real2/qwen14"), ("Qwen3-32B", "real2/qwen32"), ("Llama 3.1 8B", "real2_llama/llama8")]:
        V[m] = load(P17 / sub / "verdicts17.json")
        Lm = np.load(P17 / sub / "effects17.npz")["logits_intact"]
        V[m]["cycles"], V[m]["could"] = cycle_sets(Lm)
        assert V[m]["cycles"] == V[m]["cycles_margin2"]
        V[m]["scale_gap"] = 2.0 * Lm.std() / np.load(P17 / "real2/qwen14/effects17.npz")["logits_intact"].std()
        V[m]["cycles_scaled"], V[m]["could_scaled"] = cycle_sets(Lm, V[m]["scale_gap"])
    return e, V


def fig_f4():
    e, _ = f4_data()
    W_, H_ = 1096, 360
    labels = {"film": ("Choose tonight's film", 0, 0, "left"), "job": ("Make what they love their job", W_, 0, "right"),
              "year": ("One more year of life", W_, H_ - 36, "right"), "pay": ("A decade of back pay", 0, H_ - 36, "left")}
    nodes = "".join(f"<div class='node {al}' style='{'left' if al == 'left' else 'right'}:{0}px;top:{t}px'>{esc(tx)}</div>"
                    for tx, l, t, al in labels.values())
    arrows = [((300, 18), (760, 18), (530, -16), "c", e["film>job"]),
              ((990, 52), (990, H_ - 52), (1006, H_ / 2), "r", e["job>year"]),
              ((800, H_ - 18), (300, H_ - 18), (550, H_ + 16), "c", e["year>pay"]),
              ((110, H_ - 52), (110, 52), (94, H_ / 2), "l", e["pay>film"])]
    svg = [f"<svg width='{W_}' height='{H_}' style='position:absolute;left:0;top:0;overflow:visible' "
           "xmlns='http://www.w3.org/2000/svg'><defs><marker id='ah' viewBox='0 0 10 10' refX='8.5' refY='5' "
           f"markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{BLUE}'/>"
           "</marker></defs>"]
    nums = []
    for p0, p1, lp, side, pr in arrows:
        svg.append(f"<line x1='{p0[0]}' y1='{p0[1]}' x2='{p1[0]}' y2='{p1[1]}' stroke='{BLUE}' stroke-width='3' "
                   f"marker-end='url(#ah)'/>")
        nums.append(f"<div class='pc {side}' style='left:{lp[0]}px;top:{lp[1]}px'>{pct(pr)}</div>")
    svg.append("</svg>")
    css = """
    .diagram { position:relative; width:1096px; height:360px; margin:44px 0 28px; }
    .node { position:absolute; font-size:23px; font-weight:600; line-height:36px; white-space:nowrap; }
    .pc { position:absolute; font-size:20px; font-weight:600; color:var(--ink); white-space:nowrap;
          font-variant-numeric:tabular-nums; }
    .pc.c { transform:translate(-50%,-50%); }
    .pc.r { transform:translate(0,-50%); }
    .pc.l { transform:translate(-100%,-50%); }
    """
    body = header("Qwen's preferences can go in circles", "Each arrow points from the option Qwen3-14B picks") + \
        f"<div class='diagram'>{''.join(svg)}{nodes}{''.join(nums)}</div>"
    return page(body, extra_css=css)


# ---------------------------------------------------------------- F5: the comparator
F5_MODELS = [("Llama 3.1 8B", "real2_llama/llama8"), ("Qwen3-14B", "real2/qwen14"), ("Qwen3-32B", "real2/qwen32")]


def fig_f5():
    Vs = {name: load(P17 / sub / "verdicts17.json") for name, sub in F5_MODELS}
    tasks = [("answer_position_sentences", "outcomes", BLUE), ("quantity", "apples", ORANGE), ("weight", "weight", AQUA)]
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.3), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.19, wspace=0.08)
    for ax, (name, _) in zip(axs, F5_MODELS):
        V = Vs[name]
        nb = V["blocks"]
        for key, lab, col in tasks:
            d = V[key] if key.startswith("answer") else V["generic"][key]["answer_position"]
            L = sorted(d, key=lambda k: int(k.split("@")[1]))
            ax.plot([int(k.split("@")[1]) / nb * 100 for k in L], [d[k] for k in L], "-", color=col, lw=2.2,
                    label=lab, zorder=3, solid_capstyle="round")
        ax.set_xlim(35, 95)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(name, fontsize=13)
        ax.grid(color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[1].set_xlabel("depth through the model (%)")
    axs[0].set_ylabel("decision at the answer")
    axs[1].legend(loc="upper left", handlelength=1.3, fontsize=11.5, borderaxespad=0.3)
    top = svg_of(fig)
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 2.5), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.96, bottom=0.27, wspace=0.08)
    labs = ["outcomes", "apples", "weight", "random parts"]
    cols = [BLUE, ORANGE, AQUA, AXIS]
    yy = np.arange(4)[::-1]
    for ax, (name, _) in zip(axs, F5_MODELS):
        V = Vs[name]
        rnd = max(V["Q2b"]["random"] + V["Q3b"]["random_apples"] + V["Q3b"]["random_weight"])
        vals = [V["Q2b"]["transfer"], V["Q3b"]["apples"], V["Q3b"]["weight"], max(rnd, 0)]
        ax.barh(yy, vals, color=cols, height=0.6)
        for yi, v in zip(yy, vals):
            ax.text(v + 0.025, yi, f"{v:.2f}", va="center", ha="left", fontsize=11.5, color=INK2)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(yy, labs)
    axs[1].set_xlabel("share carried by the same 20 internal parts")
    bottom = svg_of(fig)
    body = header("Outcomes are compared like piles of apples") + \
        f"<div class='chart'>{top}</div><div class='chart' style='margin-top:22px'>{bottom}</div>"
    return page(body)


# ---------------------------------------------------------------- F6: when the price is finished
def f6_rows():
    rows = []
    for name, sub in F5_MODELS:
        pb = load(P17 / sub / "verdicts17.json")["price_before_B"]
        r = []
        for pos in ("A_nl", "B_nl"):                     # layer chosen on one half of the values, r on the other
            ks = [k for k in pb if k.startswith(pos)]
            r.append(pb[max(ks, key=lambda k: pb[k]["halfA"])]["halfB"])
        rows.append((name, *r))
    return rows


def fig_f6():
    rows = f6_rows()
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.25)
    yy = np.arange(len(rows))[::-1]
    for yi, (name, ra, rb) in zip(yy, rows):
        ax.plot([ra, rb], [yi, yi], color=RULE, lw=3, zorder=1)
        ax.scatter([ra], [yi], s=120, color=LIGHT_BLUE, edgecolors="white", linewidths=1.5, zorder=3)
        ax.scatter([rb], [yi], s=120, color=BLUE, edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(ra - 0.02, yi, f"{ra:.2f}", ha="right", va="center", fontsize=12, color=INK2)
        ax.text(rb + 0.02, yi, f"{rb:.2f}", ha="left", va="center", fontsize=12, color=INK2)
    for x, lab, col in ((rows[0][1], "after option A", LIGHT_BLUE), (rows[0][2], "after option B", BLUE)):
        ax.scatter([x - 0.075], [len(rows) - 0.42], s=60, color=col, edgecolors="white", linewidths=1, clip_on=False)
        ax.text(x - 0.055, len(rows) - 0.42, lab, ha="left", va="center", fontsize=12, color=INK2)
    ax.set_yticks(yy, [r[0] for r in rows])
    for t in ax.get_yticklabels():
        t.set_color(INK)
        t.set_fontsize(12.5)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("how well the internal price tracks one-word edits (r)")
    body = header("The price is finished only after both options") + f"<div class='chart'>{svg_of(fig)}</div>"
    return page(body)


# ---------------------------------------------------------------- hero
HERO_A = "each feel a brief sense of awe while looking at the stars"
HERO_B = "each turn the activity they love into their full-time job"
HERO_WORD = "brief"


def hero_data():
    """The minimal pair in the hero: Qwen3-14B's answer to one menu, with and without one word (pilot 16), and the
    same deletion in the other two models (pilot 17)."""
    S = [s["text"] for s in load(P10 / "sentences10.json")["sentences"]]
    i = S.index(HERO_A)
    Z = np.load(P16 / "effects16.npz")
    r = [r for r, (p, w) in enumerate(Z["var_key"]) if p == i and HERO_A.split()[w] == HERO_WORD]
    assert len(r) == 1
    k = load(P16 / "verdicts16.json")["B"].index(HERO_B)
    p_with, p_without = sig(Z["logits_intact"][i, k]), sig(Z["logits_deleted"][r[0], k])
    assert p_with < 0.5 < p_without                       # with the word Qwen picks B; without it, A
    other = {}
    for name, sub in F5_MODELS:
        V17 = load(P17 / sub / "verdicts17.json")
        Z17 = np.load(P17 / sub / "effects17.npz")
        k17 = V17["B"].index(HERO_B)
        other[name] = (sig(Z17["logits_intact"][i, k17]), sig(Z17["logits_deleted"][r[0], k17]))
    return p_with, p_without, other


def hero():
    p_with, p_without, other = hero_data()
    assert other["Qwen3-32B"][0] < 0.5 < other["Qwen3-32B"][1]      # the 32B model switches too
    assert other["Llama 3.1 8B"][0] > 0.5                              # Llama already picks the stars
    words = HERO_A.split()[1:]                       # the column header already says "8 people each"
    j = words.index(HERO_WORD)
    a_with = " ".join(words[:j]) + f" <b class='hl'>{HERO_WORD}</b> " + " ".join(words[j + 1:])
    a_without = " ".join(words[:j]) + f" <s>{HERO_WORD}</s> " + " ".join(words[j + 1:])

    def lane(a_html, b_html, w_a, w_b, sign, winner, p_win):
        col = "a" if winner == "A" else "b"
        return f"""
        <div class='lane'>
          <div class='opts'><div class='opt'><span class='la'>A</span><span>{a_html}</span></div>
            <div class='opt'><span class='lb'>B</span><span>{b_html}</span></div></div>
          <div class='bars'><div class='bar a' style='width:{w_a}%'></div><div class='bar b' style='width:{w_b}%'></div></div>
          <div class='cmp'><span class='la'>A</span><span class='sign'>{sign}</span><span class='lb'>B</span></div>
          <div class='ans'><span class='l{col}'>{winner}</span><span class='p'>{pct(p_win)}</span></div>
        </div>"""

    css = """
    #fig.hero { width:1600px; height:900px; padding:100px 72px 0; position:relative; overflow:hidden; }
    .hero h1 { font-size:82px; line-height:1.02; letter-spacing:-.012em; white-space:nowrap; }
    .hero .sub { font-size:31px; margin-top:18px; }
    .grid, .lane { display:grid; grid-template-columns:700px 290px 180px 190px; column-gap:32px; }
    .heads { margin-top:80px; padding-bottom:14px; border-bottom:1px solid var(--rule); font-size:20px;
             font-weight:600; color:var(--muted); }
    .lane { align-items:center; padding:40px 0; }
    .lane + .lane { border-top:1px solid var(--rule); }
    .opts { display:flex; flex-direction:column; gap:12px; }
    .opt { display:flex; gap:18px; font-size:28px; line-height:1.3; }
    .la, .lb { font-weight:700; } .la { color:var(--blue); } .lb { color:var(--orange); }
    .opt .la, .opt .lb { width:22px; flex:none; }
    .hl { font-weight:700; }
    s { color:var(--red); text-decoration-thickness:3px; font-weight:600; }
    .same { color:var(--muted); }
    .bars { display:flex; flex-direction:column; gap:14px; }
    .bar { height:26px; border-radius:2px; } .bar.a { background:var(--blue); } .bar.b { background:var(--orange); }
    .cmp { display:flex; align-items:center; gap:18px; font-size:50px; }
    .cmp .sign { font-weight:500; color:var(--ink); }
    .ans { display:flex; align-items:baseline; gap:18px; }
    .ans .la, .ans .lb { font-size:74px; line-height:1; }
    .ans .p { font-size:31px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; }
    .foot { position:absolute; left:72px; bottom:44px; font-size:18px; color:var(--muted); }
    """
    body = f"""
    <h1>A vibe check, then “which is bigger?”</h1>
    <div class='sub'>Each option gets a gut-feel size. The bigger one wins.</div>
    <div class='grid heads'><div>8 people each …</div><div>vibe check</div><div>which is bigger?</div><div>answer</div></div>
    {lane(a_with, esc(HERO_B.split(' ', 1)[1]), 52, 66, "&lt;", "B", 1 - p_with)}
    {lane(a_without, "<span class='same'>(the same)</span>", 84, 66, "&gt;", "A", p_without)}
    <div class='foot'>Qwen3-14B. Bars are illustrative.</div>
    """
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + css + "</style></head><body>"
            "<div id='fig' class='hero'>" + body + "</div></body></html>")


# ---------------------------------------------------------------- captions (the details kept out of the images)
def captions():
    p_with, p_without, other = hero_data()
    _, _, mean_range, med_span, max_span = f1_data()
    e, V = f4_data()
    E = load(P13 / "external_ratings.json")
    V13 = load(P13 / "verdicts13.json")
    W = {s["text"]: s for s in load(P14 / "words14.json")}
    allc = np.array([abs(c) for s in W.values() for c in s["contribution"]])
    rnd = {n: max(load(P17 / s / "verdicts17.json")["Q2b"]["random"] + load(P17 / s / "verdicts17.json")["Q3b"]["random_apples"]
                  + load(P17 / s / "verdicts17.json")["Q3b"]["random_weight"]) for n, s in F5_MODELS}
    q5 = load(P17 / "real2/qwen14/verdicts17.json")["pilot5_pieces"]["overlap_with_decision_writers"]
    cs = {n: (load(P17 / s / "verdicts17.json")["Q1"]["common_share_words"], load(P17 / s / "verdicts17.json")["Q1"]["common_share_outcomes"])
          for n, s in F5_MODELS}
    lo = min(min(v) for v in cs.values())
    hi = max(max(v) for v in cs.values())
    q, l8 = V["Qwen3-32B"], V["Llama 3.1 8B"]
    return f"""# Captions for the post figures

Generated by `make_post_figures.py` from the result files. These hold the details kept out of the images; use or
adapt them as figure captions.

## hero_x.png
One real menu, asked twice. The prompt also says: "{CONTEXT}" and asks "{QUESTION}". With "brief", Qwen3-14B answers
B (the job) with probability {1 - p_with:.2f}; with the word deleted it answers A (the stars) with probability
{p_without:.4f}. Qwen3-32B switches the same way ({other['Qwen3-32B'][0]:.2f} → {other['Qwen3-32B'][1]:.2f} for A);
Llama 3.1 8B picks the stars either way ({other['Llama 3.1 8B'][0]:.4f}). The bars are illustrative. "Which is bigger?":
in all three models, the 20 internal parts that write the decision also carry "which pile has more apples?" and "which
is heavier?" (figure 5).

## f1_same_value.png
Prices of 96 outcomes (four for each of 24 values) fitted to 45,600 forced choices by Qwen3-14B, with
P(A) = σ(u_A − u_B + b·log(n_A/n_B) + c). Four of the 24 values are shown, each with its cheapest and dearest outcome
(after "8 people each"). The averages of the 24 values span {mean_range:.1f} price units; the four outcomes of one value
span {med_span:.1f} at the median and up to {max_span:.1f}. Outcome size explains 63% of the spread, the value 20%. Each
value came in a small, two medium and a large version, so part of the spread is by design, but in 7 of the 24 values a
bigger version is priced below a smaller one (here Respect and Justice).

## f2_vibe_check.png
Left: Qwen3-14B's own answer to "how good would it be if …?" (expected digit 0-9, averaged over three wordings)
against the price, for the 96 outcomes and 24 new ones priced in 13,824 new choices. Right: how well each predictor
tracks the price, on the 96 (held out by value, except the rating, a plain correlation) and on the 24 new outcomes
(from fits on the 96). Nine named features: rated blind by Claude (with Qwen's own answers: {V13['P1']['r']:.2f} and
{V13['P4']['r']:.2f}). Internal state: a ridge readout at layer 20. A week travelling somewhere they have never been
and deciding where to go for lunch are rated about the same and priced far apart.

## f3_one_word.png
Each word shaded by what it adds to Qwen3-14B's log-odds of choosing the outcome (the drop when it is deleted),
averaged over 8 alternatives; numbers shown for effects of 3 or more. These are log-odds of the choice, about 2.4
times the price units of figure 1. Over all {len(allc)} single-word deletions the median moves the choice
{np.median(allc):.1f}; {np.mean(allc > 5) * 100:.0f}% move it more than 5. Deleting "saved" flips all 8 menus.

## f4_cycle.png
Four real menus put to Qwen3-14B; the percentages are the probability of its answer. The outcomes, after "8 people
each": "get to choose which film to watch tonight", "turn the activity they love into their full-time job", "have
their life extended by a year", "have a decade of unequal pay made up to them in full". No fixed ranking allows a
cycle, however the choices are squeezed or stretched. Cycles are rare: among sets of two outcomes and two
alternatives where each alternative clearly beats one outcome and loses to the other (margins above 2 log-odds),
{V['Qwen3-14B']['cycles']} of {V['Qwen3-14B']['could']:,} cycle in Qwen3-14B and {q['cycles']} of {q['could']:,} in
Qwen3-32B. Llama 3.1 8B has {l8['cycles']} of {l8['could']}: its choices are much less extreme, and with the margin
scaled to its spread it has {l8['cycles_scaled']} of {l8['could_scaled']}. Order flips are common: {V['Qwen3-14B']['order_flips']:.0%},
{q['order_flips']:.0%} and {l8['order_flips']:.0%} of clearly separated pairs swap order between alternatives.

## f5_apples.png
Top: swapping the model's internal state at the answer between two prompts; the share of the change in the answer
that comes along, by depth, for outcome menus, "Which of these two piles has more apples?" and "Which of these two
is heavier?". Bottom: swapping only 20 internal parts, the attention heads and MLPs that write the decision's
direction, ranked on half of the outcome values: held-out outcomes, apples and weight, which played no part in
choosing them, and 20 random parts from the same blocks (at most {max(rnd.values()):.2f}). In Qwen3-14B the 20 include
{q5} of the 20 comparator parts found earlier with another method.

## f6_price_timing.png
A ridge readout of the price, trained on the 96 whole outcomes and applied to their 994 one-word deletions: its r with
the choice's own word effects, at the line break after option A (before B is read) and after option B. Layer chosen
on half of the values, r on the other half. Of the variation in the choice's prices, {lo:.0%}-{hi:.0%} is common to all 9
alternatives.
"""


# ---------------------------------------------------------------- render
FIGS = [("hero_x", hero, 1600, 900), ("f1_same_value", fig_f1, 1200, None), ("f2_vibe_check", fig_f2, 1200, None),
        ("f3_one_word", fig_f3, 1200, None), ("f4_cycle", fig_f4, 1200, None), ("f5_apples", fig_f5, 1200, None),
        ("f6_price_timing", fig_f6, 1200, None)]


def main(only=None):
    from playwright.sync_api import sync_playwright
    HTML_DIR.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for name, fn, w, h in FIGS:
            if only and name not in only:
                continue
            p = HTML_DIR / f"{name}.html"
            p.write_text(fn())
            pg = br.new_page(viewport={"width": w, "height": h or 1000}, device_scale_factor=2)
            pg.goto(p.as_uri())
            pg.evaluate("document.fonts.ready")
            pg.wait_for_timeout(250)
            pg.locator("#fig").screenshot(path=str(HERE / f"{name}.png"))
            pg.close()
            print("wrote", name)
        br.close()
    if not only:
        (HERE / "CAPTIONS.md").write_text(captions())
        print("wrote CAPTIONS.md")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
