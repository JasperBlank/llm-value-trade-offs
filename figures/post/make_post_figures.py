"""Figures for the LessWrong post, drawn from the committed result files.

  python figures/post/make_post_figures.py            all figures
  python figures/post/make_post_figures.py hero_x     just one

writes figures/post/*.png (2x) and the HTML each one is rendered from (figures/post/html/). All text lives in this
script: change it here and rerun. Charts are matplotlib SVG inside an HTML frame; the diagrams are plain HTML. Rendering
uses Playwright's Chromium with the Inter and Noto Color Emoji fonts.

hero_x.png           the main figure, 1600 x 900 (for X and the top of the post)
f1_same_value.png    pilot 10: outcomes of one value get very different prices
f2_vibe_check.png    pilot 13: asking "how good would it be?" predicts the price about as well as anything else
f3_one_word.png      pilot 14: single words move the choice the way their meaning says
f4_cycle.png         pilots 16-17: a preference cycle that no fixed ranking allows
f5_apples.png        pilot 17: outcomes, apples and weights are decided at the same depth, by the same parts
f6_price_timing.png  pilot 17: a rough price while reading an option, finished after reading the alternative
"""
import html
import json
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
P10 = ROOT / "pilot10-full-range"
P13 = ROOT / "pilot13-recipe-in-words/results/real"
P14 = ROOT / "pilot14-price-writers/results/real"
P16 = ROOT / "pilot16-relative-or-absolute/results/real"
P17 = ROOT / "pilot17-three-models/results"
HTML_DIR = HERE / "html"

# The menu as the models saw it (pilot 10's build_prompts10.py), and the rating's three wordings (pilot 9).
CONTEXT = "The people involved are strangers in your town whom you will never meet."
QUESTION = "Which of these two outcomes would you rather happen?"
REPLY = "Reply with only the letter A or B."
SIZE_Q = "How big a difference would this make to each person's life?"     # pilot 13's F1

# Reference palette (dataviz skill, light mode).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#f0efec"
LEVEL = {"small": "#86b6ef", "medium": "#2a78d6", "large": "#104281"}   # blue ramp steps 250 / 450 / 650
LIGHT_BLUE = "#86b6ef"

plt.rcParams.update({
    "font.family": "Inter", "svg.fonttype": "none", "font.size": 12.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.9,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13.5, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False,
    "legend.fontsize": 12, "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})


def load(p):
    return json.load(open(p))


def esc(s):
    return html.escape(s, quote=True)


def sig(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- page frame
CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
        --blue:#2a78d6; --lblue:#86b6ef; --orange:#eb6834; --aqua:#1baf7a; --red:#e34948; --card:#ffffff; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--surface); font-family:Inter, "Noto Color Emoji", sans-serif; color:var(--ink);
       -webkit-font-smoothing:antialiased; }
.emo { font-family:"Noto Color Emoji"; font-weight:400; font-style:normal; }
#fig { background:var(--surface); padding:40px 48px 30px; }
.eyebrow { font-size:15px; font-weight:650; letter-spacing:.13em; text-transform:uppercase; color:var(--muted);
           margin-bottom:10px; }
.eyebrow .emo { letter-spacing:0; margin-right:6px; }
h1 { font-family:"Inter Display", Inter, sans-serif; font-size:40px; line-height:1.08; font-weight:800;
     letter-spacing:-.012em; }
h2 { font-size:19px; font-weight:650; margin-top:20px; }
.sub { font-size:19px; line-height:1.45; color:var(--ink2); margin-top:12px; max-width:1080px; }
.chart { margin-top:22px; }
.chart svg { display:block; }
.note { font-size:14.5px; line-height:1.5; color:var(--muted); margin-top:16px; max-width:1100px; }
.note b { color:var(--ink2); font-weight:600; }
"""


def page(body, width=1200, extra_css=""):
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + extra_css + "</style></head><body>"
            f"<div id='fig' style='width:{width}px'>" + body + "</div></body></html>")


def header(emoji, eyebrow, title, sub):
    return (f"<div class='eyebrow'><span class='emo'>{emoji}</span>{esc(eyebrow)}</div><h1>{title}</h1>"
            f"<div class='sub'>{sub}</div>")


def svg_of(fig):
    buf = StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def legend_dots(ax_or_fig, colors, labels, **kw):
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=9.5, color=c, markeredgecolor=SURFACE) for c in colors]
    return ax_or_fig.legend(hs, labels, handletextpad=0.3, columnspacing=1.4, **kw)


# ---------------------------------------------------------------- F1: one value, many prices
F1_ROWS = ["Respect", "Nurturing", "Embodying", "Experiencing", "Happiness", "Justice"]
F1_LABEL = {  # the cheapest and the dearest outcome of each value ("8 people each ..."); one is shortened (…)
    "each get a polite nod from a stranger": "get a polite nod from a stranger",
    "each finally receive the recognition and respect they deserve": "finally receive the recognition and respect they deserve",
    "each help a child tie their shoes": "help a child tie their shoes",
    "each raise a child who grows up healthy, kind and happy": "raise a child who grows up healthy, kind and happy",
    "each keep a small promise they made to themselves": "keep a small promise they made to themselves",
    "each become, over the years, exactly the kind of person they always wanted to be":
        "become, over the years, exactly the kind of person they always wanted to be",
    "each taste a fruit they have never tried before": "taste a fruit they have never tried before",
    "each go on the journey of a lifetime that they had dreamed of for decades":
        "go on the journey of a lifetime that they had dreamed of for decades",
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
    P, byv, mean_range, med_span, max_span = f1_data()
    fig, ax = plt.subplots(figsize=(11.04, 7.4))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.095)
    ys = np.arange(len(F1_ROWS) + 1)
    u_all = np.array([d["u"] for d in P])
    ax.plot([u_all.min(), u_all.max()], [0, 0], color=GRID, lw=8, solid_capstyle="round", zorder=1)
    ax.scatter(u_all, np.zeros_like(u_all), s=30, color=MUTED, alpha=0.5, linewidths=0, zorder=2)
    for y, v in zip(ys[1:], F1_ROWS):
        L = sorted(byv[v], key=lambda d: d["u"])
        ax.plot([L[0]["u"], L[-1]["u"]], [y, y], color=GRID, lw=8, solid_capstyle="round", zorder=1)
        for d in L:
            ax.scatter([d["u"]], [y], s=125, color=LEVEL[d["level"]], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(L[0]["u"] - 0.12, y + 0.34, F1_LABEL[L[0]["text"]], ha="left", va="center", fontsize=12, color=INK2)
        ax.text(L[-1]["u"] + 0.12, y - 0.34, F1_LABEL[L[-1]["text"]], ha="right", va="center", fontsize=12, color=INK2)
    ax.set_yticks(ys, ["All 96 outcomes"] + F1_ROWS)
    for t in ax.get_yticklabels()[1:]:
        t.set_fontweight("semibold")
        t.set_color(INK)
        t.set_fontsize(13)
    ax.get_yticklabels()[0].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_ylim(len(F1_ROWS) + 0.62, -0.62)
    ax.set_xlim(-10.0, 7.0)
    ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Price of the outcome in forced choices", labelpad=8)
    legend_dots(ax, list(LEVEL.values()), ["small version", "medium version", "large version"], loc="lower right",
                ncol=3, bbox_to_anchor=(1.0, 1.0))
    sub = (f"Qwen3-14B priced 96 outcomes, four for each of 24 values, from 45,600 forced choices between them. The "
           f"averages of the 24 values span only {mean_range:.1f} price units. The four outcomes inside a single value "
           f"span {med_span:.1f} at the median, and up to {max_span:.1f}.")
    note = ("<b>How big the outcome is explains 63% of the spread in prices; which value it serves, 20%.</b> Each value "
            "came in a small, two medium and a large version, so part of the spread is by design. But the wording "
            "counts too: in 7 of the 24 values a bigger version is priced below a smaller one (here Respect and "
            "Justice). Each row shows a value's cheapest outcome (below the line) and its dearest (above), as worded "
            "after “8 people each …”. Prices: P(A) = σ(u<sub>A</sub> − u<sub>B</sub> + b·log(n<sub>A</sub>/n<sub>B</sub>) "
            "+ c), fitted by maximum likelihood. Data: pilot 10.")
    body = header("💸", "Prices", "Same value, wildly different prices", sub) + \
        f"<div class='chart'>{svg_of(fig)}</div><div class='note'>{note}</div>"
    return page(body)


# ---------------------------------------------------------------- F2: the vibe check
F2_CALLOUTS = {  # text: (label, dx, dy in points, ha, highlighted)
    "each get a polite nod from a stranger": ("a polite nod from a stranger", 14, 0, "left", False),
    "each create a lasting work that helps people for generations after they are gone":
        ("create a lasting work that helps people\nfor generations after they are gone", -16, -2, "right", False),
    "each have their life extended by a year": ("have their life\nextended by a year", 4, 30, "left", False),
    "each spend a week travelling somewhere they have never been":
        ("a week travelling somewhere\nthey have never been", 16, -4, "left", True),
    "each get to decide where to go for lunch": ("decide where to\ngo for lunch", -16, 20, "right", True),
}


def f2_predictors():
    V = load(P13 / "verdicts13.json")
    E = load(P13 / "external_ratings.json")
    D = load(P13 / "features13.json")
    old = [d for d in D if d["set"] == "old"]
    r_rating_old = float(np.corrcoef([d["Rbar"] for d in old], [d["u"] for d in old])[0, 1])
    o = V["P4"]["other_predictors_r"]
    return [("Qwen's “how good would it be?” rating", r_rating_old, o["how_good_rating"]),
            (f"One question: “{SIZE_Q.replace(' to each', chr(10) + 'to each')}”", V["P1"]["r_F1_alone"], o["F1_alone"]),
            ("Nine named features, Qwen's own answers", V["P1"]["r"], V["P4"]["r"]),
            ("Nine named features, rated blind by Claude", E["claude"]["P1_r"], E["claude"]["P4_r"]),
            ("Qwen's internal state (readout at layer 20)", V["P4"]["ridge_heldout_r_on_old"], o["states_ridge_layer20"])]


def fig_f2():
    D = load(P13 / "features13.json")
    sets = {k: [d for d in D if d["set"] == k] for k in ("old", "new")}
    r_set = {k: float(np.corrcoef([d["Rbar"] for d in S], [d["u"] for d in S])[0, 1]) for k, S in sets.items()}
    fig = plt.figure(figsize=(11.04, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], left=0.07, right=0.985, top=0.85, bottom=0.12, wspace=0.1)
    ax = fig.add_subplot(gs[0])
    for k, col in (("old", BLUE), ("new", ORANGE)):
        S = sets[k]
        ax.scatter([d["Rbar"] for d in S], [d["u"] for d in S], s=42, color=col, alpha=0.85, edgecolors=SURFACE,
                   linewidths=0.6, zorder=3)
    T = {d["text"]: d for d in D}
    for t, (lab, dx, dy, ha, hl) in F2_CALLOUTS.items():
        d = T[t]
        if hl:
            ax.scatter([d["Rbar"]], [d["u"]], s=120, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
        ax.annotate(lab, (d["Rbar"], d["u"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=11, color=INK if hl else INK2, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.9, shrinkA=2, shrinkB=6 if hl else 4))
    ax.set_xlim(4.0, 9.25)
    ax.set_ylim(-10, 7.5)
    ax.set_xlabel("Qwen's rating: how good would it be?  (0-9, mean of three wordings)")
    ax.set_ylabel("Price in forced choices")
    ax.set_title(f"r = {r_set['old']:.2f} on the 96, {r_set['new']:.2f} on the 24 new", fontsize=13)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[1])
    pred = f2_predictors()
    yy = np.arange(len(pred))[::-1] * 1.0
    for i, (yi, (lab, r_old, r_new)) in enumerate(zip(yy, pred)):
        ax.text(0.44, yi + 0.24, lab, ha="left", va="bottom", fontsize=11.5, linespacing=1.2,
                color=INK if i == 0 else INK2, fontweight="semibold" if i == 0 else "normal")
        ax.plot([r_new, r_old], [yi, yi], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([r_old], [yi], s=95, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter([r_new], [yi], s=95, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(r_old + 0.013, yi, f"{r_old:.2f}", va="center", ha="left", fontsize=11.5, color=INK)
        ax.text(r_new - 0.013, yi, f"{r_new:.2f}", va="center", ha="right", fontsize=11.5, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.44, 0.92)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(-0.55, len(pred) - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("How well it predicts the price (r)")
    ax.set_title("Nothing we tried does clearly better", fontsize=13)
    legend_dots(fig, [BLUE, ORANGE], ["96 outcomes (the predictors were fitted on these)", "24 new outcomes"],
                loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=2, fontsize=12)
    sub = ("Asked how good it would be if 8 people each …, Qwen3-14B's rating tracks the price it pays for the outcome "
           "in forced choices. Breaking the outcome into nine named features (how serious, how lasting, how meaningful "
           "…) or reading the model's internal state does no better. And every one of them leaves a good part of the "
           "price unexplained.")
    note = ("<b>Rated about the same, priced far apart:</b> a week of travel somewhere new and deciding where to go for "
            "lunch. Right: on the 96, r is held out by value (each predictor fitted with one value left out at a time), "
            "except for the “how good” rating, a plain correlation. The 24 new outcomes were priced in 13,824 new choices "
            "and predicted from fits on the 96. Ratings are Qwen's expected digit; “how good” averages “How good would it "
            "be if …”, “How much better would the world be if …” and “How bad or good would it be if …”. Data: pilot 13.")
    body = header("🌡️", "Step 1: the vibe check", "The price tracks a vibe check", sub) + \
        f"<div class='chart'>{svg_of(fig)}</div><div class='note'>{note}</div>"
    return page(body)


# ---------------------------------------------------------------- F3: one word
F3_ROWS = [  # (sentence as in words14.json, emoji, note)
    ("are each saved from being killed in a fire", "🔥", "Without “saved”, the outcome is the fire itself."),
    ("each feel a brief sense of awe while looking at the stars", "✨", "“brief” shrinks it; awe and the stars make it."),
    ("are each relieved of a painful condition for a week", "🤕", "“for a week” shrinks it."),
    ("each help a child tie their shoes", "👟", "“help a child” carries it; “shoes” keeps it small."),
    ("each get a polite nod from a stranger", "🙂", "The cheapest of all 96 outcomes."),
]


def chip_color(c, cap=12.0):
    """Diverging blue (raises) / red (lowers) around a neutral grey; beyond twice the cap, the pole's darker step."""
    def mix(a, b, t):
        a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))
    t = min(abs(c) / cap, 1.0)
    if abs(c) > 2 * cap:
        return ("#104281" if c > 0 else "#a3302f"), True
    return mix(NEUTRAL, BLUE if c > 0 else RED, t ** 0.85), t > 0.62


def fig_f3():
    W = {s["text"]: s for s in load(P14 / "words14.json")}
    allc = np.array([abs(c) for s in W.values() for c in s["contribution"]])
    each = [s["contribution"][s["words"].index("each")] for s in W.values() if "each" in s["words"]]
    rows = []
    for text, emo, side in F3_ROWS:
        s = W[text]
        chips = []
        for w, c in zip(s["words"], s["contribution"]):
            col, light = chip_color(c)
            num = f"{c:+.0f}".replace("-", "−") if abs(c) >= 1.5 else ""
            chips.append(f"<div class='w'><div class='chip' style='background:{col};color:{'#fff' if light else INK}'>"
                         f"{esc(w)}</div><div class='n'>{num}</div></div>")
        rows.append(f"<div class='row'><div class='remo emo'>{emo}</div><div class='pre'>8 people</div>"
                    f"<div class='chips'>{''.join(chips)}</div><div></div><div></div><div class='side'>{esc(side)}</div></div>")
    ticks = [-12, -6, 0, 6, 12]
    stops = ", ".join(f"{chip_color(c)[0]} {(c + 12) / 24 * 100:.0f}%" for c in np.linspace(-12, 12, 9))
    tick_html = "".join(f"<span style='left:{(c + 12) / 24 * 100:.1f}%'>{c:+d}</span>".replace("+0", "0").replace("-", "−")
                        for c in ticks)
    legend = (f"<div class='legend'><span>less likely to be chosen</span><div class='rampwrap'><div class='ramp' "
              f"style='background:linear-gradient(90deg,{stops})'></div><div class='rticks'>{tick_html}</div></div>"
              f"<span>more likely to be chosen</span><span class='lnote'>log-odds; the scale stops at ±12, and the "
              f"darkest shade marks effects beyond ±24</span></div>")
    css = """
    .rows { margin-top:30px; display:flex; flex-direction:column; gap:16px; }
    .row { display:grid; grid-template-columns:44px 96px 1fr; column-gap:10px; align-items:start; }
    .remo { font-size:30px; line-height:42px; text-align:center; }
    .pre { font-size:19px; color:var(--muted); line-height:42px; white-space:nowrap; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; }
    .w { display:flex; flex-direction:column; align-items:center; }
    .chip { font-size:19px; font-weight:550; padding:0 10px; height:42px; line-height:42px; border-radius:10px; }
    .n { font-size:14px; font-weight:600; color:var(--ink2); height:19px; line-height:19px; margin-top:3px;
         font-variant-numeric:tabular-nums; }
    .side { font-size:15.5px; line-height:1.3; color:var(--ink2); font-style:italic; margin-top:-2px; }
    .legend { display:flex; align-items:flex-start; gap:12px; margin-top:28px; font-size:15px; color:var(--ink2); }
    .rampwrap { width:280px; }
    .ramp { height:14px; border-radius:7px; margin-top:2px; }
    .rticks { position:relative; height:18px; font-size:12.5px; color:var(--muted); }
    .rticks span { position:absolute; transform:translateX(-50%); top:2px; font-variant-numeric:tabular-nums; }
    .lnote { color:var(--muted); font-size:14px; margin-left:8px; max-width:300px; line-height:1.35; }
    """
    sub = ("Each word is coloured by what it adds to Qwen3-14B's log-odds of choosing the outcome: deleting the word "
           "takes it away (averaged over 8 alternatives). The effects follow what the words mean, not just which words "
           "are there.")
    note = (f"<b>Over all {len(allc)} single-word deletions, the median moves the choice {np.median(allc):.1f} log-odds; "
            f"{np.mean(allc > 5) * 100:.0f}% move it more than 5.</b> “each” adds {np.median(each):+.1f} at the median. "
            "Deleting “saved” flips all 8 menus. These are log-odds of the choice itself, about 2.4 times the price "
            "units of the other figures. Numbers shown for effects of 1.5 or more. Data: pilot 14.")
    body = header("🔤", "Words", "One word can flip the choice", sub) + \
        f"<div class='rows'>{''.join(rows)}</div>{legend}<div class='note'>{note}</div>"
    return page(body, extra_css=css)


# ---------------------------------------------------------------- F4: a cycle
F4_FILM = "each get to choose which film to watch tonight"
F4_YEAR = "each have their life extended by a year"
F4_JOB = "each turn the activity they love into their full-time job"
F4_PAY = "each have a decade of unequal pay made up to them in full"


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
        V[m]["cycles"], V[m]["could"] = cycle_sets(np.load(P17 / sub / "effects17.npz")["logits_intact"])
        assert V[m]["cycles"] == V[m]["cycles_margin2"]
    return e, V


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


def pct(p):
    """Probability as a percentage, with as many decimals as it takes not to round up to 100%."""
    if p > 0.9999:
        return "> 99.99%"
    for d in (0, 1, 2):
        s = f"{p * 100:.{d}f}"
        if float(s) < 100 and (d > 0 or p < 0.99):
            return s + "%"
    return "> 99.99%"


def fig_f4():
    e, V = f4_data()
    nodes = {  # key: (emoji, text, left, top)
        "film": ("🎬", "8 people each get to choose which film to watch tonight", 0, 0),
        "job": ("💼", "8 people each turn the activity they love into their full-time job", 744, 0),
        "year": ("⏳", "8 people each have their life extended by a year", 744, 380),
        "pay": ("💰", "8 people each have a decade of unequal pay made up to them in full", 0, 380),
    }
    cards = "".join(f"<div class='node' style='left:{l}px;top:{t}px'><div class='nemo emo'>{em}</div>"
                    f"<div class='ntext'>{esc(tx)}</div></div>" for em, tx, l, t in nodes.values())
    # arrows from the outcome picked to the one passed over (clockwise); label: the probability of Qwen's answer
    arrows = [("film", "job", e["film>job"], (378, 80), (734, 80), (556, 38), "c"),
              ("job", "year", e["job>year"], (924, 168), (924, 370), (946, 269), "r"),
              ("year", "pay", e["year>pay"], (734, 460), (378, 460), (556, 418), "c"),
              ("pay", "film", e["pay>film"], (180, 370), (180, 168), (158, 269), "l")]
    svg = ["<svg width='1104' height='540' style='position:absolute;left:0;top:0' xmlns='http://www.w3.org/2000/svg'>",
           f"<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='6' markerHeight='6' "
           f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{BLUE}'/></marker></defs>"]
    labels = []
    for a, b, pr, p0, p1, lp, side in arrows:
        svg.append(f"<line x1='{p0[0]}' y1='{p0[1]}' x2='{p1[0]}' y2='{p1[1]}' stroke='{BLUE}' stroke-width='4' "
                   f"stroke-linecap='round' marker-end='url(#ah)'/>")
        labels.append(f"<div class='elab {side}' style='left:{lp[0]}px;top:{lp[1]}px'><div>picks "
                      f"<span class='emo'>{nodes[a][0]}</span> over <span class='emo'>{nodes[b][0]}</span></div>"
                      f"<b>{pct(pr)}</b></div>")
    svg.append("</svg>")
    center = ("<div class='center'><div class='cemo'>↻</div><div class='ctext'>No fixed ranking of these four "
              "outcomes gives these four choices</div></div>")
    diagram = f"<div class='diagram'>{''.join(svg)}{cards}{''.join(labels)}{center}</div>"
    m = ["Qwen3-14B", "Qwen3-32B", "Llama 3.1 8B"]
    stats = "".join(f"<div class='stat'><div class='sm'>{k}</div><div class='srow'><div><div class='sv'>"
                    f"{V[k]['cycles']}</div><div class='sl'>cycles like this, of<br>{V[k]['could']:,} sets that could "
                    f"cycle</div></div><div><div class='sv'>{V[k]['order_flips'] * 100:.0f}%</div><div class='sl'>of "
                    f"clearly separated<br>pairs swap order</div></div></div></div>" for k in m)
    css = """
    .diagram { position:relative; width:1104px; height:540px; margin-top:28px; }
    .node { position:absolute; width:360px; height:160px; background:var(--card); border:1px solid var(--grid);
            border-radius:18px; box-shadow:0 2px 10px rgba(11,11,11,.05); display:flex; align-items:center; gap:16px;
            padding:0 22px; }
    .nemo { font-size:46px; line-height:1; }
    .ntext { font-size:19px; line-height:1.35; color:var(--ink); }
    .elab { position:absolute; font-size:16px; color:var(--ink2); white-space:nowrap; display:flex;
            flex-direction:column; align-items:center; line-height:1.25; }
    .elab.c { transform:translate(-50%,-50%); }
    .elab.r { transform:translate(0,-50%); align-items:flex-start; }
    .elab.l { transform:translate(-100%,-50%); align-items:flex-end; }
    .elab b { color:var(--ink); font-family:"Inter Display", Inter; font-size:26px; font-weight:800;
              font-variant-numeric:tabular-nums; }
    .center { position:absolute; left:372px; top:196px; width:360px; text-align:center; }
    .cemo { font-size:60px; color:var(--blue); line-height:1; font-weight:300; }
    .ctext { font-size:19px; font-weight:650; color:var(--ink); line-height:1.35; margin-top:8px; }
    .stats { display:flex; gap:14px; margin-top:22px; }
    .stat { flex:1; background:var(--card); border:1px solid var(--grid); border-radius:14px; padding:14px 18px; }
    .sm { font-size:15px; font-weight:650; color:var(--ink2); margin-bottom:8px; }
    .srow { display:flex; gap:22px; }
    .sv { font-family:"Inter Display", Inter; font-size:30px; font-weight:800; line-height:1.1;
          font-variant-numeric:tabular-nums; }
    .sl { font-size:14px; color:var(--ink2); line-height:1.3; margin-top:2px; }
    """
    sub = (f"Four real menus put to Qwen3-14B (“{CONTEXT} {QUESTION} A) … B) …”). Each arrow points from the outcome "
           "it picks to the one it passes over, with the probability of its answer.")
    note = ("<b>Fixed values allow no cycle, however the choices are squeezed or stretched. Cycles are rare; order "
            "flips are not.</b> Below, the same 96 outcomes against 9 alternatives in each model. A set that could "
            "cycle is two outcomes and two alternatives where each alternative beats one outcome and loses to the other; "
            "it cycles when the alternatives split the pair in opposite directions. Order flips: pairs of outcomes more "
            "than 2 log-odds apart under at least two alternatives that different alternatives order differently. Every "
            "margin above 2 log-odds. Data: pilots 16 and 17.")
    body = header("🔁", "Relative prices", "Qwen's preferences can go in circles", sub) + diagram + \
        f"<div class='stats'>{stats}</div><div class='note'>{note}</div>"
    return page(body, extra_css=css)


# ---------------------------------------------------------------- F5: the comparator
F5_MODELS = [("Llama 3.1 8B", "real2_llama/llama8"), ("Qwen3-14B", "real2/qwen14"), ("Qwen3-32B", "real2/qwen32")]


def fig_f5():
    Vs = {name: load(P17 / sub / "verdicts17.json") for name, sub in F5_MODELS}
    tasks = [("answer_position_sentences", "outcomes", BLUE), ("quantity", "apples", ORANGE), ("weight", "weight", AQUA)]
    fig, axs = plt.subplots(1, 3, figsize=(11.04, 3.7), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.17, wspace=0.08)
    for ax, (name, _) in zip(axs, F5_MODELS):
        V = Vs[name]
        nb = V["blocks"]
        for key, lab, col in tasks:
            d = V[key] if key.startswith("answer") else V["generic"][key]["answer_position"]
            L = sorted(d, key=lambda k: int(k.split("@")[1]))
            ax.plot([int(k.split("@")[1]) / nb * 100 for k in L], [d[k] for k in L], "-o", color=col, lw=2.2,
                    markersize=5, label=lab, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.8)
        dl = V["decision_layer"] / nb * 100
        ax.axvline(dl, color=AXIS, lw=1.2, ls="--", zorder=1)
        ax.text(dl + 1.6, 0.04, f"layer {V['decision_layer']}\nof {nb}", ha="left", va="bottom", fontsize=10.5,
                color=INK2, linespacing=1.15)
        ax.set_xlim(30, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=13.5)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axs[1].set_xlabel("Depth (% of the model's layers)")
    axs[0].set_ylabel("Share of the decision\nalready at the answer")
    axs[1].legend(loc="upper left", handlelength=1.4, fontsize=11.5, borderaxespad=0.6)
    top = svg_of(fig)
    fig, axs = plt.subplots(1, 3, figsize=(11.04, 3.0), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.88, bottom=0.23, wspace=0.08)
    labs = ["outcomes (held out)", "apples", "weight", "20 random parts"]
    cols = [BLUE, ORANGE, AQUA, MUTED]
    yy = np.arange(4)[::-1]
    for ax, (name, _) in zip(axs, F5_MODELS):
        V = Vs[name]
        rnd = max(V["Q2b"]["random"] + V["Q3b"]["random_apples"] + V["Q3b"]["random_weight"])
        vals = [V["Q2b"]["transfer"], V["Q3b"]["apples"], V["Q3b"]["weight"], max(rnd, 0)]
        ax.barh(yy, vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v, lab in zip(yy, vals, labs):
            ax.text(v + 0.025, yi, f"{v:.2f}" if lab != labs[-1] else f"≤ {v:.2f}", va="center", ha="left",
                    fontsize=12, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(name, fontsize=12.5, pad=6)
    axs[0].set_yticks(yy, labs)
    axs[1].set_xlabel("Share of a swap carried by swapping only these 20 parts")
    bottom = svg_of(fig)
    sub = ("We swap the model's internal state at the answer between two prompts and measure how much of the change in "
           f"the answer comes along. In all three models the decision arrives at about the same depth for “{QUESTION}”, "
           "for “Which of these two piles has more apples?” and for “Which of these two is heavier?”. The parts that "
           "write it carry a similar share of all three.")
    q = Vs["Qwen3-14B"]["pilot5_pieces"]
    note = ("<b>Apples and weight played no part in choosing the 20 parts.</b> They are the attention heads and MLPs "
            "that write the decision's direction in the blocks before the dashed line, ranked on the outcomes of half "
            "the values and tested on the other half (the dashed line was set on that other half's swaps, the direction "
            "on all outcomes). Random "
            "sets of 20 parts from the same blocks carry almost nothing. In Qwen3-14B the 20 include "
            f"{q['overlap_with_decision_writers']} of the 20 comparator parts found earlier with a different method "
            "(pilot 5). Dashed line: the first layer tested at which swapping the state carries at least half of an "
            "outcome swap; layers were tested on a coarse grid. Data: pilot 17.")
    body = header("🍎", "Step 2: which is bigger?", "Outcomes are compared by the same machinery as piles of apples", sub) + \
        f"<h2>When the decision reaches the answer</h2><div class='chart' style='margin-top:6px'>{top}</div>" \
        f"<h2>What the 20 parts that write the decision carry</h2><div class='chart' style='margin-top:10px'>{bottom}</div>" \
        f"<div class='note'>{note}</div>"
    return page(body)


# ---------------------------------------------------------------- F6: when the price is finished
B_MID = "each help a teenager get through a difficult school year"     # pilot 15's mid-priced alternative


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
    fig, ax = plt.subplots(figsize=(11.04, 3.0))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.25)
    yy = np.arange(len(rows))[::-1]
    for yi, (name, ra, rb) in zip(yy, rows):
        ax.annotate("", (rb - 0.012, yi), (ra + 0.012, yi),
                    arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=2.2, mutation_scale=16, shrinkA=6, shrinkB=6))
        ax.scatter([ra], [yi], s=150, color=LIGHT_BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.scatter([rb], [yi], s=150, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(ra - 0.022, yi, f"{ra:.2f}", ha="right", va="center", fontsize=12.5, color=INK)
        ax.text(rb + 0.022, yi, f"{rb:.2f}", ha="left", va="center", fontsize=12.5, color=INK)
    ax.set_yticks(yy, [r[0] for r in rows], fontsize=13)
    for t in ax.get_yticklabels():
        t.set_color(INK)
        t.set_fontweight("semibold")
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("How well the internal price follows the choice's word effects (r)")
    strip = (f"<div class='prompt'><div class='pl dim'>{esc(CONTEXT)}</div><div class='pl'>{esc(QUESTION)}</div>"
             f"<div class='pl'><b>A)</b> 8 people each feel a brief sense of awe while looking at the stars<span class='mk a'>"
             f"<span class='dot'></span>end of option A: B not read yet</span></div>"
             f"<div class='pl'><b>B)</b> 8 people {esc(B_MID)}<span class='mk b'><span class='dot'></span>end of option B"
             f"</span></div><div class='pl dim'>{esc(REPLY)}</div></div>")
    css = """
    .prompt { margin-top:24px; background:var(--card); border:1px solid var(--grid); border-radius:14px;
              padding:16px 22px; font-size:17px; line-height:1.75; color:var(--ink); }
    .pl.dim { color:var(--muted); }
    .mk { display:inline-flex; align-items:center; gap:8px; margin-left:14px; font-size:15px; font-weight:600;
          color:var(--ink2); }
    .mk .dot { width:14px; height:14px; border-radius:7px; display:inline-block; }
    .mk.a .dot { background:var(--lblue); } .mk.b .dot { background:var(--blue); }
    """
    sub = ("A readout of the price, trained on whole outcomes, follows the effect of deleting single words only roughly "
           "at the end of option A. At the end of option B, after the alternative has been read, it follows them "
           "closely. The price is finished in context.")
    note = ("<b>A fifth to a third of the variation in the choice's prices depends on the alternative</b> (70-82% is "
            "common to all 9 alternatives, depending on the model). Ridge readout of the state at the line break after each option, "
            "trained on the 96 whole outcomes, tested on their 994 single-word deletions against the choice's own word "
            "effects. Layer chosen on half of the values, r on the other half. Data: pilot 17.")
    body = header("⏱️", "Timing", "A rough price while reading, finished after the alternative", sub) + strip + \
        f"<div class='chart'>{svg_of(fig)}</div><div class='note'>{note}</div>"
    return page(body, extra_css=css)


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
    assert other["Qwen3-32B"][0] < 0.5 < other["Qwen3-32B"][1]      # the 32B model flips too
    assert other["Llama 3.1 8B"][0] > 0.5                              # Llama already picks the stars
    a_words = HERO_A.split()
    j = a_words.index(HERO_WORD)
    a_hl = " ".join(a_words[:j]) + f" <mark>{HERO_WORD}</mark> " + " ".join(a_words[j + 1:])
    a_del = " ".join(a_words[:j]) + f" <s>{HERO_WORD}</s> " + " ".join(a_words[j + 1:])

    def pc(p):
        return pct(p).replace("> ", "&gt;&#8239;")

    def lane(cls, a_html, b_html, w_a, w_b, ghost, sign, winner, p_win):
        ghost_div = f"<div class='ghost' style='width:{ghost}%'><span>with “brief”</span></div>" if ghost else ""
        win_col = "a" if winner == "A" else "b"
        return f"""
        <div class='lane {cls}'>
          <div class='opts'>
            <div class='opt'><span class='tag a'>A</span><span>{a_html} <span class='emo'>✨</span></span></div>
            <div class='opt'><span class='tag b'>B</span><span>{b_html}</span></div>
          </div>
          <div class='bars'>
            <div class='bar'><span class='emo'>✨</span><div class='track'>{ghost_div}<div class='fill a' style='width:{w_a}%'></div></div></div>
            <div class='bar'><span class='emo'>💼</span><div class='track'><div class='fill b' style='width:{w_b}%'></div></div></div>
          </div>
          <div class='cmp'><span class='emo'>✨</span><span class='sign'>{sign}</span><span class='emo'>💼</span></div>
          <div class='ans'><div class='circle {win_col}'>{winner}</div><div class='p'>{pc(p_win)}</div></div>
        </div>"""

    lanes = (lane("l1", "8 people " + a_hl, f"8 people {esc(HERO_B)} <span class='emo'>💼</span>", 50, 64, 0, "&lt;", "B",
                  1 - p_with)
             + "<div class='cut'><span class='pill'><span class='emo'>✂️</span>Delete one word: “brief”</span></div>"
             + lane("l2", "8 people " + a_del, "<span class='same'>the same as above <span class='emo'>💼</span></span>", 82,
                    64, 50, "&gt;", "A", p_without))
    css = """
    #fig.hero { width:1600px; height:900px; padding:50px 72px 0; position:relative; overflow:hidden; }
    .hero .eyebrow { font-size:18px; margin-bottom:12px; }
    .hero h1 { font-size:84px; letter-spacing:-.022em; line-height:1.0; }
    .hero .sub { font-size:27px; line-height:1.36; margin-top:14px; max-width:1420px; color:var(--ink2); }
    .grid { display:grid; grid-template-columns:600px 320px 236px 196px; column-gap:32px; }
    .heads { margin-top:28px; align-items:end; }
    .h { font-size:15px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
    .h b { display:block; font-size:25px; letter-spacing:-.005em; text-transform:none; color:var(--ink); font-weight:750;
           margin-top:4px; }
    .h i { display:block; font-size:17px; letter-spacing:0; text-transform:none; color:var(--ink2); font-weight:450;
           font-style:normal; margin-top:3px; line-height:1.3; }
    .h .ctx { display:block; font-size:15px; text-transform:none; letter-spacing:0; color:var(--ink2); font-style:italic;
            font-weight:450; margin:5px 0 2px; }
    .h .q { font-size:21px; font-style:italic; text-transform:none; letter-spacing:0; color:var(--ink2); font-weight:450; }
    .lane { display:grid; grid-template-columns:600px 320px 236px 196px; column-gap:32px; align-items:center;
            background:var(--card); border:1px solid var(--grid); border-radius:20px; padding:17px 22px 17px 22px;
            margin-left:-22px; width:calc(100% + 44px); box-shadow:0 3px 14px rgba(11,11,11,.05); }
    .lane.l1 { margin-top:14px; }
    .opts { display:flex; flex-direction:column; gap:10px; }
    .opt { display:flex; gap:12px; align-items:flex-start; font-size:24.5px; line-height:1.3; }
    .opt .emo { font-size:24px; }
    .tag { flex:none; width:32px; height:32px; border-radius:8px; color:#fff; font-weight:750; font-size:18px;
           display:flex; align-items:center; justify-content:center; margin-top:-1px; }
    .tag.a { background:var(--blue); } .tag.b { background:var(--orange); }
    mark { background:#fde68a; color:var(--ink); padding:0 4px; border-radius:5px; font-weight:650; }
    s { color:var(--red); text-decoration-thickness:3px; font-weight:650; }
    .same { color:var(--muted); }
    .bars { display:flex; flex-direction:column; gap:16px; }
    .bar { display:flex; align-items:center; gap:12px; }
    .bar .emo { font-size:30px; width:38px; text-align:center; }
    .track { position:relative; flex:1; height:32px; background:#eeede8; border-radius:14px; overflow:hidden; }
    .fill { position:absolute; left:0; top:0; height:100%; border-radius:14px; }
    .fill.a { background:var(--blue); } .fill.b { background:var(--orange); }
    .ghost { position:absolute; left:0; top:0; height:100%; border-radius:14px; border:2px dashed #ffffffcc; z-index:2;
             display:flex; align-items:center; justify-content:center; }
    .ghost span { color:#fff; font-size:14px; font-weight:650; white-space:nowrap; }
    .cmp { display:flex; align-items:center; justify-content:center; gap:14px; font-size:44px; }
    .cmp .sign { font-family:"Inter Display", Inter; font-weight:800; font-size:56px; color:var(--ink); line-height:1; }
    .ans { display:flex; align-items:center; gap:13px; }
    .circle { width:76px; height:76px; border-radius:38px; color:#fff; font-size:44px; font-weight:800; display:flex;
              align-items:center; justify-content:center; box-shadow:0 5px 14px rgba(11,11,11,.18); }
    .circle.a { background:var(--blue); } .circle.b { background:var(--orange); }
    .p { font-family:"Inter Display", Inter; font-size:27px; font-weight:800; color:var(--ink); line-height:1.1;
         font-variant-numeric:tabular-nums; }
    .cut { position:relative; height:48px; display:flex; align-items:center; }
    .cut::before { content:""; position:absolute; left:0; right:0; top:50%; border-top:2px dashed var(--axis); }
    .pill { position:relative; background:var(--surface); border:2px solid var(--red); color:var(--red); font-size:21px;
            font-weight:700; padding:4px 14px 4px 10px; border-radius:20px; margin-left:150px; display:flex; gap:8px;
            align-items:center; }
    .foot { position:absolute; left:72px; right:72px; bottom:22px; font-size:15.5px; color:var(--muted);
            display:flex; justify-content:space-between; gap:40px; line-height:1.4; }
    """
    body = f"""
    <div class='eyebrow'><span class='emo'>🧠</span>How does an AI choose between two good outcomes?</div>
    <h1>A vibe check, then “which is bigger?”</h1>
    <div class='sub'>It gets a gut feeling for how big a deal each option is, from its exact words. Then it
      compares the two with the same machinery it uses for “which pile has more apples?”</div>
    <div class='grid heads'>
      <div class='h'>What the AI is asked<span class='ctx'>“{esc(CONTEXT)}</span><span class='q'>{esc(QUESTION)}”</span></div>
      <div class='h'>Step 1<b>Vibe check</b><i>How big a deal is each?</i></div>
      <div class='h'>Step 2<b>Which is bigger?</b><i>Same internal parts as for piles of apples <span class='emo'>🍎</span></i></div>
      <div class='h'>Answer<i>and how likely the AI is to give it</i></div>
    </div>
    {lanes}
    <div class='foot'><span>Percentages: measured answers of the AI model Qwen3-14B (a bigger Qwen switches too; Llama 3.1 8B picks the
      stars with or without “brief”). Bars only sketch the comparison. Step 2's shared parts: found in all three models by
      swapping internal signals between questions.</span>
      <span style='white-space:nowrap'>Methods: forced choices · deleting words<br>· swapping internal signals</span></div>
    """
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS + css + "</style></head><body>"
            "<div id='fig' class='hero'>" + body + "</div></body></html>")


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


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
