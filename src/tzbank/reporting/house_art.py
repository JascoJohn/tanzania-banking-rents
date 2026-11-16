"""House-style page art (prompt 16; D77): the cover and the faint corner network of inner pages.

The cover follows ``reports/design/house_style_cover.py`` (approved sample): A4 at 300 dpi, paper-coloured margin, a
bottle-green panel fading slightly toward forest green, a faint network in the upper right, a thin gold frame and a
short gold rule under the eyebrow. The inner-page art is the page colour with a faint network in two corners (top
right, bottom left); every area that carries text (running header, body, footer) is masked out, so the pattern is
never behind text. Both are deterministic (fixed seeds).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

PAPER, GOLD = (248, 250, 249), (230, 179, 35)
A4_CM = (21.0, 29.7)


def cover(out: Path, seed: int = 7) -> Path:
    """Cover art (house_style_cover.py, unchanged geometry)."""
    random.seed(seed)
    w, h = 2480, 3508
    inset, frame = 110, 9
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    for y in range(inset, h - inset):
        t = y / h
        d.line([(inset, y), (w - inset, y)], fill=(int(6 - 6 * t * 0.3), int(69 + 38 * t * 0.35),
                                                   int(46 + 17 * t * 0.35)))
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    pts = [(random.randint(int(w * 0.5), w - 200), random.randint(200, int(h * 0.5))) for _ in range(60)]
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            if math.dist(p, q) < 320:
                od.line([p, q], fill=(229, 241, 232, 30), width=3)
    for p in pts:
        od.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=(229, 241, 232, 55))
    img = Image.alpha_composite(img.convert("RGBA"), ov)
    d = ImageDraw.Draw(img)
    d.rectangle([inset, inset, w - inset, h - inset], outline=GOLD, width=frame)
    d.rectangle([270, 470, 560, 480], fill=GOLD)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, optimize=True)
    return out


def _network(od: ImageDraw.ImageDraw, box: tuple[int, int, int, int], n: int, reach: int, rng: random.Random) -> None:
    x0, y0, x1, y1 = box
    pts = [(rng.randint(x0, x1), rng.randint(y0, y1)) for _ in range(n)]
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            if math.dist(p, q) < reach:
                od.line([p, q], fill=(150, 175, 160, 60), width=2)
    for p in pts:
        od.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(150, 175, 160, 80))


def page(out: Path, margins_cm: dict, seed: int = 11) -> Path:
    """Inner-page art at 150 dpi: page colour, corner networks, text areas masked out.

    ``margins_cm``: left, right, top (body), bottom (body), header (distance), footer (distance), in cm."""
    rng = random.Random(seed)
    dpi = 150
    w, h = (round(c / 2.54 * dpi) for c in A4_CM)
    px = lambda cm: round(cm / 2.54 * dpi)  # noqa: E731
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    _network(od, (int(w * 0.66), 0, w, int(h * 0.11)), 34, px(2.2), rng)
    _network(od, (0, int(h * 0.88), int(w * 0.34), h), 34, px(2.2), rng)
    m = margins_cm
    mask = Image.new("L", (w, h), 255)
    md = ImageDraw.Draw(mask)
    left, right = px(m["left"]) - px(0.2), w - px(m["right"]) + px(0.2)
    # the whole text column from the top of the running header to the bottom of the footer line
    md.rectangle([left, px(m["header"]) - px(0.15), right, h - px(m["footer"]) + px(0.15)], fill=0)
    alpha = ov.getchannel("A")
    ov.putalpha(Image.composite(alpha, Image.new("L", (w, h), 0), mask))
    img = Image.alpha_composite(Image.new("RGBA", (w, h), PAPER + (255,)), ov).convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    return out
