"""Cell shading for the OOD report tables (relative to a reference, dead zone unshaded) and its typst legend."""

DEAD_ZONE, FULL_TINT_AT, MAX_TINT = 0.02, 0.15, 0.32
BETTER, WORSE = (0x55, 0xA8, 0x68), (0xC4, 0x4E, 0x52)


def shade(delta, dead=DEAD_ZONE, full=FULL_TINT_AT, one_sided=False):
    """(r, g, b) for a signed distance from the reference (positive = better), None in the dead zone."""
    if delta > 0 and one_sided:
        return None
    strength = min(max(abs(delta) - dead, 0) / (full - dead), 1.0) * MAX_TINT
    if strength <= 0:
        return None
    return tuple(round(255 + (c - 255) * strength) for c in (BETTER if delta > 0 else WORSE))


def rgb(delta, **kw):
    c = shade(delta, **kw)
    return None if c is None else "#" + "".join(f"{v:02x}" for v in c)


def fill(text, delta, **kw):
    """A typst table cell, tinted or plain."""
    colour = rgb(delta, **kw)
    return f'table.cell(fill: rgb("{colour}"))[{text}]' if colour else f"[{text}]"


def legend(caption, span, low, high, one_sided=False, full=FULL_TINT_AT, reverse=False):
    """Colour bar under a table, sampled from shade(). `reverse` for axes where high is bad."""
    n = 29
    stops = []
    for i in range(n):
        d = -full * 1.15 + 2 * full * 1.15 * i / (n - 1)
        stops.append(shade(-d if reverse else d, full=full, one_sided=one_sided)
                     or (255, 255, 255))
    colours = ", ".join('rgb("#{}")'.format("".join(f"{c:02x}" for c in s)) for s in stops)
    ticks = f"[{low}], [reference], [{high}]"
    return f"""table.cell(colspan: {span}, inset: (top: 7pt, bottom: 3pt))[
    #stack(spacing: 4pt,
      text(size: 0.85em)[{caption}],
      rect(width: 100%, height: 8pt, stroke: 0.3pt + luma(170),
           fill: gradient.linear({colours})),
      text(size: 0.85em, grid(columns: (1fr, 1fr, 1fr),
        align: (left, center, right), {ticks})),
    )
  ],"""
