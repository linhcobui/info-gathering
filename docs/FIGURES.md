# Figure style — rules for every graphic

Applies to every plot, chart, video overlay and diagram made for this project. In matplotlib, call
`figure_style.apply()` from [scripts/figure_style.py](../scripts/figure_style.py) before plotting; it sets the font,
colours and axes below.

## Colours

Use only these ten colours.

| Hex | Name | Role |
|---|---|---|
| `#ff9500` | orange | **highlight**: our method, or the one thing the figure is about. At most one highlighted thing per figure. |
| `#0048a6` | dark blue | interesting baseline / alternative (1st choice) |
| `#0091ff` | bright blue | interesting baseline / alternative (2nd) |
| `#740cad` | purple | interesting baseline / alternative (3rd) |
| `#0d948f` | teal | interesting baseline / alternative (4th) |
| `#850f67` | plum | interesting baseline / alternative (5th) |
| `#cdb3ff` | lavender | interesting baseline / alternative (6th), or a light fill / band |
| `#000000` | black | "obvious" baseline or reference (e.g. Naive, oracle line); text and axes |
| `#6b6b6b` | dark grey | "obvious" baseline; secondary text, tick labels |
| `#b3b3b3` | light grey | "obvious" baseline; things that don't matter (unused codes, background points), grid |

Rules:
- **One highlight.** Orange marks what the reader should look at: our method, or the condition the figure is
  about. Never use orange for anything else.
- **Dull for obvious, bright for interesting.** Baselines whose behaviour is expected (Naive, deterministic prior,
  chance level) get black / greys. Methods worth comparing against get the blues and purples, in the table's
  order.
- **Colour follows the thing, not its rank.** A method keeps its colour across every figure in a set.
- Don't rely on colour alone: label lines or bars directly where possible, and always include a legend for 2 or
  more series.

## Font

- **Palatino**, everywhere (titles, axis labels, ticks, legends, annotations).
- Palatino Linotype is a commercial font and is not installed here. Use its metric-compatible clone **P052**
  (URW Palladio, already installed at `/usr/share/fonts/opentype/urw-base35/`). The font list is
  `["Palatino Linotype", "P052", "TeX Gyre Pagella", "DejaVu Sans"]`, so a licensed Palatino Linotype is used
  automatically if one is ever dropped into `~/.local/share/fonts/` (then delete `~/.cache/matplotlib/fontlist-*.json`).
- PDFs embed the font (`pdf.fonttype = 42`).

## Every figure also

- Says exactly what each number is, in the caption or subtitle: the quantity (MSE, cross-entropy in nats,
  softmax probability, accuracy, ...), in which representation (memory `z`, frame embedding `e`, pixels, ...),
  against which target, averaged over what, and any normalisation. Define every symbol.
- States N (episodes / windows / blocks) per panel.
- Is saved under `figures/<topic>/` as PNG (dpi 150+) and, for anything going into a paper, PDF.
