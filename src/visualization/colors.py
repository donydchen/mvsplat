from PIL import ImageColor

# https://sashamaps.net/docs/resources/20-colors/
DISTINCT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]


def get_distinct_color(index: int) -> tuple[float, float, float]:
    hex = DISTINCT_COLORS[index % len(DISTINCT_COLORS)]
    return tuple(x / 255 for x in ImageColor.getcolor(hex, "RGB"))
