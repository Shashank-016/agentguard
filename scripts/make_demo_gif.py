"""Render docs/demo_output.txt as a typewriter-style terminal GIF (docs/demo.gif).

Decouples capture from rendering so the GIF is reproducible and deterministic
without relying on asciinema/agg (which don't run well on Windows).

Run:
    python scripts/make_demo_gif.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "docs" / "demo_output.txt"
OUTPUT_FILE = ROOT / "docs" / "demo.gif"

BG = (13, 17, 23)
TITLE_BAR_BG = (22, 27, 34)
COLORS = {
    "red": (248, 81, 73),
    "green": (63, 185, 80),
    "amber": (210, 153, 34),
    "cyan": (88, 166, 255),
    "grey": (201, 209, 217),
}
DOT_COLORS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]

WIDTH = 900
FONT_SIZE = 18
LINE_HEIGHT = 24
PADDING = 20
TITLE_BAR_HEIGHT = 36
FRAME_DURATION_MS = 70
FINAL_HOLD_MS = 3000


def color_for(line: str) -> tuple[int, int, int]:
    lower = line.lower()
    if any(k in line for k in ("BLOCKED", "CRITICAL", "✗", "DEGRADED")):
        return COLORS["red"]
    if any(k in line for k in ("OK", "✓")) or "granted" in lower or "intact" in lower:
        return COLORS["green"]
    if "WARNING" in line or "flag" in lower:
        return COLORS["amber"]
    if any(c in line for c in ("━", "=")):
        return COLORS["cyan"]
    return COLORS["grey"]


def load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\CascadiaCode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    try:
        import matplotlib.font_manager as fm

        return ImageFont.truetype(fm.findfont("DejaVu Sans Mono"), size)
    except Exception:
        return ImageFont.load_default()


def clip_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text + "…"


def draw_title_bar(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> None:
    draw.rectangle([(0, 0), (WIDTH, TITLE_BAR_HEIGHT)], fill=TITLE_BAR_BG)
    for i, color in enumerate(DOT_COLORS):
        cx = 22 + i * 22
        cy = TITLE_BAR_HEIGHT // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=color)
    label = "agentmoat mcp proxy demo"
    text_width = draw.textlength(label, font=font)
    draw.text(
        ((WIDTH - text_width) / 2, (TITLE_BAR_HEIGHT - FONT_SIZE) / 2 - 1),
        label,
        font=font,
        fill=COLORS["grey"],
    )


def render_frame(
    lines: list[str], reveal_count: int, font: ImageFont.FreeTypeFont, height: int
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    draw_title_bar(draw, font)

    text_max_width = WIDTH - 2 * PADDING
    y = TITLE_BAR_HEIGHT + PADDING
    for line in lines[:reveal_count]:
        clipped = clip_to_width(draw, line, font, text_max_width)
        draw.text((PADDING, y), clipped, font=font, fill=color_for(line))
        y += LINE_HEIGHT
    return image


def main() -> None:
    raw_lines = INPUT_FILE.read_text(encoding="utf-8").splitlines()
    lines = [line if line.strip() else " " for line in raw_lines]

    font = load_font()
    height = TITLE_BAR_HEIGHT + 2 * PADDING + LINE_HEIGHT * len(lines)

    frames = [render_frame(lines, count, font, height) for count in range(1, len(lines) + 1)]
    durations = [FRAME_DURATION_MS] * (len(frames) - 1) + [FINAL_HOLD_MS]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT_FILE,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Wrote {OUTPUT_FILE} — {WIDTH}x{height}px, {len(frames)} frames, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
