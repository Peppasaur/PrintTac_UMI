#!/usr/bin/env python3
"""Generate a paper-ready one-slide overview of the first-frame diffusion policy."""

from __future__ import annotations

import argparse
import html
import zipfile
from pathlib import Path


W, H = 1600, 900

NAVY = "#17324D"
TEXT = "#183247"
MUTED = "#5B6B7A"
LINE = "#B8C5D0"
BLUE = "#1F78B4"
CYAN = "#1B9AAA"
GREEN = "#2E8B57"
ORANGE = "#D97706"
VIOLET = "#6C5CE7"
RED = "#C0392B"
PALE_BLUE = "#EAF4FB"
PALE_CYAN = "#E8F7F8"
PALE_GREEN = "#EAF6EF"
PALE_ORANGE = "#FFF4E5"
PALE_VIOLET = "#F0EDFF"
PALE_RED = "#FCEDEA"


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def rect(x, y, w, h, fill, stroke=LINE, sw=1.5, rx=14, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def line(x1, y1, x2, y2, color=LINE, sw=2, dashed=False, arrow=False):
    attrs = [
        f'x1="{x1}"',
        f'y1="{y1}"',
        f'x2="{x2}"',
        f'y2="{y2}"',
        f'stroke="{color}"',
        f'stroke-width="{sw}"',
        'fill="none"',
    ]
    if dashed:
        attrs.append('stroke-dasharray="8 6"')
    if arrow:
        attrs.append('marker-end="url(#arrow)"')
    return f'<line {" ".join(attrs)}/>'


def path(d, color=LINE, sw=2, dashed=False, arrow=False):
    attrs = [f'd="{d}"', f'stroke="{color}"', f'stroke-width="{sw}"', 'fill="none"']
    if dashed:
        attrs.append('stroke-dasharray="8 6"')
    if arrow:
        attrs.append('marker-end="url(#arrow)"')
    return f'<path {" ".join(attrs)}/>'


def text(x, y, value, size=18, color=TEXT, weight=400, anchor="start", family="Arial"):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}px" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def multiline(x, y, lines, size=16, color=TEXT, weight=400, gap=22, anchor="start"):
    output = [
        f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}px" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
    ]
    for index, value in enumerate(lines):
        dy = 0 if index == 0 else gap
        output.append(f'<tspan x="{x}" dy="{dy if index else 0}">{esc(value)}</tspan>')
    output.append("</text>")
    return "".join(output)


def pill(x, y, label, fill, color, width=None):
    width = width or (len(label) * 9 + 28)
    return (
        rect(x, y, width, 30, fill, stroke=fill, sw=0, rx=15)
        + text(x + width / 2, y + 21, label, size=13, color=color, weight=700, anchor="middle")
    )


def card(x, y, w, h, title, body, fill, accent, body_size=15, title_size=20):
    parts = [rect(x, y, w, h, fill, stroke=accent, sw=1.8, rx=16)]
    parts.append(text(x + 20, y + 32, title, size=title_size, color=accent, weight=700))
    if isinstance(body, str):
        body = [body]
    parts.append(multiline(x + 20, y + 60, body, size=body_size, color=TEXT, gap=21))
    return "".join(parts)


def sensor_icon(x, y, color):
    return (
        f'<circle cx="{x}" cy="{y}" r="13" fill="{color}" opacity="0.14" stroke="{color}" stroke-width="2"/>'
        + f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>'
    )


def build_svg():
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#718394"/></marker>',
        '<marker id="arrowBlue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1F78B4"/></marker>',
        "</defs>",
        rect(0, 0, W, H, "#FFFFFF", stroke="#FFFFFF", sw=0, rx=0),
        text(44, 54, "First-frame visual-tactile diffusion policy", size=31, color=NAVY, weight=700),
        text(45, 84, "Episode-level visual context + temporally reactive magnetic tactile feedback", size=17, color=MUTED),
        pill(1192, 30, "RGB ONCE / EPISODE", PALE_BLUE, BLUE, 175),
        pill(1380, 30, "TACTILE TEMPORAL", PALE_GREEN, GREEN, 166),
    ]

    # Column containers.
    columns = [(40, 480, "01", "Observation conditioning", PALE_BLUE, BLUE),
               (560, 480, "02", "Conditional diffusion", PALE_VIOLET, VIOLET),
               (1080, 480, "03", "Action generation & control", PALE_RED, RED)]
    for x, w, number, label, fill, accent in columns:
        s.append(rect(x, 125, w, 610, "#FFFFFF", stroke="#D7E0E8", sw=1.4, rx=20))
        s.append(rect(x, 125, w, 58, fill, stroke=fill, sw=0, rx=20))
        s.append(rect(x, 163, w, 20, fill, stroke=fill, sw=0, rx=0))
        s.append(text(x + 22, 162, number, size=17, color=accent, weight=700))
        s.append(text(x + 72, 162, label, size=20, color=NAVY, weight=700))

    # Observation column.
    s.append(card(65, 205, 430, 126, "Episode-level RGB", [
        "First wrist-camera image only",
        "Resize + ImageNet normalization",
        "ResNet-18 -> 512D feature; cache for episode",
    ], PALE_BLUE, BLUE, body_size=14, title_size=19))
    # Camera glyph.
    s.append(rect(407, 241, 65, 45, "#FFFFFF", stroke=BLUE, sw=2, rx=5))
    s.append(rect(417, 230, 22, 12, PALE_BLUE, stroke=BLUE, sw=2, rx=4))
    s.append('<circle cx="439" cy="264" r="12" fill="none" stroke="#1F78B4" stroke-width="2"/>')
    s.append('<circle cx="439" cy="264" r="4" fill="#1F78B4"/>')
    s.append(text(439, 309, "t = 0", size=13, color=BLUE, weight=700, anchor="middle"))

    s.append(card(65, 365, 430, 146, "Current magnetic tactile", [
        "Five sensors -> XYZ channels -> 15D embedding",
        "Updated at every policy step",
        "Provides contact-dependent corrections",
    ], PALE_GREEN, GREEN, body_size=14, title_size=19))
    for idx in range(5):
        xx = 100 + idx * 56
        s.append(sensor_icon(xx, 483, GREEN if idx != 2 else ORANGE))
        s.append(text(xx, 507, f"S{idx + 1}", size=12, color=GREEN if idx != 2 else ORANGE, weight=700, anchor="middle"))
    s.append(line(248, 331, 248, 365, color=BLUE, sw=2, arrow=True))
    s.append(line(248, 511, 248, 550, color=GREEN, sw=2, arrow=True))

    s.append(rect(65, 550, 430, 140, PALE_ORANGE, stroke=ORANGE, sw=1.8, rx=16))
    s.append(text(85, 582, "Sequence-aware fusion", size=19, color=ORANGE, weight=700))
    # Sequence slots.
    s.append(text(90, 616, "visual", size=13, color=MUTED, weight=700))
    s.append(rect(143, 598, 96, 28, "#D7ECFA", stroke=BLUE, sw=1.3, rx=5))
    s.append(text(191, 618, "512D", size=13, color=BLUE, weight=700, anchor="middle"))
    s.append(rect(248, 598, 96, 28, "#F3F5F7", stroke=LINE, sw=1.3, rx=5))
    s.append(text(296, 618, "0", size=13, color=MUTED, weight=700, anchor="middle"))
    s.append(text(90, 658, "tactile", size=13, color=MUTED, weight=700))
    s.append(rect(143, 640, 96, 28, "#DDF1E5", stroke=GREEN, sw=1.3, rx=5))
    s.append(text(191, 660, "15D", size=13, color=GREEN, weight=700, anchor="middle"))
    s.append(rect(248, 640, 96, 28, "#DDF1E5", stroke=GREEN, sw=1.3, rx=5))
    s.append(text(296, 660, "15D", size=13, color=GREEN, weight=700, anchor="middle"))
    s.append(text(365, 618, "t-1", size=12, color=MUTED, weight=700, anchor="middle"))
    s.append(text(365, 660, "t", size=12, color=MUTED, weight=700, anchor="middle"))
    s.append(line(495, 620, 545, 620, color=ORANGE, sw=2.4, arrow=True))

    # Diffusion column.
    s.append(card(590, 205, 420, 76, "Noisy action trajectory x_t", ["16 control steps x 10D action"], PALE_ORANGE, ORANGE, body_size=15, title_size=19))
    s.append(text(610, 332, "Diffusion step", size=17, color=VIOLET, weight=700))
    s.append(rect(590, 347, 176, 75, PALE_VIOLET, stroke=VIOLET, sw=1.6, rx=13))
    s.append(text(678, 378, "t", size=25, color=VIOLET, weight=700, anchor="middle"))
    s.append(text(678, 403, "sinusoidal + MLP -> 128D", size=12, color=TEXT, anchor="middle"))
    s.append(rect(790, 347, 220, 75, PALE_VIOLET, stroke=VIOLET, sw=1.6, rx=13))
    s.append(text(900, 378, "Condition fusion", size=17, color=VIOLET, weight=700, anchor="middle"))
    s.append(text(900, 403, "c_obs + time embedding", size=12, color=TEXT, anchor="middle"))
    s.append(line(678, 422, 678, 455, color=VIOLET, sw=2, arrow=True))
    s.append(line(900, 422, 900, 455, color=VIOLET, sw=2, arrow=True))
    s.append(rect(590, 455, 420, 174, PALE_VIOLET, stroke=VIOLET, sw=1.9, rx=16))
    s.append(text(610, 484, "Conditional 1D U-Net denoiser", size=19, color=VIOLET, weight=700))
    # U-Net blocks.
    blocks = [(610, 520, 85, 70, "512", "down"), (718, 520, 85, 70, "1024", "down"),
              (826, 510, 105, 90, "2048", "mid"), (954, 520, 36, 70, "", "up")]
    for bx, by, bw, bh, label, kind in blocks:
        fill = "#DCD5FF" if kind == "mid" else "#FFFFFF"
        s.append(rect(bx, by, bw, bh, fill, stroke=VIOLET, sw=1.5, rx=8))
        if label:
            s.append(text(bx + bw / 2, by + bh / 2 + 5, label, size=17, color=VIOLET, weight=700, anchor="middle"))
    s.append(line(695, 555, 718, 555, color=VIOLET, sw=1.8, arrow=True))
    s.append(line(803, 555, 826, 555, color=VIOLET, sw=1.8, arrow=True))
    s.append(line(931, 555, 954, 555, color=VIOLET, sw=1.8, arrow=True))
    s.append(path("M 650 515 C 650 490, 950 490, 970 515", color="#9B8FEF", sw=1.5, dashed=True))
    s.append(path("M 760 515 C 760 495, 950 495, 970 535", color="#9B8FEF", sw=1.5, dashed=True))
    s.append(text(800, 618, "Conv1D k=5 | GN(8) | Mish | residual + skip", size=12, color=MUTED, anchor="middle"))
    s.append(text(800, 644, "FiLM injects channel-wise scale + bias", size=13, color=VIOLET, weight=700, anchor="middle"))
    s.append(rect(590, 655, 420, 50, PALE_VIOLET, stroke=VIOLET, sw=1.6, rx=12))
    s.append(text(800, 687, "epsilon_pred [16 x 10]", size=18, color=VIOLET, weight=700, anchor="middle"))
    s.append(line(1010, 680, 1060, 680, color=VIOLET, sw=2.4, arrow=True))
    s.append(path("M 545 620 C 560 620, 565 370, 590 370", color=ORANGE, sw=2.2, arrow=True))

    # Action/control column.
    s.append(card(1110, 205, 420, 80, "DDIM reverse diffusion", ["100 denoising steps -> clean trajectory"], PALE_ORANGE, ORANGE, body_size=15, title_size=19))
    s.append(line(1060, 680, 1085, 680, color=VIOLET, sw=2.2, arrow=True))
    s.append(path("M 1085 680 C 1090 680, 1090 245, 1110 245", color=ORANGE, sw=2.2, arrow=True))
    s.append(card(1110, 330, 420, 82, "Clean action trajectory", ["[16 x 10] = 3D position + 6D rotation + gripper"], PALE_BLUE, BLUE, body_size=14, title_size=19))
    s.append(line(1320, 285, 1320, 330, color=ORANGE, sw=2.2, arrow=True))
    s.append(card(1110, 455, 420, 110, "Receding-horizon execution", ["Execute current action chunk", "Re-observe tactile and re-plan at 12 Hz"], PALE_RED, RED, body_size=15, title_size=19))
    s.append(line(1320, 412, 1320, 455, color=BLUE, sw=2.2, arrow=True))
    s.append(card(1110, 610, 420, 92, "Robot command", ["Relative TCP action; absolute gripper opening"], PALE_RED, RED, body_size=15, title_size=19))
    s.append(line(1320, 565, 1320, 610, color=RED, sw=2.2, arrow=True))
    s.append(path("M 1515 535 C 1555 535, 1555 430, 1515 430", color=RED, sw=1.8, dashed=True, arrow=True))
    s.append(text(1512, 485, "tactile", size=12, color=RED, weight=700, anchor="end"))
    s.append(text(1512, 501, "feedback", size=12, color=RED, weight=700, anchor="end"))

    # Training-only strip.
    s.append(rect(40, 770, 1520, 92, "#F8FAFC", stroke="#C9D4DE", sw=1.4, rx=16))
    s.append(text(62, 798, "TRAINING OBJECTIVE", size=13, color=MUTED, weight=700))
    s.append(pill(62, 812, "x_0 action", "#FFFFFF", NAVY, 104))
    s.append(line(170, 827, 237, 827, color="#718394", sw=1.8, dashed=True, arrow=True))
    s.append(pill(245, 812, "add Gaussian noise", "#FFFFFF", NAVY, 145))
    s.append(line(395, 827, 462, 827, color="#718394", sw=1.8, dashed=True, arrow=True))
    s.append(pill(470, 812, "x_t at random t", "#FFFFFF", NAVY, 135))
    s.append(line(610, 827, 678, 827, color="#718394", sw=1.8, dashed=True, arrow=True))
    s.append(pill(686, 812, "U-Net -> eps_pred", "#FFFFFF", VIOLET, 150))
    s.append(line(842, 827, 906, 827, color="#718394", sw=1.8, dashed=True, arrow=True))
    s.append(text(925, 833, "L = ||eps_pred - eps||^2", size=19, color=NAVY, weight=700))
    s.append(line(1275, 795, 1340, 795, color="#718394", sw=1.8, dashed=True))
    s.append(text(1350, 801, "dashed = training", size=13, color=MUTED, weight=700))
    s.append(line(1275, 830, 1340, 830, color="#718394", sw=1.8))
    s.append(text(1350, 836, "solid = inference", size=13, color=MUTED, weight=700))

    s.append(text(800, 888, "RGB enters the policy once per episode; magnetic tactile features remain temporally reactive.", size=14, color=MUTED, anchor="middle"))
    s.append("</svg>")
    return "".join(s)


def _content_types():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="svg" ContentType="image/svg+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def _presentation():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>"""


def _presentation_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""


def _slide_master():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Master"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def _slide_layout():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"""


def _slide():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Diffusion policy overview"><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    <p:pic><p:nvPicPr><p:cNvPr id="2" name="Diffusion policy network overview"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def _theme():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Paper">
  <a:themeElements><a:clrScheme name="Paper"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="17324D"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="1F78B4"/></a:accent1><a:accent2><a:srgbClr val="6C5CE7"/></a:accent2><a:accent3><a:srgbClr val="2E8B57"/></a:accent3><a:accent4><a:srgbClr val="D97706"/></a:accent4><a:accent5><a:srgbClr val="C0392B"/></a:accent5><a:accent6><a:srgbClr val="1B9AAA"/></a:accent6><a:hlink><a:srgbClr val="1F78B4"/></a:hlink><a:folHlink><a:srgbClr val="6C5CE7"/></a:folHlink></a:clrScheme><a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Paper"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements>
</a:theme>"""


def _rels(targets):
    body = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for relation_id, relation_type, target in targets:
        body.append(f'<Relationship Id="{relation_id}" Type="{relation_type}" Target="{target}"/>')
    body.append("</Relationships>")
    return "".join(body)


def write_pptx(path: Path, svg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = {
            "[Content_Types].xml": _content_types(),
            "_rels/.rels": _rels([
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
                ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
                ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
            ]),
            "ppt/presentation.xml": _presentation(),
            "ppt/_rels/presentation.xml.rels": _presentation_rels(),
            "ppt/slideMasters/slideMaster1.xml": _slide_master(),
            "ppt/slideMasters/_rels/slideMaster1.xml.rels": _rels([
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
            ]),
            "ppt/slideLayouts/slideLayout1.xml": _slide_layout(),
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels": _rels([
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
            ]),
            "ppt/theme/theme1.xml": _theme(),
            "ppt/slides/slide1.xml": _slide(),
            "ppt/slides/_rels/slide1.xml.rels": _rels([
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", "../media/network_overview.svg"),
                ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ]),
            "docProps/core.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>First-frame visual-tactile diffusion policy</dc:title><dc:creator>Reactive Diffusion Policy</dc:creator><dc:description>Paper-ready network overview</dc:description></cp:coreProperties>""",
            "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Reactive Diffusion Policy</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>1</Slides></Properties>""",
        }
        for name, content in files.items():
            archive.writestr(name, content.encode("utf-8"))
        archive.writestr("ppt/media/network_overview.svg", svg.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    svg_path = args.output_dir / "diffusion_policy_network_overview.svg"
    pptx_path = args.output_dir / "diffusion_policy_network_overview.pptx"
    svg_path.write_text(svg, encoding="utf-8")
    write_pptx(pptx_path, svg)
    print(f"Wrote {svg_path}")
    print(f"Wrote {pptx_path}")


if __name__ == "__main__":
    main()
