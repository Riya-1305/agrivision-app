import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def build_pdf_report(img, disease, confidence, top3_idx, predictions, class_names, treatment):
    """
    Generates a production-grade PDF report using ReportLab with custom spacing, 
    organic forest-green typography, and clear visual hierarchies.
    """
    buffer = io.BytesIO()
    
    # Secure edge margins to prevent physical printer clipping
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=40, 
        leftMargin=40,
        topMargin=45,  
        bottomMargin=40
    )

    story = []

    # ── Typography Styles ───────────────────────────────────
    title_style = ParagraphStyle(
        "BannerTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#ffffff"),
        alignment=TA_CENTER,
        leading=22
    )
    
    sub_tag_style = ParagraphStyle(
        "SubTag",
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#10B981"), # Tech Emerald
        alignment=TA_CENTER,
        spaceBefore=8,
        spaceAfter=2,
        leading=12
    )
    
    meta_style = ParagraphStyle(
        "MetaText",
        fontSize=9,
        fontName="Helvetica",
        textColor=colors.HexColor("#5C6B61"), # Slate Sage Neutral
        alignment=TA_CENTER,
        spaceAfter=14,
        leading=12
    )

    sec_style = ParagraphStyle(
        "SectionHeading",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#142C1E"),
        spaceAfter=4
    )
        
    body_style = ParagraphStyle(
        "BodyText",
        fontSize=10,
        fontName="Helvetica",
        textColor=colors.HexColor("#1E2923"),
        leading=16,
        spaceAfter=4
    )

    # ── 1. Header Block Component ───────────────────────────
    story.append(Spacer(1, 10)) # Safety clearance padding at top of page

    title_p = Paragraph("AgriVision — Crop Disease Detection Report", title_style)
    banner_table = Table([[title_p]], colWidths=[6.5*inch])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#142C1E")), # Deep Forest Green
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 16),
        ("BOTTOMPADDING", (0,0), (-1,-1), 16),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    story.append(banner_table)
    
    story.append(Paragraph("[ Crop Disease Detection System ]", sub_tag_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=20))

    # ── 2. Leaf Image Processing Component ──────────────────
    img_buffer = io.BytesIO()
    img_rgb = img.convert("RGB")
    img_rgb = img_rgb.resize((300, 300))
    img_rgb.save(img_buffer, format="JPEG", quality=85)
    img_buffer.seek(0)

    rl_img = RLImage(img_buffer, width=2.8*inch, height=2.8*inch)
    rl_img.hAlign = "CENTER"
    story.append(rl_img)
    story.append(Spacer(1, 16))

    # ── 3. Primary Diagnosis Telemetry Matrix ───────────────
    is_healthy = "healthy" in disease.lower()
    status = "HEALTHY" if is_healthy else "DISEASE DETECTED"
    status_color = colors.HexColor("#142C1E") if is_healthy else colors.HexColor("#EF4444")
    disease_clean = disease.replace("_", " ").replace("  ", " ")

    diag_data = [
        ["Status",     status],
        ["Diagnosis",  disease_clean],
        ["Confidence", f"{confidence:.1f}%"],
        ["Crop",       disease_clean.split(" ")[0]],
    ]
    diag_table = Table(diag_data, colWidths=[1.8*inch, 4.7*inch])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#FAF9F5")),
        ("BACKGROUND",    (1, 0), (1,  0), status_color),
        ("TEXTCOLOR",     (1, 0), (1,  0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",      (0, 0), (0,  -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("FONTSIZE",      (1, 0), (1,   0), 11),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E6E4DD")),
        ("PADDING",       (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 20))

    # ── 4. Treatment Protocol Recommendations ───────────────
    story.append(Paragraph("Treatment Recommendation", sec_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=10))
    story.append(Paragraph(treatment, body_style))
    story.append(Spacer(1, 20))

    # ── 5. Multi-Class Softmax Differential Diagnosis ────────
    story.append(Paragraph("Top 3 Predictions", sec_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=10))

    pred_data = [["Rank", "Disease", "Confidence"]]
    for rank, i in enumerate(top3_idx, 1):
        pred_data.append([
            f"#{rank}",
            class_names[i].replace("_", " "),
            f"{predictions[0][i]*100:.1f}%"
        ])

    pred_table = Table(pred_data, colWidths=[0.8*inch, 4.4*inch, 1.3*inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#142C1E")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1,-1), 10),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
        ("GRID",          (0, 0), (-1,-1), 0.5, colors.HexColor("#E6E4DD")),
        ("PADDING",       (0, 0), (-1,-1), 9),
        ("ALIGN",         (0, 0), (-1,-1), "CENTER"),
        ("ALIGN",         (1, 0), (1, -1), "LEFT"),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 24))

    # ── 6. Legal / System Metadata Footer ───────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=10))
    footer_style = ParagraphStyle("footer", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#5C6B61"), alignment=TA_CENTER, leading=12)
    story.append(Paragraph(
        "AgriVision · MobileNetV2 Architecture · PlantVillage Dataset Control · "
        "For advisory purposes only — consult a local professional agronomist for active field validation.",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer