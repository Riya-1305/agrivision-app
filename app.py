import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision — Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ── Load model + class names ──────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("agrivision_model.h5")

@st.cache_resource
def load_class_names():
    with open("class_names.txt") as f:
        return [line.strip() for line in f.readlines()]

model      = load_model()
class_names = load_class_names()

# ── Treatment dictionary (all 38 classes) ────────────────
TREATMENTS = {
    "Apple___Apple_scab":
        "Apply fungicide (captan or myclobutanil) every 7–10 days. Remove fallen leaves. Prune for airflow.",
    "Apple___Black_rot":
        "Remove infected fruit and cankers. Spray copper-based fungicide. Avoid overhead irrigation.",
    "Apple___Cedar_apple_rust":
        "Apply myclobutanil fungicide at bud break. Remove nearby juniper hosts if possible.",
    "Apple___healthy":
        "No disease detected. Maintain regular watering and balanced NPK fertilisation.",
    "Blueberry___healthy":
        "No disease detected. Ensure acidic soil pH (4.5–5.5) and good drainage.",
    "Cherry_(including_sour)___Powdery_mildew":
        "Apply sulfur or potassium bicarbonate spray. Improve air circulation by pruning.",
    "Cherry_(including_sour)___healthy":
        "No disease detected. Monitor regularly during humid conditions.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Apply strobilurin fungicide. Rotate crops annually. Use resistant hybrids next season.",
    "Corn_(maize)___Common_rust_":
        "Apply propiconazole fungicide early. Plant rust-resistant varieties. Monitor weekly.",
    "Corn_(maize)___Northern_Leaf_Blight":
        "Apply azoxystrobin fungicide. Practice crop rotation. Remove infected debris after harvest.",
    "Corn_(maize)___healthy":
        "No disease detected. Ensure adequate nitrogen and consistent moisture.",
    "Grape___Black_rot":
        "Apply mancozeb or myclobutanil fungicide. Remove mummified fruit. Prune for airflow.",
    "Grape___Esca_(Black_Measles)":
        "No chemical cure. Remove infected wood. Protect pruning wounds with fungicide paste.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Apply copper-based fungicide. Avoid wetting leaves. Remove heavily infected leaves.",
    "Grape___healthy":
        "No disease detected. Maintain proper canopy management and irrigation.",
    "Orange___Haunglongbing_(Citrus_greening)":
        "No cure available. Remove and destroy infected trees immediately. Control psyllid vectors with insecticide.",
    "Peach___Bacterial_spot":
        "Apply copper hydroxide spray during dormancy. Avoid overhead watering. Use resistant varieties.",
    "Peach___healthy":
        "No disease detected. Monitor for signs of bacterial spot during wet seasons.",
    "Pepper,_bell___Bacterial_spot":
        "Apply copper-based bactericide. Use disease-free seeds. Avoid working with wet plants.",
    "Pepper,_bell___healthy":
        "No disease detected. Ensure consistent watering and calcium-rich fertilisation.",
    "Potato___Early_blight":
        "Spray mancozeb or chlorothalonil every 7 days. Remove lower infected leaves. Ensure good drainage.",
    "Potato___Late_blight":
        "Apply metalaxyl + mancozeb immediately. Destroy infected plants. Avoid overhead irrigation.",
    "Potato___healthy":
        "No disease detected. Hill soil regularly and maintain consistent moisture.",
    "Raspberry___healthy":
        "No disease detected. Prune old canes after harvest to prevent disease buildup.",
    "Soybean___healthy":
        "No disease detected. Monitor for sudden death syndrome and soybean cyst nematode.",
    "Squash___Powdery_mildew":
        "Apply potassium bicarbonate or neem oil spray. Improve airflow. Avoid excess nitrogen.",
    "Strawberry___Leaf_scorch":
        "Remove infected leaves. Apply captan fungicide. Avoid overhead watering.",
    "Strawberry___healthy":
        "No disease detected. Mulch around plants to prevent soil splash.",
    "Tomato___Bacterial_spot":
        "Apply copper bactericide every 5–7 days. Use certified disease-free transplants.",
    "Tomato___Early_blight":
        "Spray chlorothalonil or mancozeb fungicide. Remove lower leaves. Practice crop rotation.",
    "Tomato___Late_blight":
        "Apply metalaxyl fungicide immediately. Remove infected plants. Avoid wetting foliage.",
    "Tomato___Leaf_Mold":
        "Improve greenhouse ventilation. Apply chlorothalonil fungicide. Reduce humidity below 85%.",
    "Tomato___Septoria_leaf_spot":
        "Apply mancozeb or copper fungicide. Remove infected leaves. Avoid overhead irrigation.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Apply miticide (abamectin). Spray undersides of leaves. Introduce predatory mites.",
    "Tomato___Target_Spot":
        "Apply azoxystrobin fungicide. Practice crop rotation. Remove plant debris after harvest.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "No cure. Remove infected plants immediately. Control whitefly vectors with imidacloprid.",
    "Tomato___Tomato_mosaic_virus":
        "No cure. Destroy infected plants. Disinfect tools with bleach solution. Control aphids.",
    "Tomato___healthy":
        "No disease detected. Maintain consistent watering and calcium supplementation.",
}

def generate_report(img, disease, confidence, top3_idx, predictions, class_names, treatment):
    buffer = io.BytesIO()
    
    # Increased topMargin slightly to 45 to protect against physical printer clipping
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=45,  bottomMargin=40)

    styles  = getSampleStyleSheet()
    story   = []

    # ── Header Styles ───────────────────────────────────────
    # AgAmerica Style Typography: Elegant Title Layout
    title_style = ParagraphStyle(
        "BannerTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#ffffff"),
        alignment=TA_CENTER,
        leading=22 # Explicit leading keeps text centered vertically in the box
    )
    
    sub_tag_style = ParagraphStyle(
        "SubTag",
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#10B981"), # Tech Emerald Accent
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

    # ── FIXED: Safe & Elegant Title Banner Block ───────────
    # A tiny padding spacer block that forces the table down away from the PDF ceiling
    story.append(Spacer(1, 10))

    # Wrapped text cleanly in a Paragraph so it responds correctly to vertical table styles
    title_p = Paragraph("AgriVision — Crop Disease Detection Report", title_style)
    
    # 6.5 inches perfectly spans your custom A4 horizontal margins safely
    banner_table = Table([[title_p]], colWidths=[6.5*inch])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#142C1E")), # Deep Corporate Forest Green
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 16), # Heavy internal padding creates a tall, premium banner box
        ("BOTTOMPADDING", (0,0), (-1,-1), 16),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]), # Clean rounded radius edges
    ]))
    story.append(banner_table)
    
    # Sub-Header Meta Data Structure
    story.append(Paragraph("[ Crop Disease Detection System ]", sub_tag_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", meta_style))
    
    # Thin divider rule mimicking AgAmerica's airy lines
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#E6E4DD"),
                            spaceAfter=20))

    # ── Leaf image ────────────────────────────────────────
    img_buffer = io.BytesIO()
    img_rgb    = img.convert("RGB")
    img_rgb    = img_rgb.resize((300, 300))
    img_rgb.save(img_buffer, format="JPEG", quality=85)
    img_buffer.seek(0)

    rl_img = RLImage(img_buffer, width=2.8*inch, height=2.8*inch)
    rl_img.hAlign = "CENTER"
    story.append(rl_img)
    story.append(Spacer(1, 16))

    # ── Diagnosis box ─────────────────────────────────────
    is_healthy    = "healthy" in disease.lower()
    status        = "HEALTHY" if is_healthy else "DISEASE DETECTED"
    status_color  = colors.HexColor("#142C1E") if is_healthy else colors.HexColor("#EF4444")
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

    # ── Treatment section ─────────────────────────────────
    sec_style = ParagraphStyle("sec",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#142C1E"),
        spaceAfter=4)
        
    body_style = ParagraphStyle("body",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#1E2923"),
        leading=16, spaceAfter=4)

    story.append(Paragraph("Treatment Recommendation", sec_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=10))
    story.append(Paragraph(treatment, body_style))
    story.append(Spacer(1, 20))

    # ── Top 3 predictions ─────────────────────────────────
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

    # ── Footer ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E6E4DD"), spaceAfter=10))
    footer_style = ParagraphStyle("footer", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#5C6B61"), alignment=TA_CENTER, leading=12)
    story.append(Paragraph(
        "AgriVision · MobileNetV2 Architecture · PlantVillage Dataset Control · "
        "For advisory purposes only — consult a local professional agronomist for active field validation.",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
    
# ── UI ────────────────────────────────────────────────────
st.title("🌿 AgriVision")
st.markdown("### Crop Disease Detector")
st.markdown(
    "Upload a leaf image to instantly detect disease and get treatment advice. "
    "Powered by MobileNetV2 trained on 54,000+ plant images."
)

st.divider()

uploaded = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"],
    help="Take a clear photo of the affected leaf in natural light"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="Uploaded leaf", use_column_width=True)

    with col2:
        with st.spinner("Analysing..."):
            img_resized = img.resize((224, 224))
            arr         = np.array(img_resized) / 255.0
            arr         = np.expand_dims(arr, axis=0)
            predictions = model.predict(arr, verbose=0)
            idx         = np.argmax(predictions[0])
            disease     = class_names[idx]
            confidence  = predictions[0][idx] * 100
            top3_idx    = np.argsort(predictions[0])[::-1][:3]

        # ── Results ───────────────────────────────────────
        is_healthy = "healthy" in disease.lower()
        status_icon = "✅" if is_healthy else "⚠️"

        st.markdown(f"#### {status_icon} Diagnosis")

        disease_display = disease.replace("_", " ").replace("  ", " ")
        st.metric("Detected", disease_display)
        st.metric("Confidence", f"{confidence:.1f}%")

        st.divider()
        st.markdown("#### 💊 Treatment Advice")
        treatment = TREATMENTS.get(disease, "Consult a local agronomist for advice.")
        if is_healthy:
            st.success(treatment)
        else:
            st.warning(treatment)

        st.divider()
        st.markdown("#### 📊 Top 3 Predictions")
        for i in top3_idx:
            name  = class_names[i].replace("_", " ")
            score = predictions[0][i] * 100
            st.progress(int(score), text=f"{name}  {score:.1f}%")

        st.divider()
        st.markdown("#### 📄 Download Report")
        report = generate_report(
            img, disease, confidence,
            top3_idx, predictions, class_names, treatment
        )
        filename = f"AgriVision_Report_{disease}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=report,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("👆 Upload a leaf image above to get started.")
    st.markdown("""
    **Supported crops:** Apple · Blueberry · Cherry · Corn · Grape ·
    Orange · Peach · Pepper · Potato · Raspberry · Soybean ·
    Squash · Strawberry · Tomato
    """)

st.divider()
st.caption(
    "AgriVision · MobileNetV2 · 95.44% accuracy · "
    "PlantVillage dataset · Built for Indian farmers"
)