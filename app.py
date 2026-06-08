import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from datetime import datetime

# ── Cleaned Imports: Internal PDF generator removed to prevent conflicts ──
from report_generator import build_pdf_report

st.set_page_config(
    page_title="AgriVision — Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ──────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load model + class names ──────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("agrivision_model.h5")

@st.cache_resource
def load_class_names():
    with open("class_names.txt") as f:
        return [line.strip() for line in f.readlines()]

model       = load_model()
class_names = load_class_names()

# ── Treatments ────────────────────────────────────────────
TREATMENTS = {
    "Apple___Apple_scab": "Apply fungicide (captan or myclobutanil) every 7–10 days. Remove fallen leaves. Prune for airflow.",
    "Apple___Black_rot": "Remove infected fruit and cankers. Spray copper-based fungicide. Avoid overhead irrigation.",
    "Apple___Cedar_apple_rust": "Apply myclobutanil fungicide at bud break. Remove nearby juniper hosts if possible.",
    "Apple___healthy": "No disease detected. Maintain regular watering and balanced NPK fertilisation.",
    "Blueberry___healthy": "No disease detected. Ensure acidic soil pH (4.5–5.5) and good drainage.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur or potassium bicarbonate spray. Improve air circulation by pruning.",
    "Cherry_(including_sour)___healthy": "No disease detected. Monitor regularly during humid conditions.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply strobilurin fungicide. Rotate crops annually. Use resistant hybrids next season.",
    "Corn_(maize)___Common_rust_": "Apply propiconazole fungicide early. Plant rust-resistant varieties. Monitor weekly.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply azoxystrobin fungicide. Practice crop rotation. Remove infected debris after harvest.",
    "Corn_(maize)___healthy": "No disease detected. Ensure adequate nitrogen and consistent moisture.",
    "Grape___Black_rot": "Apply mancozeb or myclobutanil fungicide. Remove mummified fruit. Prune for airflow.",
    "Grape___Esca_(Black_Measles)": "No chemical cure. Remove infected wood. Protect pruning wounds with fungicide paste.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicide. Avoid wetting leaves. Remove heavily infected leaves.",
    "Grape___healthy": "No disease detected. Maintain proper canopy management and irrigation.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure available. Remove and destroy infected trees immediately. Control psyllid vectors with insecticide.",
    "Peach___Bacterial_spot": "Apply copper hydroxide spray during dormancy. Avoid overhead watering. Use resistant varieties.",
    "Peach___healthy": "No disease detected. Monitor for signs of bacterial spot during wet seasons.",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericide. Use disease-free seeds. Avoid working with wet plants.",
    "Pepper,_bell___healthy": "No disease detected. Ensure consistent watering and calcium-rich fertilisation.",
    "Potato___Early_blight": "Spray mancozeb or chlorothalonil every 7 days. Remove lower infected leaves. Ensure good drainage.",
    "Potato___Late_blight": "Apply metalaxyl + mancozeb immediately. Destroy infected plants. Avoid overhead irrigation.",
    "Potato___healthy": "No disease detected. Hill soil regularly and maintain consistent moisture.",
    "Raspberry___healthy": "No disease detected. Prune old canes after harvest to prevent disease buildup.",
    "Soybean___healthy": "No disease detected. Monitor for sudden death syndrome and soybean cyst nematode.",
    "Squash___Powdery_mildew": "Apply potassium bicarbonate or neem oil spray. Improve airflow. Avoid excess nitrogen.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply captan fungicide. Avoid overhead watering.",
    "Strawberry___healthy": "No disease detected. Mulch around plants to prevent soil splash.",
    "Tomato___Bacterial_spot": "Apply copper bactericide every 5–7 days. Use certified disease-free transplants.",
    "Tomato___Early_blight": "Spray chlorothalonil or mancozeb fungicide. Remove lower leaves. Practice crop rotation.",
    "Tomato___Late_blight": "Apply metalaxyl fungicide immediately. Remove infected plants. Avoid wetting foliage.",
    "Tomato___Leaf_Mold": "Improve greenhouse ventilation. Apply chlorothalonil fungicide. Reduce humidity below 85%.",
    "Tomato___Septoria_leaf_spot": "Apply mancozeb or copper fungicide. Remove infected leaves. Avoid overhead irrigation.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticide (abamectin). Spray undersides of leaves. Introduce predatory mites.",
    "Tomato___Target_Spot": "Apply azoxystrobin fungicide. Practice crop rotation. Remove plant debris after harvest.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No cure. Remove infected plants immediately. Control whitefly vectors with imidacloprid.",
    "Tomato___Tomato_mosaic_virus": "No cure. Destroy infected plants. Disinfect tools with bleach solution. Control aphids.",
    "Tomato___healthy": "No disease detected. Maintain consistent watering and calcium supplementation.",
}

# ── HTML SECTIONS ─────────────────────────────────────────
def html(content):
    st.markdown(content, unsafe_allow_html=True)

# Announcement bar
html("""
<div class="announcement">
  AgriVision supports 38 disease classes across 14 crops &nbsp;·&nbsp;
  <a href="#crops">View all supported crops ↓</a>
</div>
""")

# Nav
html("""
<div class="nav">
  <div class="nav-logo">🌿 AgriVision</div>
  <div class="nav-links">
    <a href="#how-it-works">How It Works</a>
    <a href="#features">Features</a>
    <a href="#crops">Crops</a>
    <a href="#detect" class="nav-cta">Detect Disease →</a>
  </div>
</div>
""")

# Hero
html("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Agriculture</div>
  <h1>Detect Crop Disease<br/>Before It <em>Spreads.</em></h1>
  <p class="hero-sub">Upload a leaf photo. Get an instant diagnosis, confidence score,
  and treatment plan — powered by MobileNetV2 trained on 54,000+ plant images.</p>
  <div class="hero-btns">
    <a href="#detect" class="btn-primary">Analyse a Leaf →</a>
  </div>
  <div class="hero-stats">
    <div class="hero-stats-container" style="display: flex; justify-content: center; gap: 40px; margin-top: 30px;">
      <div class="hero-stat"><div class="val">95.4%</div><div class="lbl">Accuracy</div></div>
      <div class="hero-stat"><div class="val">38</div><div class="lbl">Diseases</div></div>
      <div class="hero-stat"><div class="val">54K+</div><div class="lbl">Images</div></div>
      <div class="hero-stat"><div class="val">14</div><div class="lbl">Crops</div></div>
    </div>
  </div>
</div>
""")

# How it works
html("""
<div class="section section-white" id="how-it-works">
  <div class="section-eyebrow">Simple by design</div>
  <div class="section-title">From photo to diagnosis in seconds</div>
  <div class="section-sub">No expertise required. AgriVision does the heavy lifting.</div>
  <div class="steps-grid">
    <div class="step"><div class="step-num">01</div><div class="step-icon">📷</div>
      <h3>Upload a Leaf Photo</h3><p>Take a clear photo of any affected leaf in natural light and upload it to the app.</p></div>
    <div class="step"><div class="step-num">02</div><div class="step-icon">🧠</div>
      <h3>AI Analyses the Image</h3><p>MobileNetV2 scans across 38 disease patterns and returns results in under 2 seconds.</p></div>
    <div class="step"><div class="step-num">03</div><div class="step-icon">⚠️</div>
      <h3>Get Your Diagnosis</h3><p>See the detected disease, confidence score, and top 3 alternative predictions.</p></div>
    <div class="step"><div class="step-num">04</div><div class="step-icon">💊</div>
      <h3>Download PDF Report</h3><p>Get a detailed treatment plan and shareable PDF report — instantly.</p></div>
  </div>
</div>
""")

# Accuracy banner
html("""
<div class="accuracy-banner">
  <div class="accuracy-left">
    <h2>Built on world-class agricultural AI research</h2>
    <p>Trained on the PlantVillage benchmark dataset — the gold standard for plant pathology machine learning.</p>
  </div>
  <div class="accuracy-stats">
    <div class="acc-stat"><div class="val">95.4%</div><div class="lbl">Test Accuracy</div></div>
    <div class="acc-stat"><div class="val">54K</div><div class="lbl">Training Images</div></div>
    <div class="acc-stat"><div class="val">38</div><div class="lbl">Disease Classes</div></div>
  </div>
</div>
""")

# Features
html("""
<div class="section section-cream" id="features">
  <div class="section-eyebrow">Everything you need</div>
  <div class="section-title">A complete disease management toolkit</div>
  <div class="section-sub">AgriVision goes beyond detection — giving farmers the full picture to act fast.</div>
  <div class="features-grid">
    <div class="feature-card"><div class="feature-icon">🔬</div><h3>Transfer Learning Model</h3>
      <p>MobileNetV2 fine-tuned with two-stage training — ImageNet features adapted to plant pathology.</p></div>
    <div class="feature-card"><div class="feature-icon">📊</div><h3>Confidence Scoring</h3>
      <p>Every diagnosis includes a confidence percentage and top 3 alternative predictions.</p></div>
    <div class="feature-card"><div class="feature-icon">💊</div><h3>Treatment Recommendations</h3>
      <p>Crop-specific treatment advice including fungicide names, spray schedules, and prevention tips.</p></div>
    <div class="feature-card"><div class="feature-icon">📄</div><h3>PDF Report Export</h3>
      <p>Professional diagnostic report with leaf image, diagnosis, and treatment — shareable with agronomists.</p></div>
    <div class="feature-card"><div class="feature-icon">⚡</div><h3>Real-Time Inference</h3>
      <p>Results in under 2 seconds. Upload your image and get your diagnosis instantly from any device.</p></div>
    <div class="feature-card"><div class="feature-icon">🌾</div><h3>Indian Crop Coverage</h3>
      <p>Covers tomato, potato, corn and other crops central to Indian agriculture.</p></div>
  </div>
</div>
""")

# Crops
html("""
<div class="section section-white" id="crops">
  <div class="section-eyebrow">Coverage</div>
  <div class="section-title">14 crops. 38 disease classes.</div>
  <div class="section-sub">From staple grains to high-value fruits — covering the crops that matter most.</div>
  <div class="crops-grid">
    <span class="crop-tag">🍅 Tomato</span>
    <span class="crop-tag">🥔 Potato</span>
    <span class="crop-tag">🌽 Corn / Maize</span>
    <span class="crop-tag">🍎 Apple</span>
    <span class="crop-tag">🍇 Grape</span>
    <span class="crop-tag">🍑 Peach</span>
    <span class="crop-tag">🍒 Cherry</span>
    <span class="crop-tag">🫐 Blueberry</span>
    <span class="crop-tag">🍊 Orange</span>
    <span class="crop-tag">🫑 Pepper</span>
    <span class="crop-tag">🍓 Strawberry</span>
    <span class="crop-tag">🌿 Soybean</span>
    <span class="crop-tag">🌱 Raspberry</span>
    <span class="crop-tag">🥒 Squash</span>
  </div>
</div>
""")

# ── DETECT SECTION ────────────────────────────────────────
html("""
<div class="section section-cream" id="detect">
  <div class="section-eyebrow">Live detection</div>
  <div class="section-title">Upload a leaf. Get your diagnosis.</div>
  <div class="section-sub">No signup required. Upload any leaf photo and get an instant AI-powered diagnosis with treatment advice.</div>
</div>
""")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # 1. Create a temporary check to see if a file has already been uploaded in the session
    # This prevents the NameError while letting us control the HTML visibility
    has_file = st.session_state.get("file_uploader_key")
    
    if not has_file:
        html("""
        <div class="upload-zone" style="margin-bottom: 15px;">
          <div class="upload-icon">🍃</div>
          <div class="upload-title">Drop your leaf image here</div>
          <div class="upload-sub">JPG or PNG · Clear photo in natural light</div>
        </div>
        """)
    
    # 2. Render the interactive uploader tool cleanly below your graphic block
    uploaded = st.file_uploader(
        "Select leaf specimen image:",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
        key="file_uploader_key" # Using a key syncs it up with our session check above
    )
    
    # 3. Handle image rendering if a user successfully inputs a leaf photo
    if uploaded:
        st.success("✓ Image uploaded successfully!")
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True, caption="Uploaded Leaf Preview")

with col2:
    if uploaded:
        with st.spinner("Analysing leaf..."):
            img_resized = img.resize((224,224))
            arr         = np.array(img_resized) / 255.0
            arr         = np.expand_dims(arr, 0)
            preds       = model.predict(arr, verbose=0)
            idx         = np.argmax(preds[0])
            disease     = class_names[idx]
            confidence  = preds[0][idx] * 100
            top3_idx    = np.argsort(preds[0])[::-1][:3]
            is_healthy  = "healthy" in disease.lower()
            treatment   = TREATMENTS.get(disease, "Consult a local agronomist.")
            disease_clean = disease.replace("_"," ").replace("  "," ")

        badge     = "badge-healthy" if is_healthy else "badge-disease"
        badge_txt = "Healthy" if is_healthy else "Disease Detected"
        conf_w    = int(confidence)

        top3_html = ""
        for rank, i in enumerate(top3_idx, 1):
            n = class_names[i].replace("_", " ").replace("  ", " ")
            p = float(preds[0][i] * 100)
            w = int(p)
            top3_html += (
                '<div class="pred-row">'
                f'<span class="pred-rank">#{rank}</span>'
                f'<span class="pred-name">{n}</span>'
                f'<div class="pred-bar-wrap"><div class="pred-bar" style="width:{w}%"></div></div>'
                f'<span class="pred-pct">{p:.1f}%</span>'
                '</div>'
            )
        
        html(f"""
        <div class="result-card">
          <div class="result-header">
            <div class="disease-name">{disease_clean}</div>
            <div class="{badge}">{badge_txt}</div>
          </div>
          <div class="result-body">
            <div class="metrics-row">
              <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{conf_w}%"></div></div>
              </div>
              <div class="metric-box">
                <div class="metric-label">Crop</div>
                <div class="metric-value">{disease_clean.split()[0]}</div>
              </div>
              <div class="metric-box">
                <div class="metric-label">Status</div>
                <div class="metric-value">{"OK" if is_healthy else "Alert"}</div>
              </div>
            </div>
            <div class="treatment-box">
              <div class="treatment-label">Treatment Recommendation</div>
              <div class="treatment-text">{treatment}</div>
            </div>
            <div class="top3-section">
              <div class="top3-label">Top 3 Predictions</div>
              {top3_html}
            </div>
          </div>
        </div>
        """)

        # ── FIXED: Routing telemetry objects into report_generator.py cleanly ──
        pdf = build_pdf_report(
            img, disease, confidence, 
            top3_idx, preds, class_names, treatment
        )
        
        fname = f"AgriVision_Report_{disease_clean.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True
        )
    else:
        html("""
        <div style="padding:60px 20px;text-align:center;color:#777;border:1px solid rgba(15,61,34,0.1);border-radius:12px;background:#fff">
          <div style="font-size:48px;margin-bottom:16px">🌿</div>
          <div style="font-size:16px;font-weight:500;color:#0f3d22;margin-bottom:8px">Upload a leaf image to get started</div>
          <div style="font-size:13px;line-height:1.6">Your diagnosis, confidence score, and treatment<br/>recommendation will appear here.</div>
        </div>
        """)

# CTA
html("""
<div class="cta-section">
  <h2>Protect your crops.<br/>Start for free.</h2>
  <p>No signup. No fees. Upload a leaf image and get your diagnosis right now.</p>
  <a href="#detect" class="btn-gold">Analyse a Leaf →</a>
</div>
""")

# Footer
html("""
<div class="footer">
  <div class="footer-grid">
    <div class="footer-brand">
      <div class="footer-logo">🌿 AgriVision</div>
      <p>AI-powered crop disease detection for Indian farmers. Built with MobileNetV2 trained on 54,000+ plant images.</p>
      <span class="accuracy-pill">95.44% Accuracy</span>
    </div>
    <div class="footer-col">
      <h4>Product</h4>
      <a href="#detect">Detect Disease</a>
      <a href="#how-it-works">How It Works</a>
      <a href="#crops">Crops Supported</a>
    </div>
    <div class="footer-col">
      <h4>Technology</h4>
      <a href="https://github.com/Riya-1305/agrivision-app" target="_blank">GitHub Repo</a>
      <a href="#">MobileNetV2</a>
      <a href="#">PlantVillage Dataset</a>
    </div>
    <div class="footer-col">
      <h4>About</h4>
      <a href="#">Built for Kisan Udyog</a>
      <a href="#">Model Accuracy</a>
      <a href="#">Disclaimer</a>
    </div>
  </div>
  <div class="footer-bottom">
    <span>AgriVision · MobileNetV2 · For advisory purposes only — consult a local agronomist for confirmation.</span>
    <span>Built with ❤️ for Indian farmers</span>
  </div>
</div>
""")