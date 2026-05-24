import streamlit as st
import requests
from PIL import Image
import io
import os
from datetime import datetime

# ============================================
# Hugging Face Spaces Configuration
# ============================================

# Set cache directories
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_cache"

# Create directories if they don't exist
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/torch_cache", exist_ok=True)

# Optimize for CPU
import torch
torch.set_num_threads(2)

# ============================================
# Load Models with Caching
# ============================================

@st.cache_resource
def load_models():
    """Load AI models once and cache them"""
    from transformers import CLIPModel, CLIPProcessor
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    with st.spinner("🔄 Loading AI models (CLIP + BLIP)... First load takes 30-60 seconds"):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    return clip_model, clip_processor, blip_model, blip_processor

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="EnviroAudit - Environmental Compliance",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 EnviroAudit")
st.caption("AI-Powered Environmental Compliance Monitoring | Running on Hugging Face Spaces")

# ============================================
# Load Models
# ============================================

try:
    clip_model, clip_processor, blip_model, blip_processor = load_models()
    st.success("✅ AI Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.header("📋 Configuration")
    
    project_id = st.text_input("Project ID", value=f"PROJ-{datetime.now().strftime('%Y%m%d')}")
    
    analysis_type = st.radio(
        "Input Method:",
        ["Upload Image", "Image URL", "Sample Images"]
    )
    
    st.markdown("---")
    st.caption(f"Space: {os.environ.get('SPACE_ID', 'Local')}")
    st.caption("Models: CLIP + BLIP")
    
    if st.button("🔄 Clear Results"):
        st.session_state.clear()
        st.rerun()

# ============================================
# Main Content
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Input")
    
    image = None
    
    if analysis_type == "Upload Image":
        uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)
    
    elif analysis_type == "Image URL":
        url = st.text_input(
            "Image URL", 
            value="https://images.unsplash.com/photo-1581094794329-c8112a89af12"
        )
        if st.button("📥 Load Image") and url:
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load: {e}")
    
    else:  # Sample Images
        samples = {
            "🏗️ Construction Site": "https://images.unsplash.com/photo-1581094794329-c8112a89af12",
            "⛏️ Mining Operation": "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09",
            "🌳 Natural Landscape": "https://images.unsplash.com/photo-1501854140801-50d01698950b",
            "🏙️ Urban Development": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b"
        }
        
        selected = st.selectbox("Choose a sample:", list(samples.keys()))
        
        if st.button("📥 Load Sample"):
            try:
                response = requests.get(samples[selected], timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load: {e}")

with col2:
    st.subheader("🔍 Analysis Results")
    
    if image is not None:
        if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing image with AI... (10-20 seconds)"):
                try:
                    # Resize for faster processing
                    img = image.copy()
                    img.thumbnail((512, 512))
                    
                    # ===== Classification with CLIP =====
                    labels = [
                        "construction site with heavy machinery",
                        "mining or quarry operation", 
                        "land clearing or deforestation",
                        "natural landscape with no construction",
                        "agricultural field",
                        "urban area with buildings",
                        "water body or river"
                    ]
                    
                    inputs = clip_processor(
                        text=labels, 
                        images=img, 
                        return_tensors="pt", 
                        padding=True
                    )
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]
                    
                    best_idx = probs.argmax().item()
                    primary_label = labels[best_idx]
                    confidence = probs[best_idx].item()
                    
                    # ===== Caption with BLIP =====
                    caption_inputs = blip_processor(img, return_tensors="pt")
                    out = blip_model.generate(**caption_inputs, max_length=50)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    
                    # ===== Risk Assessment =====
                    high_risk_labels = [
                        "construction site with heavy machinery",
                        "mining or quarry operation",
                        "land clearing or deforestation"
                    ]
                    
                    if primary_label in high_risk_labels:
                        risk_level = "HIGH"
                        risk_color = "⚠️"
                        action = "Schedule inspection within 48 hours"
                        recommendation = "Immediate environmental review recommended"
                    elif primary_label == "urban area with buildings":
                        risk_level = "MEDIUM"
                        risk_color = "📋"
                        action = "Routine monitoring recommended"
                        recommendation = "Document site conditions for baseline"
                    else:
                        risk_level = "LOW"
                        risk_color = "✅"
                        action = "No immediate action required"
                        recommendation = "Continue standard monitoring"
                    
                    # ===== Display Results =====
                    st.metric("Primary Classification", primary_label)
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    st.info(f"📝 **Image Description:** {caption}")
                    
                    st.subheader("⚠️ Compliance Assessment")
                    
                    if risk_level == "HIGH":
                        st.warning(f"**Risk Level: {risk_level} {risk_color}**")
                        st.warning(f"**Action:** {action}")
                        st.warning(f"**Recommendation:** {recommendation}")
                    elif risk_level == "MEDIUM":
                        st.info(f"**Risk Level: {risk_level} {risk_color}**")
                        st.info(f"**Action:** {action}")
                        st.info(f"**Recommendation:** {recommendation}")
                    else:
                        st.success(f"**Risk Level: {risk_level} {risk_color}**")
                        st.success(f"**Action:** {action}")
                        st.success(f"**Recommendation:** {recommendation}")
                    
                    # ===== Save to session =====
                    st.session_state.last_analysis = {
                        "project_id": project_id,
                        "timestamp": datetime.now().isoformat(),
                        "primary_label": primary_label,
                        "confidence": confidence,
                        "caption": caption,
                        "risk_level": risk_level,
                        "action": action
                    }
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Show previous analysis
    if st.session_state.get("last_analysis"):
        with st.expander("📋 Last Analysis Details", expanded=False):
            st.json(st.session_state.last_analysis)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌍 <b>EnviroAudit</b> - AI-Powered Environmental Compliance Monitoring</p>
    <p>Powered by Hugging Face Transformers (CLIP + BLIP)</p>
</div>
""", unsafe_allow_html=True)