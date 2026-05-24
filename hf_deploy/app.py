import streamlit as st
import requests
from PIL import Image
import io
import os
from datetime import datetime

# ============================================
# Hugging Face Spaces Configuration
# ============================================

IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

if IS_HF_SPACE:
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
    os.environ["TORCH_HOME"] = "/tmp/torch_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)
    os.makedirs("/tmp/torch_cache", exist_ok=True)
    
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
# Page Config
# ============================================

st.set_page_config(
    page_title="EnviroAudit",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 EnviroAudit")
st.caption("AI-Powered Environmental Compliance Monitoring")

# ============================================
# Load Models
# ============================================

try:
    clip_model, clip_processor, blip_model, blip_processor = load_models()
    st.success("✅ Models ready!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.header("📋 Settings")
    project_id = st.text_input("Project ID", value=f"PROJ-{datetime.now().strftime('%Y%m%d')}")
    
    analysis_type = st.radio("Input Method:", ["Upload Image", "Image URL", "Sample"])
    
    if st.button("🔄 Clear"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by Hugging Face Transformers")

# ============================================
# Main
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
        url = st.text_input("Image URL", value="https://images.unsplash.com/photo-1581094794329-c8112a89af12")
        if st.button("Load") and url:
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")
    
    else:  # Sample
        samples = {
            "🏗️ Construction Site": "https://images.unsplash.com/photo-1581094794329-c8112a89af12",
            "⛏️ Mining Operation": "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09",
            "🌳 Natural Landscape": "https://images.unsplash.com/photo-1501854140801-50d01698950b"
        }
        choice = st.selectbox("Select sample:", list(samples.keys()))
        if st.button("Load Sample"):
            try:
                response = requests.get(samples[choice], timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

with col2:
    st.subheader("🔍 Results")
    
    if image is not None:
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing... (10-20 seconds)"):
                try:
                    # Resize for speed
                    img = image.copy()
                    img.thumbnail((512, 512))
                    
                    # Classification
                    labels = [
                        "construction site with heavy machinery",
                        "mining or quarry operation",
                        "land clearing or deforestation",
                        "natural landscape with no construction",
                        "agricultural field"
                    ]
                    
                    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]
                    
                    best_idx = probs.argmax().item()
                    primary_label = labels[best_idx]
                    confidence = f"{probs[best_idx].item():.1%}"
                    
                    # Caption
                    caption_inputs = blip_processor(img, return_tensors="pt")
                    out = blip_model.generate(**caption_inputs, max_length=50)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    
                    # Risk
                    high_risk = ["construction site with heavy machinery", "mining or quarry operation", "land clearing or deforestation"]
                    if primary_label in high_risk:
                        risk = "HIGH ⚠️"
                        action = "Schedule inspection within 48 hours"
                    else:
                        risk = "LOW ✅"
                        action = "Routine monitoring only"
                    
                    # Display
                    st.metric("Primary Classification", primary_label)
                    st.metric("Confidence", confidence)
                    st.info(f"📝 **Caption:** {caption}")
                    
                    if risk == "HIGH ⚠️":
                        st.warning(f"⚠️ **Risk Level: {risk}**\n\n**Action:** {action}")
                    else:
                        st.success(f"✅ **Risk Level: {risk}**\n\n**Action:** {action}")
                    
                    # Store in session
                    st.session_state.last_result = {
                        "label": primary_label,
                        "confidence": confidence,
                        "caption": caption,
                        "risk": risk,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    # Show previous result
    if st.session_state.get("last_result"):
        with st.expander("📋 Previous Result"):
            st.json(st.session_state.last_result)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.caption("EnviroAudit | Hugging Face Transformers (CLIP + BLIP)")