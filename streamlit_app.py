"""
LLM Output Bias Detection - Streamlit Demo Application

This application demonstrates bias detection in LLM-generated text using
a LoRA fine-tuned MiniLM model trained on the StereoSet dataset.

The model classifies context-sentence pairs into:
- Stereotype: Reinforces common stereotypes
- Anti-stereotype: Challenges common stereotypes
- Unrelated: Not meaningfully related to the context
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.predictor import BiasPredictor
from src.preprocess import LABEL_NAMES


# Page configuration
st.set_page_config(
    page_title="LLM Bias Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .card-title {
        color: #1a1a2e;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Result styling */
    .result-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stereotype { color: #e74c3c; }
    .anti-stereotype { color: #27ae60; }
    .unrelated { color: #95a5a6; }
    
    .confidence-bar {
        height: 12px;
        background: #e0e0e0;
        border-radius: 6px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease-out;
    }
    
    .confidence-text {
        color: #666;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
        border-left: 4px solid #17a2b8;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .info-box p {
        color: #0c5460;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the bias predictor model (cached)."""
    return BiasPredictor()


def get_label_color(label: str) -> str:
    """Get the CSS color class for a label."""
    return label.replace("-", "-")


def get_label_emoji(label: str) -> str:
    """Get an emoji for each label."""
    emojis = {
        "stereotype": "⚠️",
        "anti-stereotype": "✅",
        "unrelated": "➖",
    }
    return emojis.get(label, "❓")


def get_label_description(label: str) -> str:
    """Get a description for each label."""
    descriptions = {
        "stereotype": "This response reinforces a common stereotype about the target group.",
        "anti-stereotype": "This response challenges or contradicts a common stereotype.",
        "unrelated": "This response is not meaningfully related to the context.",
    }
    return descriptions.get(label, "")


def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🔍 LLM Bias Detector</h1>
            <p class="header-subtitle">Detect stereotypical bias in AI-generated text</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            <p><strong>How it works:</strong> Enter a context sentence and a candidate response. 
            The model will classify whether the response contains stereotypical bias, 
            challenges stereotypes, or is unrelated to the context.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Please ensure the model files are present in the `models/lora_minilm/` directory.")
        return
    
    # Input section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">📝 Input</p>', unsafe_allow_html=True)
    
    context = st.text_area(
        "Context Sentence",
        placeholder="Enter a context that establishes a scenario...\n\nExample: The software engineer was working on a new project.",
        height=100,
        key="context_input",
    )
    
    sentence = st.text_area(
        "Candidate Response",
        placeholder="Enter a sentence to analyze for bias...\n\nExample: He stayed up all night writing code.",
        height=100,
        key="sentence_input",
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("🔬 Analyze Bias", use_container_width=True)
    
    # Results section
    if analyze_clicked:
        if not context.strip() or not sentence.strip():
            st.warning("⚠️ Please enter both a context sentence and a candidate response.")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict(context.strip(), sentence.strip())
            
            label = result["label"]
            confidence = result["confidence"]
            
            # Determine colors
            if label == "stereotype":
                bar_color = "#e74c3c"
                bg_gradient = "linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%)"
            elif label == "anti-stereotype":
                bar_color = "#27ae60"
                bg_gradient = "linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%)"
            else:
                bar_color = "#95a5a6"
                bg_gradient = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)"
            
            st.markdown(f"""
                <div class="card" style="background: {bg_gradient};">
                    <p class="card-title">🎯 Result</p>
                    <div style="text-align: center; padding: 1rem 0;">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{get_label_emoji(label)}</div>
                        <div class="result-label {label.replace('-', '-')}" style="color: {bar_color};">
                            {label.replace("-", " ").title()}
                        </div>
                        <p style="color: #666; margin-top: 0.5rem; font-size: 0.95rem;">
                            {get_label_description(label)}
                        </p>
                        <div style="margin-top: 1.5rem;">
                            <p style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Confidence</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence * 100}%; background: {bar_color};"></div>
                            </div>
                            <p class="confidence-text"><strong>{confidence:.1%}</strong></p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>Powered by LoRA fine-tuned MiniLM • Trained on StereoSet</p>
            <p style="font-size: 0.8rem; color: #aaa;">Built for responsible AI development</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
