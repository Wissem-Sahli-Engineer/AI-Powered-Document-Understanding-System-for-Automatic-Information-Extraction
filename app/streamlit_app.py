"""
Streamlit Frontend for OACA Invoice Extraction
Provides UI to upload invoices and view extracted data
"""
import streamlit as st
import requests
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="OACA Invoice Extractor",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
    }
    .field-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.2rem;
    }
    .field-value {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .status-success {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
    }
    .status-error {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📄 OACA Invoice Extractor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except:
        return None

# Sidebar - API Status
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_url = st.text_input("API URL", value=API_URL)
    if api_url != API_URL:
        API_URL = api_url
    
    st.markdown("---")
    st.header("📊 API Status")
    
    if st.button("Check Status"):
        health = check_api_health()
        if health:
            st.success(f"✅ API: {health['status']}")
            if health['model_loaded']:
                st.success("✅ Model loaded")
            else:
                st.warning("⚠️ Model not loaded")
        else:
            st.error("❌ API not reachable")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Invoice")
    
    uploaded_file = st.file_uploader(
        "Choose an invoice image",
        type=["png", "jpg", "jpeg"],
        help="Upload a PNG, JPG, or JPEG image of an OACA invoice"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Invoice", use_container_width=True)

with col2:
    st.header("📋 Extracted Data")
    
    if uploaded_file:
        if st.button("🔍 Extract Data", type="primary", use_container_width=True):
            with st.spinner("Processing invoice..."):
                try:
                    # Reset file position
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/extract", files=files, timeout=60)
                    result = response.json()
                    
                    if result["success"]:
                        st.markdown('<div class="status-success">✅ Extraction Successful</div>', unsafe_allow_html=True)
                        st.markdown("")
                        
                        data = result["data"]
                        
                        # Display extracted fields
                        fields = [
                            ("🔢 Invoice Number", data.get("invoice_number")),
                            ("📅 Date", data.get("invoice_date")),
                            ("👤 Client", data.get("client_name")),
                            ("💰 Total Amount", data.get("total_amount")),
                            ("💱 Currency", data.get("currency")),
                        ]
                        
                        for label, value in fields:
                            if value:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="field-label">{label}</div>
                                    <div class="field-value">{value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info(f"{label}: Not detected")
                        
                        # Show raw entities in expander
                        with st.expander("🔬 View Raw Entities"):
                            st.json(result.get("raw_entities", {}))
                    
                    else:
                        st.markdown(f'<div class="status-error">❌ {result["message"]}</div>', unsafe_allow_html=True)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the API server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 Upload an invoice image to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Powered by LayoutLMv3 | OACA Invoice Extraction Pipeline
    </div>
    """,
    unsafe_allow_html=True
)
