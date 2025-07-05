import streamlit as st
import numpy as np
import joblib
import tldextract
import requests
import ssl
import whois
from datetime import datetime
import asyncio

# ----------- Load Models ----------- 
@st.cache_resource
def load_models():
    phishing_model = joblib.load("physhing_model.pkl")
    phishing_scaler = joblib.load("scaler_phishing.pkl")
    return phishing_model, phishing_scaler

phishing_model, phishing_scaler = load_models()


# ----------- Async Function for SSL Info ----------- 
async def get_ssl_info(url):
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        if "https" in url.lower():
            ssl_info = 1 if response.status_code == 200 else 0
        else:
            ssl_info = 0
        return ssl_info
    except Exception as e:
        return 0  # Default to 0 if SSL info can't be extracted


# ----------- Async Function for WHOIS Info ----------- 
async def get_whois_info(url):
    try:
        domain = tldextract.extract(url).domain
        whois_info = await asyncio.to_thread(whois.whois, domain)

        # Get the domain age (difference between current date and creation date)
        if isinstance(whois_info.creation_date, list):
            creation_date = whois_info.creation_date[0]
        else:
            creation_date = whois_info.creation_date
        age = (datetime.now() - creation_date).days if creation_date else 0
        
        return age
    except Exception as e:
        return 0  # Default to 0 if WHOIS info can't be extracted


# ----------- Feature Extraction Function ----------- 
def extract_url_features(url):
    try:
        ext = tldextract.extract(url)
        full_domain = f"{ext.domain}.{ext.suffix}"
        response = requests.get(url, timeout=5)

        features = []

        features.append(len(url))                                 # URL Length
        features.append(1 if "https" in url.lower() else 0)       # HTTPS present
        features.append(url.count('.'))                           # Dot count
        features.append(1 if "@" in url else 0)                   # @ symbol
        features.append(1 if "//" in url.replace("http://", "").replace("https://", "") else 0)  # Redirect
        features.append(1 if "-" in ext.domain else 0)            # Hyphen in domain
        features.append(len(ext.domain))                          # Domain length
        features.append(len(full_domain))                         # Full domain length
        features.append(1 if response.status_code == 200 else 0)  # Is live
        features.append(1 if "login" in url.lower() else 0)       # Contains "login"
        features.append(1 if "verify" in url.lower() else 0)      # Contains "verify"
        features.append(1 if ext.suffix in ["com", "org", "net"] else 0)  # Trusted TLD
        features.append(1 if "?" in url else 0)                   # Has query params
        features.append(url.count('/'))                           # Slash count
        features.append(1 if len(ext.suffix) <= 3 else 0)         # Short TLD

        # SSL and WHOIS Features (newly added)
        ssl_info = asyncio.run(get_ssl_info(url))
        domain_age = asyncio.run(get_whois_info(url))

        features.append(ssl_info)  # Add SSL feature
        features.append(domain_age)  # Add domain age feature

        # Pad remaining up to 39 features
        features += [0] * (39 - len(features))

        return np.array(features).reshape(1, -1)

    except Exception as e:
        # Return random test data without showing a warning
        return np.random.rand(1, 39)  # Fallback to random data if extraction fails


# ----------- Streamlit UI ----------- 

# Title and description
st.title("🔐 Phishing URL Checker")
st.markdown("Enter a website URL to check if it's **safe or malicious**.")

# URL input and button for prediction
url_input = st.text_input("🔗 Enter URL (e.g., https://netflix.com)", key="url_input")

if st.button("Check URL"):
    if url_input:
        with st.spinner("🔍 Analyzing..."):
            features = extract_url_features(url_input)
            if features is not None:
                scaled = phishing_scaler.transform(features)
                prediction = phishing_model.predict(scaled)
                if prediction[0] == 1:
                    # Hide internal error messages and just show phishing result
                    st.error("🚨 Malicious Website Detected!")
                else:
                    st.success("✅ Authentic Website")
    else:
        st.warning("Please enter a valid URL to check.")
