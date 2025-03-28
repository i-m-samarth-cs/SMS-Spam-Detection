import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.stem import PorterStemmer

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #3498db;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9 !important;
        transform: scale(1.05);
    }
    .result-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .spam-result {
        color: #e74c3c;
        font-weight: bold;
    }
    .not-spam-result {
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize stemmer and load models
port_stemmer = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def clean_text(text):
    """Preprocess and clean input text"""
    text = word_tokenize(text)
    text = " ".join(text)
    text = [char for char in text if char not in string.punctuation]
    text = ''.join(text)
    text = [char for char in text if char not in re.findall(r"[0-9]", text)]
    text = ''.join(text)
    text = [word.lower() for word in text.split() if word.lower() not in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)

def main():
    # Title and Description
    st.title("🛡️ SMS Spam Shield")
    st.markdown("### Protect Your Inbox from Unwanted Messages")
    
    # Main container
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Input Section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            input_sms = st.text_input(
                "Enter SMS Message", 
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        
        with col2:
            predict_button = st.button('Analyze Message', use_container_width=True)
        
        # Result Section
        if predict_button:
            if input_sms == "":
                st.warning("Please enter a message to analyze!")
            else:
                # Preprocessing
                transform_text = clean_text(input_sms)
                
                # Vectorization and Prediction
                vector_input = tfidf.transform([transform_text])
                result = model.predict(vector_input)
                
                # Display Result
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                if result == 1:
                    st.markdown(
                        '<h2 class="spam-result">🚨 Spam Detected! 🚨</h2>', 
                        unsafe_allow_html=True
                    )
                    with st.expander("Why is this spam?"):
                        st.info(
                            "This message contains characteristics "
                            "typical of spam communications. "
                            "Be cautious before interacting with it."
                        )
                else:
                    st.markdown(
                        '<h2 class="not-spam-result">✅ Safe Message</h2>', 
                        unsafe_allow_html=True
                    )
                    with st.expander("Message Analysis"):
                        st.success(
                            "This message appears to be legitimate "
                            "and does not show typical spam indicators."
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Always be cautious with messages from unknown sources. "
        "This tool helps, but doesn't guarantee 100% spam detection."
    )

if __name__ == "__main__":
    main()