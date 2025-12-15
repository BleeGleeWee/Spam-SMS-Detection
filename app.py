import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ----------------- Page Config -----------------
st.set_page_config(page_title="Spam Classifier", layout="centered")
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

# ----------------- Session State Initialization -----------------
# We use a specific key 'input_sms_text' for the widget to ensure 2-way binding
if "input_sms_text" not in st.session_state:
    st.session_state.input_sms_text = ""

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "clear_message" not in st.session_state:
    st.session_state.clear_message = None

# ----------------- CALLBACK FUNCTIONS (The Fix) -----------------
def handle_clear():
    """
    This function runs immediately when Clear is clicked,
    BEFORE the page redraws. This prevents the API Exception.
    """
    # 1. Clear the text widget state
    st.session_state.input_sms_text = ""
    # 2. Set the success message
    st.session_state.clear_message = (
        "🧹 Input text has been completely cleared.\n\n"
        "You can now safely enter a new Email or SMS message."
    )

# ----------------- THEME LOGIC -----------------
# Check URL query parameter to trigger theme switch
query_params = st.query_params
if "theme" in query_params and st.session_state.theme != query_params["theme"]:
    st.session_state.theme = query_params["theme"]
    st.rerun()

current_theme = st.session_state.theme
is_dark = current_theme == "dark"


# ----------------- Text Processing -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words("english")]
    text = [ps.stem(i) for i in text]
    return " ".join(text)


# ----------------- Load Model -----------------
try:
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("🚨 Error: Model files not found. Please check 'vectorizer.pkl' and 'model.pkl'.")
    st.stop()

# ----------------- BACKGROUNDS (VIVID GRADIENTS) -----------------

# DARK MODE: Deep Blue -> Purple -> Black
dark_bg_css = """
background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
background-size: 400% 400%;
animation: gradientBG 15s ease infinite;
"""

# LIGHT MODE: Bright Pink -> Yellow -> Orange (Vivid)
light_bg_css = """
background: linear-gradient(135deg, #ff9a9e, #fecfef, #f6d365, #fda085);
background-size: 400% 400%;
animation: gradientBG 15s ease infinite;
"""

# Select CSS based on current theme state
selected_bg = dark_bg_css if is_dark else light_bg_css

# ----------------- CSS INJECTION -----------------
st.markdown(f"""
<style>
/* Global Reset */
* {{ user-select: none; }}

/* APP BACKGROUND */
.stApp {{
    {selected_bg}
}}

@keyframes gradientBG {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Main Container - Glassmorphism */
.block-container {{
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 3rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
}}

/* Text Colors */
h1, h2, h3, h4, h5, h6, label, .stMarkdown p {{ color: white !important; }}

/* Text Area Styling */
textarea {{
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #333 !important;
    border-radius: 10px;
}}
textarea::placeholder {{ color: #666 !important; }}

/* Button Styling */
button {{
    border-radius: 12px !important;
    height: 3em !important;
    transition: transform 0.2s;
}}
button:hover {{ transform: scale(1.05); }}

/* Footer Hidden */
footer {{ visibility: hidden; }}

/* ---- TOGGLE SWITCH CSS ---- */
.toggle-wrapper {{
    position: fixed;
    top: 50px;
    right: 20px;
    z-index: 1000;
}}

.sun-moon-label {{
    display: block;
    width: 100px;
    height: 50px;
    background-color: {"#000" if is_dark else "#87CEEB"}; /* Toggle Track Color */
    border-radius: 50px;
    cursor: pointer;
    position: relative;
    transition: 0.5s;
    box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
}}

#star {{
    position: absolute;
    top: 10px;
    left: {"55px" if is_dark else "10px"}; /* Position moves based on theme */
    width: 30px;
    height: 30px;
    background: #FFD700;
    border-radius: 50%;
    transition: 0.5s;
    box-shadow: 0 0 10px #FFD700;
}}

#moon-spot {{
    position: absolute;
    top: -5px;
    left: -10px;
    width: 30px;
    height: 30px;
    background: {"#000" if is_dark else "transparent"}; /* Shadow creates moon shape */
    border-radius: 50%;
    transition: 0.5s;
}}

</style>
""", unsafe_allow_html=True)

# ----------------- TOGGLE BUTTON HTML -----------------
# Clicking this link updates the URL parameter 'theme', forcing a page rerun
toggle_link = f"?theme={'light' if is_dark else 'dark'}"
st.markdown(f"""
<div class="toggle-wrapper">
    <a href="{toggle_link}" target="_self">
        <label class="sun-moon-label">
            <div id="star">
                <div id="moon-spot"></div>
            </div>
        </label>
    </a>
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN UI -----------------
st.title("📧 Email / SMS Spam Classifier")

# 1. SUCCESS MESSAGE LOGIC
if st.session_state.clear_message:
    st.success(st.session_state.clear_message)
    st.session_state.clear_message = None

# 2. INPUT AREA (Fixed Binding)
# key="input_sms_text" binds this widget directly to st.session_state.input_sms_text
input_sms = st.text_area(
    "Enter the message",
    placeholder="Example: Congratulations! You have won a free prize...",
    key="input_sms_text",
    height=150
)

# 3. BUTTONS
col1, col2, col3 = st.columns(3)
predict = col1.button("🚀 Predict")
analyze = col2.button("📈 Analyze")
# FIXED CLEAR BUTTON: specific on_click callback
clear = col3.button("🧹 Clear", on_click=handle_clear)
# ----------------- LOGIC -----------------

if predict:
    st.session_state.clear_message = None
    if not input_sms.strip():
        st.warning("⚠️ Please type a message first.")
    else:
        # Preprocess & Predict
        transformed_sms = transform_text(input_sms)
        vector = tfidf.transform([transformed_sms])
        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        if result == 1:
            st.error(f"🚨 **Spam Detected** (Confidence: {prob * 100:.1f}%)")
        else:
            st.success(f"✅ **Not Spam** (Confidence: {(1 - prob) * 100:.1f}%)")

if analyze:
    st.session_state.clear_message = None
    if not input_sms.strip():
        st.warning("⚠️ Please type a message first.")
    else:
        # Dummy ROC for single input illustration
        y_true = [0, 1]
        probs = model.predict_proba(tfidf.transform([transform_text(input_sms)] * 2))[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.info("ℹ️ Note: ROC curve is illustrative for single input.")

if clear:
    # Logic to clear the text area
    if not input_sms.strip():
        st.warning("⚠️ Field is already empty.")
    else:
        # 1. Clear the specific session state key bound to the text area
        st.session_state.input_sms_text = ""

        # 2. Set the persistent success message
        st.session_state.clear_message = (
            "🧹 Input text has been completely cleared.\n\n"
            "You can now safely enter a new Email or SMS message."
        )

        # 3. Force Rerun to update the UI immediately
        st.rerun()

st.markdown("---")
st.caption("🔓 Free public Streamlit Cloud deployment ready")