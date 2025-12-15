import streamlit as st
import pickle
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import roc_curve, auc

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Spam Classifier",
    layout="centered"
)

nltk.download("punkt")
nltk.download("stopwords")
ps = PorterStemmer()

# ----------------- SESSION STATE -----------------
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ----------------- THEME FROM URL -----------------
query_params = st.query_params
if "theme" in query_params:
    st.session_state.theme = query_params["theme"][0]

current_theme = st.session_state.theme
is_dark = current_theme == "dark"

# ----------------- LOAD MODEL -----------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ----------------- TEXT PROCESSING -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words("english")]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ----------------- CSS -----------------
st.markdown(f"""
<style>
/* ---------- GLOBAL ---------- */
* {{ user-select: none; }}

@keyframes gradientBG {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

.stApp.light {{
  background: linear-gradient(-45deg, #ffecd2, #fcb69f, #a1c4fd, #c2e9fb);
  background-size: 400% 400%;
  animation: gradientBG 18s ease infinite;
}}

.stApp.dark {{
  background: linear-gradient(-45deg, #020111, #191621, #000428, #004e92);
  background-size: 400% 400%;
  animation: gradientBG 18s ease infinite;
}}

.block-container {{
  padding-top: 140px;
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(18px);
  border-radius: 22px;
  padding: 3rem;
  box-shadow: 0 25px 60px rgba(0,0,0,0.45);
}}

h1, h2, h3, label {{ color: white !important; }}

textarea {{
  background: white !important;
  color: black !important;
  border-radius: 14px;
}}

button {{
  border-radius: 14px !important;
  font-weight: 600;
}}

footer {{ visibility: hidden; }}

/* ---------- TOGGLE ---------- */
.toggle-wrapper {{
  position: fixed;
  top: 60px;
  right: 20px;
  z-index: 999;
}}

.sun-moon {{
  width: 116px;
  height: 56px;
  background: {"#000" if is_dark else "#77b5fe"};
  border-radius: 56px;
  position: relative;
  overflow: hidden;
  transition: 0.4s;
}}

#star {{
  position: absolute;
  top: {"3px" if is_dark else "13px"};
  left: {"64px" if is_dark else "13px"};
  width: 30px;
  height: 30px;
  background: yellow;
  border-radius: 50%;
  transform: scale({"0.3" if is_dark else "1"});
  transition: 0.4s;
}}

.star {{
  position: absolute;
  font-size: 54px;
  top: -12px;
  left: -7px;
  color: yellow;
}}

#moon {{
  position: absolute;
  bottom: {"8px" if is_dark else "-52px"};
  right: 8px;
  width: 40px;
  height: 40px;
  background: white;
  border-radius: 50%;
  transition: 0.4s;
}}

#moon:before {{
  content: "";
  position: absolute;
  top: -12px;
  left: -17px;
  width: 40px;
  height: 40px;
  background: {"#000" if is_dark else "#03a9f4"};
  border-radius: 50%;
}}
</style>
""", unsafe_allow_html=True)

# ----------------- APPLY THEME CLASS -----------------
st.markdown(
    f"""
    <script>
      document.querySelector('.stApp').classList.add('{current_theme}');
    </script>
    """,
    unsafe_allow_html=True
)

# ----------------- TOGGLE HTML -----------------
st.markdown(f"""
<div class="toggle-wrapper">
  <a href="?theme={'light' if is_dark else 'dark'}" style="text-decoration:none">
    <div class="sun-moon">
      <div id="star">
        <div class="star">★</div>
      </div>
      <div id="moon"></div>
    </div>
  </a>
</div>
""", unsafe_allow_html=True)

# ----------------- UI -----------------
st.title("📧 Email / SMS Spam Classifier")

input_sms = st.text_area(
    "Enter the message",
    value=st.session_state.input_sms,
    placeholder="Congratulations! You won a free prize..."
)

st.session_state.input_sms = input_sms

col1, col2, col3 = st.columns(3)
predict = col1.button("🚀 Predict")
analyze = col2.button("📈 Analyze")
clear = col3.button("🧹 Clear")

# ----------------- PREDICT -----------------
if predict and input_sms.strip():
    vector = tfidf.transform([transform_text(input_sms)])
    prob = model.predict_proba(vector)[0][1]
    result = model.predict(vector)[0]

    if result:
        st.error(f"🚨 Spam Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Not Spam (Confidence: {1 - prob:.2f})")

# ----------------- ROC -----------------
if analyze and input_sms.strip():
    y_true = [0, 1]
    probs = model.predict_proba(
        tfidf.transform([transform_text(input_sms)] * 2)
    )[:, 1]

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.legend()
    st.pyplot(fig)

# ----------------- CLEAR -----------------
if clear:
    st.session_state.input_sms = ""
    st.rerun()

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("🔓 Streamlit Cloud ready | Professional UI")


