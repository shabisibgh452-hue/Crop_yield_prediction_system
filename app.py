import streamlit as st
import pandas as pd
import joblib

from config import (
    BEST_MODEL_PATH,
    AREA_ENCODER_PATH,
    ITEM_ENCODER_PATH,
    CLEAN_DATA_PATH
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD FILES
# ==========================================

model = joblib.load(BEST_MODEL_PATH)
area_encoder = joblib.load(AREA_ENCODER_PATH)
item_encoder = joblib.load(ITEM_ENCODER_PATH)

df = pd.read_csv(CLEAN_DATA_PATH)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp{
    background:#0D1117;
}

.block-container{
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

[data-testid="stSidebar"]{
    background:#161B22;
}

div.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#16A34A;
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
}

div.stButton > button:hover{
    background:#22C55E;
    color:white;
}

[data-baseweb="select"]{
    background:#1F2937;
    border-radius:10px;
}

.result-card{
    background:#161B22;
    border-radius:15px;
    padding:25px;
    border:1px solid #22C55E;
    text-align:center;
}

.metric-card{
    background:#1F2937;
    padding:18px;
    border-radius:12px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌾 Crop Yield")

    st.markdown("---")

    st.write("### Project Information")

    st.success("Machine Learning Model")

    st.info("""
✔ Random Forest

✔ Streamlit Dashboard

✔ Crop Yield Prediction

✔ Dark Theme UI
""")

    st.markdown("---")

    st.write("### Developer")

    st.write("Nazish Safdar")

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;color:#22C55E;'>
🌾 Crop Yield Prediction Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;color:lightgray;font-size:18px;'>
Predict Crop Yield using Machine Learning
</p>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# DASHBOARD CARDS
# ==========================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
    <h3>🌍 Area</h3>
    <p>Select Country</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <h3>🌱 Crop</h3>
    <p>Select Crop</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <h3>🌡 Temp</h3>
    <p>Average</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
    <h3>🌧 Rainfall</h3>
    <p>Average</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
# ==========================================
# INPUT SECTION
# ==========================================

left, right = st.columns(2)

with left:

    area = st.selectbox(
        "🌍 Select Area",
        sorted(df["Area"].unique())
    )

    item = st.selectbox(
        "🌱 Select Crop",
        sorted(df["Item"].unique())
    )

    year = st.number_input(
        "📅 Year",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=int(df["Year"].max())
    )

with right:

    rainfall = st.slider(
        "🌧 Rainfall",
        float(df["average_rain_fall_mm_per_year"].min()),
        float(df["average_rain_fall_mm_per_year"].max()),
        float(df["average_rain_fall_mm_per_year"].mean())
    )

    pesticides = st.slider(
        "🧪 Pesticides",
        float(df["pesticides_tonnes"].min()),
        float(df["pesticides_tonnes"].max()),
        float(df["pesticides_tonnes"].mean())
    )

    temperature = st.slider(
        "🌡 Temperature",
        float(df["avg_temp"].min()),
        float(df["avg_temp"].max()),
        float(df["avg_temp"].mean())
    )

st.write("")

predict = st.button("predict crop yield")
# ==========================================
# PREDICTION
# ==========================================

if predict:

    area_encoded = area_encoder.transform([area])[0]
    item_encoded = item_encoder.transform([item])[0]

    input_data = pd.DataFrame({
        "Area": [area_encoded],
        "Item": [item_encoded],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temperature]
    })

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-card">
        <h2 style="color:#22C55E;">🌾 Predicted Crop Yield</h2>
        <h1 style="color:white;">{prediction:.2f} hg/ha</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    chart = pd.DataFrame({"Yield": [prediction]})
    st.line_chart(chart)

    st.success("Prediction Completed Successfully ✅")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown(
    "<center><span style='color:gray;'>Developed by <b>Nazish Safdar</b> | Crop Yield Prediction System</span></center>",
    unsafe_allow_html=True,
)