import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import load_migration_model, start_card, end_card, analyze_sentiment

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Wildlife Monitoring System", layout="wide")

# -------------------- BACKGROUND + FULL TEXT VISIBILITY --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(5,35,28,0.75), rgba(5,35,28,0.75)),
                url("https://images.unsplash.com/photo-1470115636492-6d2b56f9146d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}

html, body, [class*="css"]  {
    color: #F1F8E9 !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.2);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-weight: 600;
}

.hero {
    text-align:center;
    padding: 10px 0 20px 0;
    font-size: 36px;
    font-weight: 800;
    color:#FFFFFF;
    text-shadow: 2px 2px 14px rgba(0,0,0,0.95);
}

.metric-card {
    background: rgba(0,0,0,0.45);
    border-radius: 20px;
    padding: 25px;
    text-align:center;
    backdrop-filter: blur(18px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    height: 190px;
    width: 100%;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    color:#FFFFFF !important;
}

h1,h2,h3,h4{
color:#FFFFFF !important;
text-shadow:1px 1px 8px rgba(0,0,0,0.8);
}

.stButton>button{
border-radius:12px;
background:linear-gradient(135deg,#2E7D32,#1B5E20);
color:white !important;
border:none;
font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
migration_model = load_migration_model()

image_model=None
labels=[]

try:

    from tensorflow.keras.models import load_model
    import json

    if os.path.exists("model/image_model.h5") and os.path.exists("model/image_classes.json"):

        image_model=load_model("model/image_model.h5")

        with open("model/image_classes.json","r") as f:
            labels=json.load(f)

        model_source="Custom Trained"

    else:

        from tensorflow.keras.applications.resnet50 import ResNet50

        image_model=ResNet50(weights="imagenet")

        model_source="ImageNet ResNet50"

except Exception as e:

    st.warning(f"Image classifier unavailable: {e}")

    model_source=None


# -------------------- LOAD DATA --------------------
df=pd.read_csv("data/wildlife_data.csv")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🐾 Wildlife Intelligence")

section=st.sidebar.radio(
"Navigate",
["Overview","Analytics Dashboard","Prediction","Image Identifier","Ranger Reports","Alerts"]
)

# -------------------- HERO --------------------
st.markdown("<div class='hero'>🦁 Intelligent Multi-Modal Wildlife Monitoring System</div>", unsafe_allow_html=True)

# ==================== OVERVIEW ====================
if section=="Overview":

    st.subheader("📌 Wildlife Overview")

    col1,col2,col3,col4=st.columns(4)

    metrics=[
    ("🐘 Total Species",df["Species"].nunique()),
    ("🦁 Total Population",int(df["Population"].sum())),
    ("🐯 High Threat Species",len(df[df["Threat_Level"]>7])),
    ("🦌 Avg Threat Level",round(df["Threat_Level"].mean(),2))
    ]

    for col,(title,value) in zip([col1,col2,col3,col4],metrics):

        with col:

            st.markdown(f"""
            <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            </div>
            """,unsafe_allow_html=True)

# ==================== ANALYTICS ====================
elif section=="Analytics Dashboard":

    st.subheader("📊 Wildlife Analytics Dashboard")

    col1,col2=st.columns(2)

    with col1:

        start_card("📊 Population by Species")

        pop=df.groupby("Species")["Population"].sum().sort_values()

        fig,ax=plt.subplots(figsize=(7,5))

        ax.barh(pop.index,pop.values)

        ax.set_xlabel("Population")

        ax.set_ylabel("Species")

        ax.grid(axis='x',linestyle='--',alpha=0.3)

        st.pyplot(fig,width="stretch")

        end_card()

    with col2:

        start_card("⚠ Threat Level Distribution")

        fig2,ax2=plt.subplots(figsize=(7,5))

        ax2.hist(df["Threat_Level"],bins=8)

        ax2.set_xlabel("Threat Level")

        ax2.set_ylabel("Count")

        ax2.grid(axis='y',linestyle='--',alpha=0.3)

        st.pyplot(fig2,width="stretch")

        end_card()

# ==================== PREDICTION ====================
elif section=="Prediction":

    st.subheader("🧠 Migration Risk Prediction")

    population=st.slider(
    "Population",
    0,
    int(df["Population"].max()),
    int(df["Population"].mean())
    )

    threat=st.slider(
    "Threat Level",
    int(df["Threat_Level"].min()),
    int(df["Threat_Level"].max()),
    int(df["Threat_Level"].mean())
    )

    if st.button("Predict"):

        input_df=pd.DataFrame(
        [[population,threat]],
        columns=["Population","Threat_Level"]
        )

        pred=migration_model.predict(input_df)[0]

        score=float(pred)

        st.success(f"Predicted Migration Risk Score: {score:.2f}")

        preds_all=migration_model.predict(df[["Population","Threat_Level"]])

        low_th=np.percentile(preds_all,33)
        high_th=np.percentile(preds_all,66)

        if score<=low_th:
            category="Low"
        elif score<=high_th:
            category="Moderate"
        else:
            category="High"

        st.info(f"Risk Category: {category}")

# ==================== IMAGE IDENTIFIER ====================
elif section=="Image Identifier":

    st.subheader("📷 Animal Image Identifier")

    if image_model is None:

        st.error("Image classifier failed to load.")

    else:

        st.info(f"Model: {model_source}")

        uploaded_file=st.file_uploader("Upload Animal Image",type=["jpg","png","jpeg"])

        if uploaded_file is not None:

            image=Image.open(uploaded_file).convert("RGB")

            col1,col2=st.columns(2)

            with col1:

                start_card("🖼 Uploaded Image")

                st.image(image,width="stretch")

                end_card()

            with col2:

                start_card("🔍 Prediction Results")

                if model_source=="Custom Trained":

                    img=image.resize((64,64))

                    img_array=np.array(img).astype("float32")/255.0

                    img_array=np.expand_dims(img_array,axis=0)

                    preds=image_model.predict(img_array,verbose=0)[0]

                    idx=np.argmax(preds)

                    confidence=float(preds[idx])*100

                    top_label=labels[idx].replace('_',' ').title()

                    st.success(f"Detected: {top_label}")

                    st.progress(int(confidence))

                    st.write(f"{confidence:.2f}% Confidence")

                else:

                    from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions

                    img=image.resize((224,224))

                    img_array=np.array(img)

                    img_array=np.expand_dims(img_array,axis=0)

                    img_array=preprocess_input(img_array)

                    preds=image_model.predict(img_array)

                    decoded=decode_predictions(preds,top=5)[0]

                    top_label=decoded[0][1].replace("_"," ").title()

                    confidence=float(decoded[0][2])*100

                    st.success(f"Detected: {top_label}")

                    st.progress(int(confidence))

                    st.write(f"{confidence:.2f}% Confidence")

                    st.write("### Top Predictions")

                    for _,name,prob in decoded:

                        clean=name.replace("_"," ").title()

                        st.write(f"{clean} — {prob*100:.2f}%")

                end_card()

# ==================== REPORT ANALYSIS ====================
elif section=="Ranger Reports":

    st.subheader("📝 Ranger Risk Analysis")

    report=st.selectbox("Select Report",df["Ranger_Report"])

    risk=analyze_sentiment(report)

    if risk=="High Risk":

        st.error("🚨 High Risk Detected!")

    elif risk=="Moderate":

        st.warning("⚠ Moderate Risk")

    else:

        st.success("✅ Low Risk")

# ==================== ALERTS ====================
elif section=="Alerts":

    st.subheader("🚨 Conservation Alerts")

    high=df[df["Threat_Level"]>7]

    if len(high)>0:

        st.error("Immediate Action Required!")

        st.dataframe(high)

    else:

        st.success("All species safe.")