import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import time
from fpdf import FPDF
import datetime
import os
import base64

#PDF Generation Function
import os
from datetime import datetime


def sanitize_text(text):
    return text.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'")
def generate_pdf_report(patient_name, prediction, confidence, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Ankylosing Spondylitis Detection Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Diagnosis: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recommendations", ln=True)

    pdf.set_font("Arial", '', 12)

    if "treatment" in recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=" Treatment & Medication", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, sanitize_text(recommendations["treatment"]))

    if "diet" in recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=" Diet Recommendation", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, sanitize_text(recommendations["diet"]))

    if "workout" in recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=" Workouts to be Done", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, sanitize_text(recommendations["workout"]))

    if "moral_support" in recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=" Moral Support", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, sanitize_text(recommendations["moral_support"]))

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{patient_name.replace(' ', '_')}_report.pdf"
    output_path = os.path.join(output_dir, filename)
    pdf.output(output_path)

    return output_path

def get_binary_file_downloader_html(bin_file, label='Download file'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/jpg;base64,{b64}" download="sample_xray.jpg">{label}</a>'
    return href

#Load Model
model = load_model("mobilenetv2_balanced_model.h5")
class_names = ['Normal', 'Unhealthy', 'Unhealthy']  

#Recommendation Dictionary
Recommendations = {
    "Normal": {
        "treatment": "- No medical treatment needed.",
        "diet": "- Continue a balanced and healthy diet.",
        "workout": "- Regular exercise and posture-maintaining activities.",
        "moral_support": "- Maintain a healthy lifestyle and have routine checkups."
    },
    "Unhealthy": {
        "treatment": """
- **Medications:** NSAIDs (e.g., *Ibuprofen*, *Naproxen*) to reduce pain and inflammation. TNF inhibitors (e.g., *Etanercept*, *Infliximab*) or IL-17 inhibitors for advanced cases.  
- **Physical Therapy:** Regular physiotherapy focused on posture correction and mobility.  
- **Medical Support:** Routine checkups with a rheumatologist. Imaging and blood tests to monitor disease progression.  
- **Surgical Intervention:** In rare and severe cases, surgery might be required to correct spinal deformities.
        """,
        "diet": """
- **Anti-inflammatory Foods:** Fatty fish (*salmon*, *mackerel*), leafy greens (*spinach*, *kale*), turmeric, ginger.  
- **Calcium & Vitamin D:** Milk, cheese, yogurt, tofu, fortified cereals, and sunlight exposure.  
- **Fiber-Rich Foods:** Whole grains, fruits, and vegetables to improve gut health and immunity.  
- **Foods to Avoid:** Processed meats, refined sugars, excess caffeine, and alcohol.
        """,
        "workout": """
- **Postural Exercises:** Wall standing, chin tucks, and neck stretches to maintain spinal alignment.  
- **Stretching & Flexibility:** Yoga, Pilates, and spinal extension exercises.  
- **Low-Impact Aerobics:** Swimming, walking, cycling — promotes joint health without stress.  
- **Breathing Exercises:** Diaphragmatic breathing to improve lung capacity.
        """,
        "moral_support": """
- **Mental Health Awareness:** Join support groups (online or offline) to share experiences and stay motivated.  
- **Mindfulness & Relaxation:** Meditation, journaling, and stress management practices.  
- **Stay Connected:** Family and friends' emotional support plays a major role in recovery.  
- **Educational Empowerment:** Learn about AS to make informed decisions about your health.
        """
    }
}

#Streamlit Layout#

st.set_page_config(page_title="AS Detection", layout="centered")
st.title("🩻 Ankylosing Spondylitis Detection")
st.write("Upload an X-ray image (jpg, jpeg, png) to detect possible signs of Ankylosing Spondylitis.")

#Buttons for sample and reset

col1, col2 = st.columns(2)
with col1:
    if st.button("📥 Download Sample Image"):
        sample_path = "sample_xray.jpg" 
        if os.path.exists(sample_path):
            st.markdown(get_binary_file_downloader_html(sample_path, "Click here to download"), unsafe_allow_html=True)
        else:
            st.warning("Sample image not found. Please add 'sample_xray.jpg' in the project folder.")

with col2:
    if st.button("🔄 Clear All"):
        st.rerun()

#Image Upload Section
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image..."):
        time.sleep(1)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        confidence = prediction[0][class_index]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** `{predicted_class}`")
    st.write(f"**Confidence Score:** `{confidence:.2f}`")

    if predicted_class == "Normal":
        st.success("✅ The image appears to be normal.")
    else:
        st.error(f"⚠️ The image shows signs of `{predicted_class}`.")

#Recommendations
    st.markdown("---")
    st.subheader("📝 Recommendations")

    rec = Recommendations[predicted_class]

    with st.expander("🩺 Treatment & Medication Recommendation"):
        st.markdown(rec["treatment"])

    with st.expander("🥗 Diet Recommendation"):
        st.markdown(rec["diet"])

    with st.expander("🏃‍♂️ Workouts to be Done"):
        st.markdown(rec["workout"])

    with st.expander("💖 Moral Support"):
        st.markdown(rec["moral_support"])

#PDF Generation
    patient_name = st.text_input("Enter Patient Name for Report", "John Doe")

    if st.button("📄 Generate PDF Report"):
        report_path = generate_pdf_report(patient_name, predicted_class, confidence, rec)
        st.success("✅ Report generated successfully!")

        with open(report_path, "rb") as file:
            st.download_button(
                label="📥 Download PDF Report",
                data=file,
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )

