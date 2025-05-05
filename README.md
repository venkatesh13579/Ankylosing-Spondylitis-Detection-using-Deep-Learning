# 🩻 Ankylosing Spondylitis Detection using Deep Learning

This Streamlit-based web application allows users to upload spinal X-ray images and predict the presence of Ankylosing Spondylitis (AS) using a pre-trained MobileNetV2 model. The app also provides treatment, diet, workout, and mental health recommendations based on the result. It can generate a PDF report of the diagnosis for download.

---

## 🚀 Features

- Upload an X-ray image (`.jpg`, `.jpeg`, `.png`)
- Predict condition: `Normal` or `Unhealthy` (AS Detected)
- Display confidence score
- Provide medical and lifestyle recommendations
- Downloadable PDF report
- Sample image download for testing

---

## 🧠 Model

The deep learning model used is **MobileNetV2**, trained on a balanced dataset of spinal X-ray images. It predicts:
- `Normal`
- `Unhealthy`

Model file: `mobilenetv2_balanced_model.h5`

---

## 📁 Folder Structure

AS-Detection-App/
│
├── app.py # Main Streamlit application
├── mobilenetv2_balanced_model.h5 # Trained DL model
├── sample_xray.jpg # Sample test image
├── reports/ # Generated PDF reports (auto-created)
├── README.md # Project documentation (this file)
└── requirements.txt # Python dependencies


---

## ⚙️ Installation & Usage

1. **Clone the repository** (or download ZIP):
   ```bash
   git clone https://github.com/venkatesh13579/Ankylosing-Spondylitis-Detection-using-Deep-Learning.git
Install dependencies:
You can install dependencies using pip:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
Open your browser and visit: http://localhost:8501

🛠 Dependencies
Create a requirements.txt file with the following content:

nginx
Copy
Edit
streamlit
tensorflow
Pillow
fpdf
numpy
Install using:

bash
Copy
Edit
pip install -r requirements.txt
