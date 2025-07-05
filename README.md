# 🩻 Chest X-ray Classifier

A web application built with Streamlit and TensorFlow to classify chest X-ray images into three categories: **Covid-19**, **Normal**, and **Pneumonia**. The app features a multi-language interface (English and Arabic) and provides detailed model performance insights.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

## ✨ Features

*   **🖼️ Image Upload:** Easily upload chest X-ray images (`.jpg`, `.jpeg`, `.png`).
*   **🤖 AI-Powered Prediction:** Uses a deep learning model (MobileNetV2) to predict the condition.
*   **📊 Interactive Results:**
    *   Displays the final prediction and confidence score.
    *   Visualizes class probabilities with an interactive Altair bar chart.
*   **🌐 Multi-Language Support:** Seamlessly switch between English and Arabic.
*   **📘 Detailed Model Info:** An "About" page provides insights into the model's architecture, training process, and performance metrics, including a classification report and confusion matrix.
*   **🎨 Custom UI:** A sleek, modern interface with custom CSS animations and styling.

## 🛠️ Technology Stack

*   **Backend & ML:** Python, TensorFlow, Keras
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas, NumPy
*   **Data Visualization:** Altair, Matplotlib, Seaborn
*   **Image Processing:** Pillow

## 🧠 Model Details

The classification model is built using transfer learning on the **MobileNetV2** architecture, a lightweight and efficient model perfect for this task.

*   **Dataset:** The model was trained on the COVID-19 Image Dataset from Kaggle.
    *   **Training Set:** 251 images (111 Covid-19, 70 Normal, 70 Pneumonia)
    *   **Testing Set:** 66 images (26 Covid-19, 20 Normal, 20 Pneumonia)
*   **Training Process:**
    *   The training data was split into 75% for training and 25% for validation.
    *   Data augmentation techniques (rotation, zoom, shift, flip) were applied to create a more robust model.
    *   Class weights were used to handle the dataset imbalance.
    *   The model was trained for 30 epochs using the Adam optimizer and `categorical_crossentropy` loss function.
    *   Callbacks like `EarlyStopping` and `ReduceLROnPlateau` were used for efficient training.
*   **Performance:** The final model achieved a **test accuracy of 96.5%**.

## 🚀 Getting Started

To run this application on your local machine, follow these steps.

### Prerequisites

*   Python 3.8+
*   `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/covid-xray-classifier.git
    cd covid-xray-classifier
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    (A `requirements.txt` file is provided in the repository)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the model file:**
    Ensure you have the model file `covid_xray_classifier_mobilenetv2.h5` in the root directory of the project.

### Running the App

Once the dependencies are installed, run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📁 Project Structure

```
.
├── covid_xray_classifier_mobilenetv2.h5  # Pre-trained Keras model
├── app.py                                # Main Streamlit application script
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

## 👤 Author

*   **Mohamed Mostafa Hassan**