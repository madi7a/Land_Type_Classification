# Land Type Classification using Deep Learning (DenseNet & ResNet)

This project focuses on classifying satellite images into different land cover types using two powerful CNN architectures: **DenseNet121** and **ResNet50**. The goal is to build a robust image classification model to distinguish between 10 classes of land types.

### 🗂️ Dataset
The dataset consists of RGB satellite images categorized into the following classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Each image is preprocessed and resized for training with transfer learning models.

---

## 🧠 Models Used

### ✅ DenseNet121
- Pretrained on ImageNet
- Fine-tuned with:
  - GlobalAveragePooling
  - Fully Connected Dense Layer (Softmax)
- EarlyStopping and ModelCheckpoint for best model selection

**📈 Performance (DenseNet):**
| Class                  | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| AnnualCrop             | 0.98      | 0.93   | 0.95     |
| Forest                 | 0.93      | 0.99   | 0.96     |
| HerbaceousVegetation   | 0.93      | 0.93   | 0.93     |
| Highway                | 0.88      | 0.93   | 0.91     |
| Industrial             | 0.95      | 0.92   | 0.94     |
| Pasture                | 0.95      | 0.93   | 0.94     |
| PermanentCrop          | 0.92      | 0.90   | 0.91     |
| Residential            | 0.91      | 1.00   | 0.95     |
| River                  | 0.95      | 0.87   | 0.91     |
| SeaLake                | 1.00      | 0.97   | 0.98     |

- **Overall Accuracy:** **94%**

---

### ✅ ResNet50
- Pretrained on ImageNet
- Architecture:
  - GlobalAveragePooling
  - Dense, Dropout, BatchNormalization
- Used EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau

**📊 Performance (based on training/validation plots & confusion matrix):**
- Consistent training and validation accuracy
- Visualizations include:
  - Accuracy & Loss plots
  - Confusion Matrix
  - ROC-AUC per class

*(Exact accuracy not stated, but visually comparable to DenseNet.)*

---

## 🔍 Comparison

| Metric            | DenseNet121 | ResNet50    |
|-------------------|-------------|-------------|
| Accuracy          | **94%**     | ~94% (visual)|
| F1-Score (avg)    | **0.94**    | N/A (similar)|
| Complexity        | Medium      | Medium      |
| Training Time     | Slightly Faster | Slightly Slower |

Both models perform very similarly, though DenseNet edges out slightly in reported metrics.

---

## 🚀 Streamlit App

This project includes a **Streamlit app** that allows users to upload satellite images and get predictions on land cover types instantly.

🔗 **Live Demo:** [landtypeclassification-by-mad.streamlit.app](https://landtypeclassification-by-mad.streamlit.app/)

### 🔧 To Run Locally:

```bash
git clone https://github.com/madi7a/Land_Type_Classification.git
cd Land_Type_Classification
pip install -r requirements.txt
streamlit run app.py
