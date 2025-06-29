# 🧬 PrediLnc

**PrediLnc** is a standalone software and web-based platform for predicting top **lncRNA–disease associations** using advanced biological feature processing and machine learning. Given an input **lncRNA** or **disease**, along with associated sequences or target data, PrediLnc uses our custom model **GARNet (Graph convolution Attention RNA Network)** to return the most relevant predictions. It serves as a powerful tool for hypothesis generation in biomedical research.

---

## 🧠 Abstract

Long non-coding RNAs (lncRNAs), transcripts >200 nucleotides, regulate gene expression, chromatin remodeling, and other key cellular processes. Dysregulation of lncRNAs is linked to a wide range of diseases like cancer, neurodegenerative, and cardiovascular conditions. Understanding their associations with diseases can reveal novel diagnostic and therapeutic avenues.

We present **PrediLnc**, built on **GARNet**, a predictive model that integrates:

- **Autoencoders** for dimensionality reduction  
- **Graph Convolutional Networks (GCNs)** for biomolecular topology modeling  
- **Self-attention mechanisms** for contextual awareness  
- **Stacked ensemble learning** including:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - A Random Forest meta-classifier

GARNet was validated using stratified cross-validation and ten real-world case studies, with predictions supported by over 500 PubMed articles. The software is available both as a live demo and for local deployment.

---

## 🌐 Live Demo

👉 [Try the Web App](http://predilnc.dhanjal-lab.iiitd.edu.in/)

---

## 🔍 Key Features

### 🔬 Predict Diseases from a lncRNA
- Choose from the list or input custom sequence, associated diseases, and targets.
- View predicted disease associations with confidence scores.

### 🧾 Predict lncRNAs from a Disease
- Select a disease or input manually.
- Provide associated gene/target data.
- Receive top-ranked lncRNAs related to the condition.

### 📚 Additional Sections
- **About the Features** – Overview of biological inputs  
- **Insights** – Workflow, methods, applications  
- **Contribute** – Add new datasets  
- **Contact** – Authors and contributors

---

## 🚀 Installation and Setup (Local Deployment)

### 🔧 Prerequisites

- Anaconda or Miniconda  
- Python 3.10  
- R ≥ 4.1.2  
- Redis 7.2.1  
- Git  
- Internet (for Zenodo model download)

---

### 📥 Step-by-Step Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Udit64/PrediLnc.git
cd PrediLnc

# 2. Install Anaconda or Miniconda (manual step – download from official site)
#    Anaconda: https://www.anaconda.com/products/distribution
#    Miniconda: https://docs.conda.io/en/latest/miniconda.html

# 3. Create and activate a new conda environment with Python and R
conda create -n rtest python=3.10 r-base=4.1.2 rpy2 r-essentials -c conda-forge
conda activate rtest

# 4. Install required Python packages
pip install -r requirements.txt

# 5. Install Redis 7.2.1
git clone https://github.com/redis/redis.git
cd redis
git checkout 7.2.1
make -j
src/redis-server &       # Run Redis in the background

# To check Redis is running correctly
redis-cli ping            # Should return: PONG
cd ..                     # Return to PrediLnc root

# 6. Download pretrained models and data from Zenodo
zenodo_get --access-token G46P8DtW8lfKSG0u7IUVmCMb4idEKAoDCBL1yHWuwUkKvnFuGPSNCIkCham2 15764921

# 7. Extract dataset contents to app directory
unzip saved_dataset.zip -d app/

# Ensure app/ contains:
# - models/
# - data/
# - feature_files/
# - Additional CSV or JSON files as needed

# 8. Run the web application
cd app
python app.py

# Access the local server at: http://127.0.0.1:5000/
