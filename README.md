# 🧬 PrediLnc

**PrediLnc** is a standalone software and web-based platform designed to predict the top **lncRNA–disease associations** using comprehensive biological inputs. Users can input either a **disease** or an **lncRNA**, along with associated data such as sequences or targets, and obtain the most relevant associations predicted by our machine learning model **GARNet (Graph convolution Attention RNA Network)**. The platform is designed to support discovery and hypothesis generation in lncRNA-related disease research.

---

## 🧠 Abstract

Long non-coding RNAs (lncRNAs) are RNA molecules longer than 200 nucleotides that do not encode proteins but play essential roles in gene regulation, chromatin remodeling, and other vital biological processes. Their dysregulation is associated with numerous diseases, including cancer, neurodegeneration, and cardiovascular conditions. Understanding lncRNA–disease associations is crucial for identifying novel diagnostic biomarkers and therapeutic targets.

We present **PrediLnc**, a predictive platform built on our model **GARNet**, which integrates multi-modal biological data and modern machine learning components to generate accurate predictions. GARNet includes:

- **Autoencoders** for dimensionality reduction of biological features.
- **Graph Convolutional Networks (GCNs)** to model topological relationships between biomolecules.
- **Self-attention mechanisms** to capture contextual dependencies.
- **Stacked ensemble learning framework** composed of:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - A meta-classifier (Random Forest) for final prediction.

GARNet has been validated through stratified cross-validation and ten real-world case studies. Predictions are supported by evidence from more than 500 peer-reviewed PubMed articles. PrediLnc is available both as a live web service and a locally deployable tool.

---

## 🌐 Live Demo

Access the web application here: [http://predilnc.dhanjal-lab.iiitd.edu.in/](http://predilnc.dhanjal-lab.iiitd.edu.in/)

### 🔍 Features

- **Predict Diseases from a lncRNA:**
  - Choose a lncRNA from the provided list.
  - If not listed, input lncRNA sequence, associated diseases, and gene targets.
  - View the top predicted diseases with confidence scores.

- **Predict lncRNAs from a Disease:**
  - Select a disease from the list or input manually.
  - Provide associated genes and metadata if needed.
  - View the top predicted lncRNAs relevant to the disease.

- **Other Sections:**
  - **About the Features:** Describes the biological features used.
  - **Insights:** Workflow, applications, and methodology.
  - **Contribute:** Submit new data to enhance the platform.
  - **Contact:** Information about authors and contributors.

---

## 🚀 Step 1: Installation and Setup (Local Deployment)

This step includes all actions required to install and run PrediLnc locally.

### 🔧 Prerequisites

- Anaconda or Miniconda
- Python 3.10
- R version ≥ 4.1.2
- Redis 7.2.1
- Git
- Internet connection for downloading Zenodo datasets

### 📥 1.1 Clone the Repository

```bash
git clone https://github.com/Udit64/PrediLnc.git
cd PrediLnc
🐍 1.2 Install Anaconda or Miniconda
Download and install from:

Anaconda: https://www.anaconda.com/products/distribution

Miniconda: https://docs.conda.io/en/latest/miniconda.html

🧪 1.3 Create and Activate the Conda Environment
bash
Copy
Edit
conda create -n rtest python=3.10 r-base=4.1.2 rpy2 r-essentials -c conda-forge
conda activate rtest
This sets up an isolated environment with Python and R support.

📦 1.4 Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
This installs Flask, scikit-learn, pandas, numpy, and other required packages.

🔁 1.5 Install Redis 7.2.1
bash
Copy
Edit
git clone https://github.com/redis/redis.git
cd redis
git checkout 7.2.1
make -j
src/redis-server &
To check if Redis is running:

bash
Copy
Edit
redis-cli ping
# Output: PONG
📡 1.6 Download Trained Models and Preprocessed Data
bash
Copy
Edit
zenodo_get --access-token G46P8DtW8lfKSG0u7IUVmCMb4idEKAoDCBL1yHWuwUkKvnFuGPSNCIkCham2 15764921
This downloads a .zip archive containing:

Trained model files

Feature matrices

Processed datasets

🗂️ 1.7 Extract the Dataset
bash
Copy
Edit
unzip saved_dataset.zip -d app/
Ensure the following are present in the app/ directory:

models/

data/

feature_files/

Any supporting CSVs or JSON files

🚀 1.8 Launch the Flask Web Server
bash
Copy
Edit
cd app
python app.py
Once the server starts, open your browser and visit:

cpp
Copy
Edit
http://127.0.0.1:5000/
This loads the local instance of PrediLnc.

🔄 Workflow
The architecture of GARNet is illustrated below:



🧬 Workflow Summary
Extract biological features (sequences, ontologies, targets).

Apply autoencoders to reduce dimensionality.

Construct heterogeneous graphs from the data.

Generate node embeddings using GCN and attention.

Train five base classifiers.

Aggregate predictions via a meta-classifier.

Return top-ranked lncRNA–disease associations.

🗂 File Structure
php
Copy
Edit
PrediLnc/
│
├── app/                   # Main web app and backend logic
│   ├── app.py             # Flask entry point
│   ├── templates/         # HTML templates
│   ├── static/            # CSS and JS assets
│   ├── models/            # Trained classifiers
│   └── data/              # Feature files, metadata
│
├── redis/                 # Redis installation
├── saved_dataset.zip      # Download from Zenodo
├── requirements.txt       # Python dependencies
└── README.md              # This file
📄 License
This project is licensed under the MIT License. You may use, modify, and distribute the software with appropriate credit.

✨ Acknowledgments
Institution: Translational Biology Lab, IIIT-Delhi

Supervisors: Dr. Jaspreet Kaur Dhanjal, Dr. Dhvani Vora

Lead Developer: Udit Kumar

Supported By: ABIDE II Consortium, Zenodo, PubMed

📫 Contact
Submit issues at: https://github.com/Udit64/PrediLnc/issues

Author and contributor details available in the “Contact” section of the web interface

🔖 Citation
If you use PrediLnc in your research, please cite:

latex
Copy
Edit
@misc{predilnc2025,
  title={PrediLnc: Predicting lncRNA-Disease Associations Using Graph Convolution and Attention},
  author={Kumar, Udit and Dhanjal, Jaspreet Kaur and Vora, Dhvani},
  year={2025},
  howpublished={\url{http://predilnc.dhanjal-lab.iiitd.edu.in/}},
  note={Bioinformatics Software}
}
