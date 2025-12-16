# 📩 Spam SMS Detection ML Project
- Deployed link✨ 
[**Here**](https://spam-sms-detection-ml.streamlit.app/)


## 📌 Project Objective
The goal of this project is to build and deploy a machine learning model that can classify SMS messages as **Spam** or **Ham** (Not Spam).  
The model is trained using a labeled dataset and deployed for real-world testing.

---

## 🛠️ Tech Stack
- Python
- Scikit-Learn
- Pandas, Numpy
- Natural Language Processing (NLP)
- NLTK
- Streamlit (for deployment) 

---

## 📚 Dataset
- **Dataset Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: 5,500 SMS messages labeled as Spam or Not Spam.

---

## 📊 Project Stages
1. **Data Cleaning**
2. **Exploratory Data Analysis** (EDA)
3. **Text Preprocessing** (tokenization, stemming, etc.)
4. **Model Building** (Naive Bayes, Logistic Regression, etc.)
5. **Vectorization** (TF-IDF, GridSearchCV)
6. **Model Evaluation** (Accuracy, Precision, Recall, F1 Score)
7. **PyCharm App Development** (Over Streamlit)

---


## 📊 Model Performance
| Metric | Score |
|:------:|:-----:|
| Accuracy | 97.9% |
| Precision | 97.5% |
| Recall | 96% |


---


## ⚙️ Steps to Run the Project

### 1. Clone the repository:
   ```bash
   git clone https://github.com/BleeGleeWee/Spam-SMS-Detection.git
   cd Spam-SMS-Detection
   ```

### 2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook spam_sms_detection.ipynb
   ```

### 4. For deployed app:
   ```bash
   streamlit run app.py
   ```

---

# 🌟 FINAL SHOWDOWN:

<img width="1114" height="700" alt="image" src="https://github.com/user-attachments/assets/fa4bf003-d6e7-42ce-a12d-ec47679b5c2d" />


#

<img width="1119" height="670" alt="Screenshot 2025-12-16 035317" src="https://github.com/user-attachments/assets/b8917a91-8bf7-4fcd-ae33-0446f0be3462" />


---


```
Email/SMS-spam-classifier
│
├── data/
│   └── spam.csv                         # Original dataset (or link to download in README)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb           # Handling nulls, duplicates, formatting
│   ├── 02_eda.ipynb                     # Visualizations and exploratory analysis
│   ├── 03_text_preprocessing.ipynb      # Tokenization, stemming, stopword removal
│   ├── 04_model_building.ipynb          # Naive Bayes, Logistic Regression, etc.
│   └── 05_model_improvement.ipynb       # TF-IDF, hyperparameter tuning, evaluation
│
├── models/
│   ├── model.pkl                        # Serialized trained model (pickle)
|   └── vectorizer.pkl                   # Trained model then vectorized
│
├── app/
│   ├── app.py                           # App entry point
│   ├── main.py                  # Utility to load the model
│   └── train_model.py                   # Training model before testing   
│
├── .gitignore                           # Ignore notebooks checkpoints, model files, etc.
├── requirements.txt                     # All dependencies (Flask/FastAPI, sklearn, etc.)
├── nltk.txt                             # NLTK dependencies (stopwords, punkt)
├── README.md                            # Full documentation 
└── LICENSE                              # MIT or any preferred open-source license
```


---

