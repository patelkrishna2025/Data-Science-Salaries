# 💼 Data Science Job Salaries — Streamlit Dashboard

## 📁 Project Structure
```
ds_salary_dashboard/
├── app.py                              ← Main Streamlit app (5 tabs)
├── data_ss.csv                         ← Dataset (607 records)
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
├── Data_Science_Job_Salaries_Updated.ipynb  ← Updated notebook
├── pages/                              ← (for future multi-page expansion)
├── utils/                              ← Helper modules
├── assets/                             ← Static assets
└── models/                             ← Saved ML models
```

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dashboard Tabs

| Tab | Feature |
|-----|---------|
| 📈 Overview | KPI cards, salary distribution, experience/company charts |
| 🔬 Deep Analysis | Yearly trends, correlation heatmap, global salary map |
| 🤖 Salary Predictor | Random Forest ML model — predict your salary |
| 🖼️ CV Analysis | Upload image → Grayscale, Edge Detection, Filters |
| 💬 Chatbot | Rule-based Q&A about dataset insights |

## 🔧 Features
- **Sidebar filters**: Experience, Employment Type, Company Size, Job Type, Year
- **ML Predictor**: R² ~0.40+, feature importance chart
- **Computer Vision**: 9 OpenCV operations + pixel statistics
- **Interactive Charts**: All Plotly (hover, zoom, download)
