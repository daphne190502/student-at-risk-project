# 🎓 Student At-Risk Prediction System

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-1D9E75?style=for-the-badge&logo=streamlit&logoColor=white)](https://student-at-risk-project-hqzaznsnhtvna7vfeaabgm.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)

---

## 🔗 Live App

👉 [Click here to open the live deployed application](https://student-at-risk-project-hqzaznsnhtvna7vfeaabgm.streamlit.app)

---

## 📌 What Is This Project?

In further education, every student withdrawal is a student who did not get
the qualification they came for — and a funding loss the college cannot recover.

Most colleges only find out a student is struggling after they have already left.

This project builds a complete end-to-end system that identifies students at
risk of withdrawal before they disengage — using only information available
on Day 1 of term — and gives tutors the information they need to intervene in time.

---

## 📊 Key Numbers

- 15,000 students across 8 departments and 8 courses
- 3,558 students withdrew — a 23.7% withdrawal rate
- 3,276 students flagged as High Risk (21.8%)
- 3,332 students flagged as Medium Risk (22.2%)
- 8,392 students flagged as Low Risk (55.9%)
- Model selected — Random Forest — 95.5% Recall
- Average attendance — 71.3%
- Average assessment score — 55.1%

---

## 🛠️ What I Built — Step by Step

---

### Step 1 — Data Generation

- Generated a fully synthetic dataset of 15,000 student records using Python
- Data covers 8 departments: Technology, Engineering, Business, Trades,
  Health, Science, Education, Creative
- Data covers 8 courses including Computing & IT, Health & Social Care,
  Business Studies, and Digital Media
- Includes student demographics, course details, attendance, assessments,
  VLE activity, and support session records

---

### Step 2 — SQL Database

- Built a SQLite relational database with 6 connected tables:
  students, courses, assessments, attendance, vle_activity, support_sessions
- Loaded 15,000 student records across all tables
- Wrote JOIN queries to build a master table combining all data sources
- Used SQL to answer real business questions:
  - Which course has the highest withdrawal rate?
  - Do students from the most deprived postcodes withdraw more?
  - Does disability status predict withdrawal?
  - Does part-time employment increase withdrawal risk?
- All answers: yes — and these insights shaped the machine learning features

---

### Step 3 — Exploratory Data Analysis

- Performed EDA using Python — Pandas and Plotly
- Produced 10 interactive charts covering:
  - Student outcome distribution (Pass / Withdrawn / Distinction / Fail)
  - Withdrawal rate by course and department
  - Attendance patterns by final result
  - Assessment score distributions
  - VLE engagement trends
  - IMD deprivation band analysis
  - Age band and disability breakdowns
- Key finding: students from the most deprived postcode bands and students
  with part-time jobs showed significantly higher withdrawal rates

---

### Step 4 — Machine Learning

- Trained and compared 3 machine learning models:
  - Logistic Regression — Recall 85.5%, AUC-ROC 92.9%
  - Random Forest — Recall 95.5%, AUC-ROC 92.7%
  - XGBoost — Recall 62.1%, AUC-ROC 92.6%

- Selected Random Forest because Recall is the critical metric in a
  retention context — it tells you how many at-risk students the model
  actually catches out of 100

- Caught and fixed a data leakage problem:
  - Initial features included attendance % and VLE clicks
  - Both are generated across the whole term — by the time this data
    exists the student has often already decided to leave
  - The model was reading the answer — both features removed
  - Rebuilt using only 11 enrolment-time features

- Final 11 features used — all known on Day 1 of term:
  - Gender
  - Disability status
  - Number of previous attempts
  - Credits studied
  - Part-time employment
  - English as first language
  - Transport method
  - Prior education level
  - IMD deprivation band
  - Support sessions planned
  - Mentor assigned

- Risk tier thresholds:
  - 70% or above probability — High Risk
  - 40% to 69% — Medium Risk
  - Below 40% — Low Risk

---

### Step 5 — Streamlit Web Application

- Built a fully deployed 5-page Streamlit application
- Live and accessible to anyone with a browser — no software needed
- Link: https://student-at-risk-project-hqzaznsnhtvna7vfeaabgm.streamlit.app

Page 1 — Dashboard
- College-wide KPI cards: total students, withdrawal rate, High Risk count,
  Medium Risk count, average attendance
- Student Outcome Distribution bar chart
- Risk Tier Distribution donut chart

Page 2 — Early Warnings
- Top 20 most at-risk students ranked by withdrawal probability
- ESFA funding at risk figure — £19,656,000
- Student details: name, course, age, risk %, attendance, disability status

Page 3 — Risk Register
- Full table of all 15,000 students
- Filter by risk tier, course, department, or search by student name
- Sort by withdrawal probability — ready to export and share with tutors

Page 4 — Student Profile
- Search any student by name
- Withdrawal probability score with risk tier badge
- Personal details, academic details, and risk factors displayed side by side
- Personalised recommendations — for example:
  Assign financial support bursary
  Schedule a mentor meeting this week
  Refer to student services

Page 5 — Predict New Student
- Enter a new student's details at the point of enrolment
- Model returns an instant withdrawal risk score
- Deployable before a student has attended a single class

---

### Step 6 — Power BI Executive Dashboard

- Built a 4-page Power BI dashboard for senior leaders and heads of department
- Every page has Department and Risk Tier slicers for drill-down analysis

Page 1 — Executive Summary
- Total students, withdrawal rate, High Risk count, ESFA funding at risk
- Risk tier distribution donut chart
- Withdrawal rate by course bar chart

Page 2 — Withdrawal Risk Analysis
- Withdrawal rate by age band
- Withdrawal rate by IMD deprivation band
- Withdrawal rate by employment status
- Attendance versus withdrawal probability scatter plot

Page 3 — Attendance and Engagement
- Average attendance by term (Autumn, Spring, Summer)
- Weekly VLE click trends across the academic year
- Average attendance by department

Page 4 — Assessment Performance
- Average score by student outcome
- Score progression across assessments 1 to 5
- Late submission rate by department
- Average score by course

---

## 🔒 Ethical Notes

- All data is entirely synthetic — no real student records are used
- The system flags students for human review only — no automated decisions
  are made about individuals without a tutor being involved
- In a real deployment, access would be restricted by role using
  row-level security so tutors only see their own students
- Model performance should be validated across demographic groups before
  any real-world deployment to check for bias

---

## ⚙️ How to Run Locally

1. Clone the repository

   git clone https://github.com/your-username/student-at-risk-system.git
   cd student-at-risk-system

2. Install dependencies

   pip install -r requirements.txt

3. Run the app

   streamlit run app.py

   Opens at http://localhost:8501

---

## 👩‍💻 Author

Daphne Dany Edwin
MSc Data Science — University of Birmingham
BSc Statistics

---

## 📄 Licence

MIT Licence — see LICENSE for details
