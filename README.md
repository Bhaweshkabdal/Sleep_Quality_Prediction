# Sleep Quality Prediction using Machine Learning

This project predicts a person's **sleep quality (Good / Average / Poor)** based on lifestyle and health-related factors.  
The purpose of this project is to show how simple Machine Learning models can evaluate sleep patterns and help identify risks related to poor sleep conditions.

---

## Features of this Project

- Takes user input such as:
  - Sleep Duration (hours)
  - Stress Level (Low / Medium / High)
  - Exercise Frequency (None / Weekly / Daily)
  - Coffee Intake (cups/day)
  - Screen Time (hours/day)
- Predicts Sleep Quality with trained ML model
- Web Interface built using **Streamlit**
- Simple and beginner-friendly project

---

## Technologies Used

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| ML Model Used | Random Forest Classifier |
| Frontend UI | Streamlit |
| Data Handling | Pandas, Label Encoding |
| Dataset Format | CSV |

---

## Project Structure

Sleep_Quality_Prediction
│── app.py # Streamlit Web App File
│── predict.py # Terminal based input prediction
│── model.py # ML Model training script
│── sleep_data.csv # Dataset used for training
│── README.md # (Create this file and paste content here)
│── .gitignore



---

## How to Run the Web App

### Step 1 – Install Required Packages


### Step 2 – Run Streamlit App


The interface will open in your browser automatically.

---

## Sample UI Input

| Input Field | Example |
|------------|----------|
| Sleep Duration | 6 hours |
| Stress Level | Medium |
| Exercise | Weekly |
| Coffee Intake | 2 cups |
| Screen Time | 5 hours |

**Output Example:**  

---

## Future Improvements

- Use a larger real-world dataset
- Add heart rate & oxygen saturation (SpO2) sensors
- Deploy the app online using Render / HuggingFace / Heroku
- Mobile friendly UI
- Add Data Visualizations (Correlation Graphs, Heatmaps)

---

## Project Developed By

**Bhawesh Kabdal (GitHub: Bhaweshkabdal)**  
Project Mentor: *ChatGPT Assistance*

---

If you like this project, give it a **Star** on GitHub.

