import streamlit as st
import pandas as pd
import sqlite3
import groq
import numpy as np
import re

# ---------------- Configuration ----------------
st.set_page_config(layout="wide")

# ---------------- Tooltip CSS ----------------
st.markdown("""
<style>
.tooltip {
  position: relative;
  display: inline-block;
  cursor: pointer;
  color: #3182ce;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 250px;
  background-color: #2c2c2c;
  color: #fff;
  text-align: left;
  border-radius: 8px;
  padding: 10px;
  position: absolute;
  z-index: 1;
  bottom: 125%; 
  left: 0;
  opacity: 0;
  transition: opacity 0.3s;
  font-size: 0.85rem;
  line-height: 1.4;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv(r'sample_3.csv')

df = load_data()
symptom_columns = [col for col in df.columns if "Symptom" in col]

import os



groq_api_key = os.getenv("GROQ_API_KEY")
# Ensure key is found
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in environment!")

groq_client = groq.Groq(api_key=groq_api_key)


# ---------------- Helper Functions ----------------
def clean_sql(raw_sql):
    cleaned = re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE).strip()
    return cleaned

def describe_unique_values(df, columns, max_per_col=10):
    context = "Here are some example values for each column:\n"
    for col in columns:
        unique_vals = df[col].dropna().unique()
        sample_vals = unique_vals[:max_per_col]
        context += f"- {col}: {list(sample_vals)}\n"
    return context

# ---------------- Category Map ----------------
categories = {
    "A. General Overview": {
        "1. General Appearance, Skin & External Features": "Itching, color, texture, lesions",
        "2. Hair, Nails, and Body Surface Changes": "Loss, brittleness, patches, changes",
        "3. Swelling, Burning, Inflammation, Allergic Reactions": "Localized or systemic reactions",
        "4. Body Weight & Nutritional Status": "Emaciation, obesity, weight change",
        "5. Appetite, Thirst & Digestion Patterns": "Cravings, reduced appetite",
        "6. Body Waste & Excretory Patterns": "Urine, stool, sweat, bleeding",
        "7. Fatigue, Sleep, and Energy-related Symptoms": "Yawning, laziness, restlessness, insomnia"
    },
    "B. Digestive & Metabolic Health": {
        "8. Abdominal & GI Tract Symptoms": "Pain, bloating, belching, nausea, constipation, acidity"
    },
    "C. Respiratory & Cardiovascular": {
        "8. Respiratory Symptoms": "Cough, breathlessness, wheezing",
        "9. Cardiovascular & Chest-related Symptoms": "Palpitation, chest pain"
    },
    "D. Neurological & Mental Health": {
        "10. Motor-Sensory Symptoms": "Numbness, twitching, cramps",
        "11. Neurological Symptoms": "Vertigo, dizziness, tremors",
        "13. Cognitive & Behavioral Symptoms": "Desires, aversions, fears",
        "14. Speech Issues": "Slurred or lost speech"
    },
    "E. Sensory Organs (Indriya)": {
        "15. ENT & Voice Disorders": "Discharge, hoarseness",
        "16. Oral, Dental, Facial Symptoms": "Toothache, ulcers, jaw pain",
        "17. Taste, Smell, and Eructations": "Loss of taste/smell"
    },
    "F. Reproductive & Urogenital Health": {
        "18. Reproductive Health & Infertility": "Irregular cycles, impotence",
        "19. Pelvic & Urinary Symptoms": "Burning, frequency"
    },
    "G. Structural & Musculoskeletal": {
        "20. Movement Disorders": "Joint pain, stiffness"
    },
    "H. Temperature Regulation": {
        "21. Temperature Sensations & Fever Patterns": "Chills, fever"
    },
    "I. Miscellaneous": {
        "22. Unclassified or Other Symptoms": "Other unexplained signs"
    }
}

# ---------------- UI Layout ----------------
st.title("🧠 Symptom Navigator + Groq Diagnosis")
left, right = st.columns([2, 1])

with left:
    st.subheader("🔍 Explore by Category")
    for cat, sub_map in categories.items():
        with st.expander(cat):
            for sub, info in sub_map.items():
                st.markdown(
                    f"""
                    <div class='tooltip'>- {sub}
                        <span class='tooltiptext'>{info}</span>
                    </div>
                    """, unsafe_allow_html=True
                )

with right:
    st.subheader("📝 Enter Symptoms")
    symptoms_text = st.text_area("Type multiple symptoms:", height=250, placeholder="e.g. burning sensation while urinating, nausea after eating")
    symptoms_text_list = symptoms_text.split()
    symptoms_text = ", ".join(symptoms_text_list)
    if st.button("🧬 Diagnose") and symptoms_text:
        st.markdown("---")
        symptom_context = describe_unique_values(df, symptom_columns)

        prompt = f"""
You are a SQL assistant for symptom-based diagnosis.

Table: DiseaseData
Columns:
{', '.join(df.columns)}

{symptom_context}

User input: {symptoms_text}

Generate a SQL query that selects Ayurvedic_Diagnosis from DiseaseData where any of these columns: {', '.join(symptom_columns)}, please note, the names need to be put verbatim contain or equal one or more of the user's symptoms using LIKE or =. For multiple symptoms, use AND to combine conditions. Only return the query.
"""

        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": "You convert symptoms to SQL queries. /nothink"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw_sql = response.choices[0].message.content
        sql_query = clean_sql(raw_sql)
        st.code(sql_query, language="sql")

        try:
            conn = sqlite3.connect(":memory:")
            df.to_sql("DiseaseData", conn, index=False, if_exists="replace")
            results = pd.read_sql_query(sql_query, conn)
            conn.close()

            if not results.empty:
                st.success("✅ Possible disease(s) found:")
                st.write(results['Ayurvedic_Diagnosis'].unique().tolist())
            else:
                st.warning("⚠️ No diseases matched the symptoms provided.")
        except Exception as e:
            st.error(f"SQL error: {e}")
