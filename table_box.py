import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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
    return pd.read_csv('sample_3.csv')

df = load_data()
symptom_columns = [col for col in df.columns if "Symptom" in col]

# ---------------- Vector DB ----------------
@st.cache_resource
def build_or_load_embeddings():
    embed_file = "embeddings.pkl"
    if os.path.exists(embed_file):
        with open(embed_file, "rb") as f:
            return pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_symptoms = df[symptom_columns].fillna("").agg(" ".join, axis=1)
    embeddings = model.encode(all_symptoms.tolist(), show_progress_bar=True)

    with open(embed_file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

symptom_embeddings = build_or_load_embeddings()
# model = SentenceTransformer("all-MiniLM-L6-v2")

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
st.title("🧠 Symptom Navigator + Local Vector Diagnosis")

col1, col2, col3 = st.columns([1.5, 1.5, 1])

with col1:
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
              
model = SentenceTransformer("all-MiniLM-L6-v2")

with col2:
    st.subheader("📝 Enter Symptoms")
    user_input = st.text_area("Type multiple symptoms:", height=250, placeholder="e.g. burning sensation while urinating, nausea after eating")

    threshold = st.slider("🔬 Similarity Threshold for 'Relevant' Matches", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

    if st.button("🧬 Vector Search Diagnose") and user_input:
        input_embedding = model.encode([user_input])[0]  # Explicit device to avoid meta errors
        similarities = cosine_similarity([input_embedding], symptom_embeddings)[0]

        df["similarity"] = similarities

        exact_match = df[df[symptom_columns].apply(lambda row: any(sym.lower() in user_input.lower() for sym in row.astype(str)), axis=1)]
        relevant_match = df[df["similarity"] >= threshold]

        st.markdown("### 🧾 Exact Match Diagnoses")
        st.dataframe(exact_match[["Ayurvedic_Diagnosis"]])

        st.markdown("### 📋 Relevant Diagnoses by Similarity")
        st.dataframe(relevant_match[["Ayurvedic_Diagnosis", "similarity"]].sort_values(by="similarity", ascending=False))

        # ---------------- Adaptive Suggestions ----------------
        st.markdown("---")
        st.subheader("🧠 Suggested Narrowing Symptoms")
        subset = pd.concat([exact_match, relevant_match]).drop_duplicates()
        symptom_pool = subset[symptom_columns].fillna("").apply(lambda x: x.str.lower())
        input_tokens = set(re.findall(r"\w+", user_input.lower()))

        entropy_scores = {}
        for col in symptom_columns:
            values = symptom_pool[col]
            freq = values.value_counts(normalize=True)
            entropy = -np.sum(freq * np.log2(freq)) if not freq.empty else 0
            entropy_scores[col] = entropy

        top_col = max(entropy_scores, key=entropy_scores.get)
        new_symptoms = set(symptom_pool[top_col]) - input_tokens - {''}
        suggestions = sorted(new_symptoms)[:10]

        if suggestions:
            st.write(f"Most discriminative symptom field: **{top_col}**")
            st.write("Try asking about:", suggestions)
        else:
            st.write("No additional narrowing symptoms found.")
