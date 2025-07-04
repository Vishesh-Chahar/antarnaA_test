import streamlit as st
import pandas as pd
import numpy as np

# --------------- STEP 1: Load data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Kaam\projects\Antarnaa\sample.csv')
    return df

df = load_data()

symptom_columns = ['Symptom P1', 'Symptom P2']

# Initialize session state
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'remaining_df' not in st.session_state:
    st.session_state.remaining_df = None

st.title("🧠 Adaptive Diagnosis System")

# STEP 2: Category selection interface
st.subheader("Step 1: Select Category and Sub Category")

category = st.selectbox("Select Category", sorted(df['Category'].dropna().unique()))

filtered_df_cat = df[df['Category'] == category]

subcategory = st.selectbox("Select Sub Category", sorted(filtered_df_cat['Sub Category'].dropna().unique()))

filtered_df = filtered_df_cat[filtered_df_cat['Sub Category'] == subcategory]

# Start button to initialize adaptive filtering
if st.button("Start Diagnosis"):
    st.session_state.remaining_df = filtered_df.copy()
    st.session_state.answers = {}
    st.rerun()

# If remaining_df is set, start adaptive questioning
if st.session_state.remaining_df is not None:

    # Display current answers
    if st.session_state.answers:
        st.subheader("Your Answers So Far:")
        for k, v in st.session_state.answers.items():
            st.write(f"{k}: {v}")

    # Check stop condition
    if st.session_state.answers and st.session_state.remaining_df['Disease No.'].nunique() == 1:
        st.success(f"Diagnosis complete: Disease No. = {st.session_state.remaining_df['Disease No.'].unique()[0]}")
        st.stop()

    # Find remaining symptoms to ask
    remaining_symptoms = [col for col in symptom_columns if col not in st.session_state.answers]

    if not remaining_symptoms:
        st.warning("All symptoms answered. No further narrowing possible.")
        st.write("Possible diseases:")
        st.write(st.session_state.remaining_df['Disease No.'].unique().tolist())
        st.stop()

    # Calculate entropy for each remaining symptom
    entropy_scores = {}
    for symptom in remaining_symptoms:
        counts = st.session_state.remaining_df[symptom].value_counts()
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        entropy_scores[symptom] = entropy

    # Pick most informative symptom (lowest entropy)
    best_symptom = min(entropy_scores, key=entropy_scores.get)

    # Ask next question
    st.subheader(f"Next question: {best_symptom}")
    options = [''] + sorted(st.session_state.remaining_df[best_symptom].dropna().unique().tolist())
    response = st.selectbox(f"Do you have {best_symptom}?", options, key=best_symptom)

    if st.button("Submit Answer"):
        if response != '':
            st.session_state.answers[best_symptom] = response
            st.session_state.remaining_df = st.session_state.remaining_df[st.session_state.remaining_df[best_symptom] == response]
        else:
            st.session_state.answers[best_symptom] = 'Unknown'
            st.session_state.remaining_df = st.session_state.remaining_df[st.session_state.remaining_df[best_symptom].isna()]
        st.rerun()

    # Show possible diseases anytime
    with st.expander("See current possible diseases"):
        st.write(st.session_state.remaining_df['Disease No.'].unique().tolist())
