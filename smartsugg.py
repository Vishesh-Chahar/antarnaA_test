import streamlit as st
import pandas as pd
import sqlite3
import groq
import numpy as np
import re

# ---------------- SQL Sanitizer ----------------
def clean_sql(raw_sql: str) -> str:
    cleaned = re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE).strip()
    return cleaned

# ---------------- Load dataset ----------------
@st.cache_data
def load_df():
    return pd.read_csv(r'C:\Kaam\projects\Antarnaa\sample.csv')

df = load_df()
symptom_columns = ['Symptom P1', 'Symptom P2']
describe_columns = ['Category', 'Sub Category', 'Symptom P1', 'Symptom P2']

# ---------------- Value Describer ----------------
def describe_unique_values(df, columns, max_per_col=10):
    context = "Here are some example values for each column:\n"
    for col in columns:
        unique_vals = df[col].dropna().unique()
        sample_vals = unique_vals[:max_per_col]
        context += f"- {col}: {list(sample_vals)}\n"
    return context

value_context = describe_unique_values(df, describe_columns)

# ---------------- Groq API Setup ----------------
groq_api_key = "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"
groq_client = groq.Groq(api_key=groq_api_key)

# ---------------- Streamlit Setup ----------------
st.title("🧠 Smart Symptom Chat (Groq + Entropy)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "symptoms_given" not in st.session_state:
    st.session_state.symptoms_given = []  # stores (column, value) pairs
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = df.copy()

user_input = st.chat_input("Enter symptoms or ask 'what should I ask next?'")

# ---------------- Entropy Helpers ----------------
def compute_entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

def most_informative_question(df_input, known_questions, top_k=5):
    required_cols = set(symptom_columns + ["Disease No."])
    if not required_cols.issubset(df_input.columns):
        df_input = load_df()

    base_entropy = compute_entropy(df_input["Disease No."])
    best_gain = -1
    best_question = None

    for col in symptom_columns:
        asked_vals = [val for c, val in known_questions if c == col]
        top_values = df_input[col].dropna().value_counts().head(top_k).index.tolist()

        for val in top_values:
            if val in asked_vals:
                continue
            filtered = df_input[df_input[col] == val]
            if filtered.empty:
                continue
            new_entropy = compute_entropy(filtered["Disease No."])
            info_gain = base_entropy - new_entropy

            if info_gain > best_gain:
                best_gain = info_gain
                best_question = (col, val)

    return best_question

# ---------------- Chat Logic ----------------
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    if "what should i ask next" in user_input.lower():
        source_df = st.session_state.filtered_df if "filtered_df" in st.session_state else df
        best = most_informative_question(source_df, st.session_state.symptoms_given)

        if best:
            col, val = best
            reply = f"💡 You should ask if the user has **'{val}'** under **{col}** — it helps narrow the diagnosis best."
        else:
            reply = "✅ No more informative questions left to ask."
        st.session_state.chat_history.append(("assistant", reply))

    else:
        # Match input symptom to known values
        matched = None
        user_input_lower = user_input.lower()

        for col in symptom_columns:
            matches = df[col].dropna().unique()
            for val in matches:
                if str(val).lower() in user_input_lower:
                    matched = (col, val)
                    break
            if matched:
                break

        if matched:
            st.session_state.symptoms_given.append(matched)
        else:
            st.session_state.symptoms_given.append(("free", user_input))

        # ---------- Prompt Generation ----------
        schema = """
Table: DiseaseData
Columns:
- Category (TEXT)
- Sub Category (TEXT)
- Symptom P1 (TEXT)
- Symptom P2 (TEXT)
- Disease No. (TEXT)
"""

        symptom_lines = [
            f"{col} = '{val}'" if col != "free" else f"'{val}'"
            for col, val in st.session_state.symptoms_given
        ]

        prompt = f"""
You are a medical SQL assistant.

Only respond with a valid SQL query. No markdown. No ```sql blocks. No explanation.

{schema}

{value_context}

User symptoms: {', '.join(symptom_lines)}

Write a SQL query that finds Disease No. from DiseaseData where Symptom P1 or Symptom P2 matches any of these using = or LIKE. Only return the query.
"""

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "You convert symptoms to SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw_sql = response.choices[0].message.content
        sql_query = clean_sql(raw_sql)
        st.session_state.chat_history.append(("assistant", f"🔍 Generated SQL:\n```sql\n{sql_query}\n```"))

        try:
            conn = sqlite3.connect(":memory:")
            df.to_sql("DiseaseData", conn, index=False, if_exists="replace")
            results = pd.read_sql_query(sql_query, conn)
            conn.close()

            if not results.empty:
                st.session_state.filtered_df = results.copy()
                st.session_state.chat_history.append(
                    ("assistant", f"📋 Possible diseases:\n{results['Disease No.'].unique().tolist()}")
                )
            else:
                st.session_state.chat_history.append(("assistant", "❌ No diseases found matching those symptoms."))

        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"⚠️ SQL error after cleaning: {e}"))

# ---------------- Display Chat ----------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
