import streamlit as st
import pandas as pd
import sqlite3
import pandasql as ps
import groq

# ------------------ Load your CSV ------------------

@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Kaam\projects\Antarnaa\sample.csv')
    return df

df = load_data()

# ------------------ Extract Schema Context ------------------

# Build dynamic context for better prompting
def generate_schema_context(df):
    schema = "Table: DiseaseData\nColumns:\n"
    for col in df.columns:
        schema += f"- {col} ({df[col].dtype})\n"
    schema += "\n"
    schema += "Possible values:\n"
    for col in ['Category', 'Sub Category']:
        unique_values = df[col].dropna().unique().tolist()
        schema += f"- {col}: {unique_values}\n"
    return schema

schema_context = generate_schema_context(df)

# ------------------ Streamlit UI ------------------

st.title("🧠 Natural Language Symptom Query (LLaMA-4 + SQL)")

st.write("Describe patient symptoms in natural language. The system will generate SQL to query possible diseases.")

user_input = st.text_area("Enter patient symptoms:")

if st.button("Submit") and user_input.strip() != "":
    
    # ------------------ Build Prompt ------------------

    full_prompt = f"""
You are a medical data assistant with access to patient records.

The database schema is as follows:

{schema_context}

The user will describe symptoms in natural language. You need to generate an equivalent SQL query that will search for matching records in this table.

Guidelines:
- Map symptoms to columns 'Symptom P1' and 'Symptom P2' wherever possible.
- Use partial matching (LIKE or CONTAINS) where exact matching may fail.
- Always select Disease No. column in the final output.
- If unsure, be conservative and select broader results.
- Output only the SQL query, no explanation or additional text.

User Input: "{user_input}"

Generate SQL:
"""

    st.write("🔗 Sending query to Groq...")

    # ------------------ Call Groq API ------------------

    groq_api_key = "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"
    groq_client = groq.Groq(api_key=groq_api_key)

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "system", "content": "You are a SQL generation expert."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0
    )

    generated_sql = response.choices[0].message.content.strip()
    st.code(generated_sql, language='sql')

    # ------------------ Execute SQL Locally ------------------

    # Create sqlite3 in-memory database
    conn = sqlite3.connect(":memory:")
    df.to_sql("DiseaseData", conn, index=False, if_exists="replace")

    try:
        result_df = pd.read_sql_query(generated_sql, conn)
        if result_df.empty:
            st.warning("No diseases found for given symptoms.")
        else:
            st.success("Matching diseases found:")
            st.dataframe(result_df)
    except Exception as e:
        st.error(f"SQL execution failed: {e}")
    finally:
        conn.close()
