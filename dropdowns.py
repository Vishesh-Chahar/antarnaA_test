import streamlit as st
import pandas as pd

# Load your CSV (replace with actual filename)
df = pd.read_csv(r'C:\Kaam\projects\Antarnaa\sample.csv')

st.title("Cascading Dropdowns: Category to Symptoms")

# First Dropdown: Category (required)
category = st.selectbox("Select Category", sorted(df['Category'].dropna().unique()))

# Filter based on Category
filtered_df1 = df[df['Category'] == category]

# Second Dropdown: Sub Category (required)
subcategory = st.selectbox("Select Sub Category", sorted(filtered_df1['Sub Category'].dropna().unique()))

# Filter based on Sub Category
filtered_df2 = filtered_df1[filtered_df1['Sub Category'] == subcategory]

# Third Dropdown: Symptom P1 (optional)
symptom_p1_options = [''] + sorted(filtered_df2['Symptom P1'].dropna().unique().tolist())
symptom_p1 = st.selectbox("Select Symptom P1 (optional)", symptom_p1_options)

if symptom_p1 != '':
    filtered_df2 = filtered_df2[filtered_df2['Symptom P1'] == symptom_p1]

# Fourth Dropdown: Symptom P2 (optional)
symptom_p2_options = [''] + sorted(filtered_df2['Symptom P2'].dropna().unique().tolist())
symptom_p2 = st.selectbox("Select Symptom P2 (optional)", symptom_p2_options)

if symptom_p2 != '':
    filtered_df2 = filtered_df2[filtered_df2['Symptom P2'] == symptom_p2]

# Show current selections
st.write(f"Your selection: {category} > {subcategory} > {symptom_p1 if symptom_p1 else 'Any'} > {symptom_p2 if symptom_p2 else 'Any'}")

# Show Disease No.
if not filtered_df2.empty:
    disease_list = filtered_df2['Disease No.'].dropna().unique()
    st.subheader("Matching Disease No.:")
    for disease in disease_list:
        st.write(f"{disease}")
else:
    st.warning("No Disease No. found for this combination.")
