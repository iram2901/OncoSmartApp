import pdfplumber
import pandas as pd
import streamlit as st

# Function to extract data from PDF and create DataFrames
def extract_pdf_data(file_path):
    tables = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extract tables from each page
                page_tables = page.extract_tables()
                for table in page_tables:
                    # Convert the extracted table to a DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])  # First row as headers
                    tables.append(df)
        if tables:
            # Concatenate all DataFrames into one DataFrame
            combined_df = pd.concat(tables, ignore_index=True)
            return combined_df
        else:
            st.error("No tables found in the PDF.")
            return None
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# Load and display the extracted PDF data
file_path = '/mnt/data/test_onco.pdf'
pdf_data_df = extract_pdf_data(file_path)
if pdf_data_df is not None:
    st.write("Extracted Data from PDF (Top 5 Rows):")
    st.write(pdf_data_df.head())
