import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from openai.error import RateLimitError
from io import BytesIO
import pdfplumber  # Add this import to handle PDF files

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure API key is set up in the environment

# Streamlit app
st.title('AI Powered - OncoSmart Insights')

# File uploader widget for multiple files (now includes PDF files)
uploaded_files = st.file_uploader("Choose one or more files", type=["xlsx", "csv", "pdf"], accept_multiple_files=True)

# Function to read the file based on type and convert columns to lowercase
def load_data(files):
    dfs = []  # List to store dataframes
    for file in files:
        file_type = file.name.split('.')[-1]
        if file_type == 'xlsx':
            df = pd.read_excel(file)
        elif file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type == 'pdf':
            try:
                # Extract tables from PDF using pdfplumber
                with pdfplumber.open(file) as pdf:
                    pdf_dfs = []
                    for page in pdf.pages:
                        # Extract table from each page and convert to DataFrame
                        table = page.extract_table()
                        if table:
                            page_df = pd.DataFrame(table[1:], columns=table[0])  # First row as headers
                            pdf_dfs.append(page_df)
                    if pdf_dfs:
                        df = pd.concat(pdf_dfs, ignore_index=True)
                    else:
                        st.error(f"No tables found in the PDF: {file.name}")
                        continue
            except Exception as e:
                st.error(f"Error reading PDF file: {file.name}. {e}")
                continue
        else:
            st.error(f"Unsupported file type: {file.name}")
            continue
        
        # Clean column names and data
        df.columns = [col.strip().lower() for col in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.lower()
        
        dfs.append(df)
    
    return dfs  # Return the list of dataframes

# Function to merge the datasets on different key columns
def merge_datasets(dfs, left_key, right_key):
    if len(dfs) > 1:
        merged_df = dfs[0]  # Start with the first dataframe
        for df in dfs[1:]:
            if left_key not in merged_df.columns or right_key not in df.columns:
                st.error(f"Key columns '{left_key}' or '{right_key}' not found in the datasets.")
                return None
            # Merge with the next dataframe on the specified keys
            merged_df = pd.merge(merged_df, df, left_on=left_key, right_on=right_key, how='inner')  # Using 'inner' join to find matches
        return merged_df
    else:
        return dfs[0]  # If there's only one dataframe, return it as is

# Function to ask OpenAI for chart instructions
def ask_openai_for_chart(df, user_query):
    data_preview = df.head().to_string()

    prompt = f"""
    I have the following data from a file (all column names and data are in lowercase):
    {data_preview}

    User query: {user_query}

    Please provide a valid Python code snippet using matplotlib or seaborn to generate the appropriate chart.
    Avoid using multi-dimensional indexing (e.g., df[:, None]). Use simple DataFrame indexing or conversion to NumPy arrays where necessary.
    Ensure that the code handles common errors and is compatible with pandas DataFrames.
    """

    try:
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract only the code snippet
        code = response.choices[0].message['content']
        return code
    except RateLimitError:
        st.error("Rate limit exceeded. Please try again later.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to clean up the generated code
def clean_generated_code(code_str):
    # Replace occurrences of 'dataframe' with 'df'
    code_str = re.sub(r'\bdataframe\b', 'df', code_str)

    # Remove irrelevant parts of the code
    code_after_import = re.sub(r"^.*?(import pandas)", r"\1", code_str, flags=re.DOTALL)
    clean_code = re.sub(r"(plt\.show\(\)).*", r"\1", code_after_import, flags=re.DOTALL)

    # Remove multi-dimensional indexing attempts like obj[:, None]
    clean_code = re.sub(r"[:,]\s*None", '', clean_code)

    return clean_code

# Function to execute the code and generate the chart
def execute_chart_code(code, df):
    try:
        # Display the generated code for debugging purposes
        # st.write("Generated Code for Review:")
        # st.code(code)

        # Execute the code
        exec(code, {'df': df, 'plt': plt, 'sns': sns, 'pd': pd})
    except SyntaxError as e:
        st.error(f"Syntax error in the generated code: {e}")
    except NameError as e:
        st.error(f"Name error in the generated code: {e}")
    except Exception as e:
        st.error(f"An error occurred while generating the chart: {e}")

if uploaded_files:
    try:
        # Load and clean the data
        dfs = load_data(uploaded_files)
        if dfs:
            # Display the preview of each uploaded file
            for i, df in enumerate(dfs):
                st.write(f"Data Preview for File {i+1} (Top 5 Rows):")
                st.write(df.head())
            
            if len(dfs) > 1:
                # Specify the key columns for merging (without auto-populating default values)
                left_key = st.text_input("Enter the column name to use as the key for merging in the first dataset:")
                right_key = st.text_input("Enter the column name to use as the key for merging in the second dataset:")
                
                # Add a submit button
                if st.button('Submit'):
                    # Ensure both keys are entered before attempting to merge
                    if left_key and right_key:
                        # Merge the datasets on the specified keys
                        merged_df = merge_datasets(dfs, left_key, right_key)
                        
                        if merged_df is not None:
                            st.write("Merged Data Preview (Top 5 Rows):")
                            st.write(merged_df.head())  # Display the top 5 rows of the merged dataset

                            # User input for chart query
                            user_query = st.text_input("Enter your chart query:")

                            if user_query:
                                # Generate chart code based on query
                                chart_code = ask_openai_for_chart(merged_df, user_query)
                                if chart_code:
                                    # Clean up the generated code
                                    chart_code_modified = clean_generated_code(chart_code)

                                    # Execute the chart code
                                    execute_chart_code(chart_code_modified, merged_df)
                                    st.pyplot(plt)  # Display the chart

    except Exception as e:
        st.error(f"An error occurred: {e}")
