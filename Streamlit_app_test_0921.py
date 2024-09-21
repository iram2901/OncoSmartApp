import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from openai import RateLimitError, OpenAIError

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Streamlit app begins here

# Function to read the file based on type and convert columns to lowercase
def load_data(file):
    file_type = file.name.split('.')[-1]
    if file_type == 'xlsx':
        df = pd.read_excel(file)
    elif file_type == 'csv':
        df = pd.read_csv(file)
    else:
        st.error("Unsupported file type")
        return None

    # Clean column names and data
    df.columns = [col.strip().lower() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.lower()

    # Dynamically check and convert all potential date columns to datetime format silently
    for col in df.columns:
        if 'date' in col:  # If 'date' is part of the column name, it could be a date
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime if not

    return df

# Function to ask OpenAI for chart instructions
def ask_openai_for_chart(df, user_query):
    headers_preview = df.columns.to_list()
    data_preview = df.head(5).to_string()

    prompt = f"""
    I have the following data from a file (all column names and data are in lowercase):
    {data_preview}

    The headers of the file are:
    {headers_preview}

    User query: {user_query}

    Please provide a valid Python code snippet using matplotlib or seaborn to generate the appropriate chart.
    Use the DataFrame 'df' that is already provided in the environment. Avoid reading files or referencing file paths.
    Avoid using multi-dimensional indexing (e.g., df[:, None]). Use simple DataFrame indexing or conversion to NumPy arrays where necessary.
    Ensure that the code handles common errors and is compatible with pandas DataFrames.

    For bar charts, please annotate each bar with the corresponding value (e.g., number of patients), so the value appears on top of each bar. 
    For line charts, ensure that the points on the line are labeled with their values.
    """

    additional_instructions = """
    Do not include any commentary like 'Here is the chart code' or anything else. 
    I just want Python code as I want to run the code as is.
    """

    full_prompt = prompt + additional_instructions

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
        code = response.choices[0].message.content
        return code

    except RateLimitError:
        st.error("Rate limit exceeded. Please try again later.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to clean up the generated code
def clean_generated_code(code_str):
    code_str = re.sub(r'\bdataframe\b', 'df', code_str)
    code_after_import = re.sub(r"^.*?(import pandas)", r"\1", code_str, flags=re.DOTALL)
    clean_code = re.sub(r"(plt\.show\(\)).*", r"\1", code_after_import, flags=re.DOTALL)
    clean_code = re.sub(r"[:,]\s*None", '', clean_code)
    return clean_code

# Function to execute the code and generate the chart without showing it in the UI
def execute_chart_code(code, df):
    try:
        clean_code = code.replace("```python", "")
        start_index = clean_code.find("import matplotlib")
        if start_index != -1:
            clean_code = clean_code[start_index:]

        exec(clean_code, {"df": df})
        plt.grid(False) # Removing grid lines from chart
        if plt.get_fignums():
            st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"An error occurred while generating the chart: {e}")

# Streamlit UI components
st.title("AI Powered - OncoSmart Insights")

# File upload for user
uploaded_file = st.file_uploader("Upload your dataset (Excel or CSV)", type=['csv', 'xlsx'])

# If a file is uploaded, process the data
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.write("Preview of your data:")
        st.dataframe(df.head())

        # Input for user query
        user_query = st.text_input("Enter your chart query")

        # Add a button to generate the chart
        if st.button("Generate Chart"):
            if user_query:
                chart_code = ask_openai_for_chart(df, user_query)
                if chart_code:
                    chart_code_clean = clean_generated_code(chart_code)

                    # Execute the chart without displaying the raw code
                    execute_chart_code(chart_code_clean, df)
            else:
                st.warning("Please enter a chart query before generating the chart.")
