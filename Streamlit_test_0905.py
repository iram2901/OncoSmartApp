import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from openai.error import RateLimitError

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure API key is set up in the environment

# Streamlit app
st.title('AI Powered - OncoSmart Insights')

# Function to read the Excel file
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to clean column names
def clean_column_names(df):
    df.columns = [col.strip().lower() for col in df.columns]
    return df

# Function to ask OpenAI for chart instructions
def ask_openai_for_chart(df, user_query):
    data_preview = df.head().to_string()

    prompt = f"""
    I have the following data from an Excel file (all column names are in lowercase):
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
        st.write("Generated Code for Review:")
        st.code(code)

        # Convert DataFrame columns to
        # arrays if the code is for a line chart or other plot type
        if "lineplot" in code or "plot(kind='line')" in code or "plot(" in code:
            x_col = re.findall(r'filtered_df\[[\'\"](.*?)[\'\"]\]', code)  # Extract column names from the code
            for col in x_col:
                if col in df.columns:
                    df[col] = df[col].to_numpy()

        # Execute the code
        exec(code, {'df': df, 'plt': plt, 'sns': sns, 'pd': pd})
    except SyntaxError as e:
        st.error(f"Syntax error in the generated code: {e}")
    except NameError as e:
        st.error(f"Name error in the generated code: {e}")
    except Exception as e:
        st.error(f"An error occurred while generating the chart: {e}")

# Hardcoded file path
file_path = "patient_dataset.xlsx"

try:
    # Load and clean the data
    df = load_excel_data(file_path)
    df = clean_column_names(df)
    st.write("Data Preview (Top 5 Rows):")
    st.write(df.head())  # Display the top 5 rows

    # User input for chart query
    user_query = st.text_input("Enter your chart query:")

    if user_query:
        # Generate chart code based on query
        chart_code = ask_openai_for_chart(df, user_query)
        if chart_code:
            # Clean up the generated code
            chart_code_modified = clean_generated_code(chart_code)

            # Execute the chart code
            execute_chart_code(chart_code_modified, df)
            st.pyplot(plt)  # Display the chart

except FileNotFoundError:
    st.error(f"File not found: {file_path}")
except Exception as e:
    st.error(f"An error occurred: {e}")
