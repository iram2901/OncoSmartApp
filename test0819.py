import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from openai import RateLimitError, OpenAIError

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')


# Function to read the file based on type and convert columns to lowercase
# Function to read the file based on type and convert columns to lowercase
def load_data(file_path):
    file_type = file_path.split('.')[-1]
    if file_type == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type")

    # Clean column names and data
    df.columns = [col.strip().lower() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.lower()

    # Dynamically check and convert all potential date columns to datetime format
    for col in df.columns:
        if 'date' in col:  # If 'date' is part of the column name, it could be a date
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"{col} is already in datetime format")
            else:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime if not
                print(f"{col} has been converted to datetime format")

    return df


# Function to ask OpenAI for chart instructions
def ask_openai_for_chart(df, user_query):
    data_preview = df.head().to_string()

    prompt = f"""
    I have the following data from a file (all column names and data are in lowercase):
    {data_preview}

    User query: {user_query}

    Please provide a valid Python code snippet using matplotlib or seaborn to generate the appropriate chart.
    Use the DataFrame 'df' that is already provided in the environment. Avoid reading files or referencing file paths.
    Avoid using multi-dimensional indexing (e.g., df[:, None]). Use simple DataFrame indexing or conversion to NumPy arrays where necessary.
    Ensure that the code handles common errors and is compatible with pandas DataFrames.
    """

    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
            # Print Prompt and then contenate "Do not include any commentary like "Here is the chart code or anything. I just want python code as I want to run the code as is"

        )
        # Extract only the code snippet
        code = response.choices[0].message.content
        return code
    except RateLimitError:
        raise Exception("Rate limit exceeded. Please try again later.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


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
        # Print the code that will be executed for generating the chart
        print("\nGenerated Python Code for the Chart:\n")
        print(code)

        clean_code = code.replace("```python", "")

        start_index = clean_code.find("import matplotlib")

        # If the line is found, slice the text from that index onward
        if start_index != -1:
            clean_code = clean_code[start_index:]
        else:
            clean_code = clean_code  # If not found, keep the original text

        print(clean_code)

        exec(clean_code)

        # Check if plt.show() was called within the code, if not, call it explicitly
        if plt.get_fignums():
            plt.show()

    except SyntaxError as e:
        print(f"Syntax error in the generated code: {e}")
    except NameError as e:
        print(f"Name error in the generated code: {e}")
    except Exception as e:
        print(f"An error occurred while generating the chart: {e}")


# Example usage
if __name__ == "__main__":
    try:
        # Hardcoded file path
        file_path = r"C:\Users\iramc\Downloads\patient_dataset.xlsx"
        df = load_data(file_path)

        # Display data preview
        print("Data Preview (Top 5 Rows):")
        print(df.head())

        # User input for chart query
        user_query = input("Enter your chart query: ")

        if user_query:
            # Generate chart code based on query
            chart_code = ask_openai_for_chart(df, user_query)
            if chart_code:
                # Clean up the generated code
                chart_code_modified = clean_generated_code(chart_code)

                # Execute the chart code and display the generated code
                execute_chart_code(chart_code_modified, df)

    except Exception as e:
        print(f"An error occurred: {e}")