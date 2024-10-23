import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
import openai  # Replacing Azure OpenAI
import io
import fitz  # PyMuPDF

# Set the OpenAI API key
openai.api_key = "sk-GvnLTmDzChPHRWrlAASLGgRTl5Zglr7cd6Zit4_jo3T3BlbkFJzTIDN_hjFSDWM5GM1MRHDtto7_ZkKpLvlYdWP2QeIA"

# Function to read the file based on type and convert columns to lowercase or extract text from PDF
def load_data(file):
    with st.spinner("Loading data..."):
        file_type = file.name.split('.')[-1]
        if file_type == 'xlsx':
            df = pd.read_excel(file)
            df.columns = [col.strip().lower() for col in df.columns]
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.lower()

            # Dynamically check and convert all potential date columns to datetime format silently
            for col in df.columns:
                if 'date' in col:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            return df

        elif file_type == 'csv':
            df = pd.read_csv(file)
            df.columns = [col.strip().lower() for col in df.columns]
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.lower()

            for col in df.columns:
                if 'date' in col:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            return df

        elif file_type == 'pdf':
            pdf_text = extract_pdf_text(file)
            return pdf_text

        else:
            st.error("Unsupported file type")
            return None

# Function to extract text from a PDF file using PyMuPDF
def extract_pdf_text(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {e}")
        return None


# Function to summarize PDF text using OpenAI
def summarize_pdf_text(pdf_text):
    with st.spinner("Summarizing PDF..."):
        prompt = f"""
        Summarize the following content from the PDF:

        {pdf_text}

        Please provide a concise summary of the key points.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message["content"]
            return summary

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None


# Function to ask OpenAI for chart generation based on user query
def ask_openai_for_chart(df, user_query):
    headers_preview = df.columns.to_list()
    data_preview = df.head(5).to_string()

    prompt = f"""
    I have the following data from a file (all column names and data are in lowercase):
    {data_preview}

    The headers of the file are:
    {headers_preview}

    User query: {user_query}

    Please generate a valid Python code snippet using matplotlib or seaborn to generate the appropriate chart.
    Use the DataFrame 'df' that is already provided in the environment. **Do not create any sample data or define new DataFrames**. Only use 'df' that is passed from the user environment.
    For bar charts, annotate each bar with the corresponding value on top of each bar. For line charts, label the points on the line with their values.
    Please ensure to import pandas as pd, matplotlib.pyplot as plt, and seaborn as sns. 
    """

    additional_instructions = """
    Return only the Python code. Do not include commentary such as 'Here is the chart code'. 
    """

    full_prompt = prompt + additional_instructions

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
        code = response.choices[0].message.content
        return code

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Rest of the code continues unchanged...
def ask_azure_openai_for_chart(df, user_query):
    headers_preview = df.columns.to_list()
    data_preview = df.head(5).to_string()

    prompt = f"""
    I have the following data from a file (all column names and data are in lowercase):
    {data_preview}

    The headers of the file are:
    {headers_preview}

    User query: {user_query}

    Please generate a valid Python code snippet using matplotlib or seaborn to generate the appropriate chart.
    Use the DataFrame 'df' that is already provided in the environment. **Do not create any sample data or define new DataFrames**. Only use 'df' that is passed from the user environment.
    Do not include any multi-dimensional indexing (e.g., df[:, None]). Use simple DataFrame indexing or conversion to NumPy arrays if necessary.
    For bar charts, annotate each bar with the corresponding value on top of each bar. For line charts, label the points on the line with their values.
    Please ensure to import pandas as pd, matplotlib.pyplot as plt, and seaborn as sns. 
    Please ensure to not use reset_index() use sort_index() while generating chart.
    Please ensure to keep the x axis as whole number do not change into decimal.
    """

    additional_instructions = """
    Return only the Python code. Do not include commentary such as 'Here is the chart code'. 
    """

    full_prompt = prompt + additional_instructions

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        code = response.choices[0].message.content
        return code

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None




# Function to clean up the generated code
def clean_generated_code(code_str):
    code_str = re.sub(r'\bdata\b', 'df', code_str)  # Replace any instance of 'data' with 'df'
    code_str = re.sub(r'\bdataframe\b', 'df', code_str)  # Replace any instance of 'dataframe' with 'df'
    sample_data_pattern = re.compile(r'data\s*=\s*{[^}]*}', re.DOTALL)
    code_str = sample_data_pattern.sub('', code_str)
    # Step 3: Replace 'reset_index()' with 'sort_index()' to avoid indexing issues with seaborn
    code_str = re.sub(r'\.reset_index\(\)', '.sort_index()', code_str)
    # In case 'reset_index(drop=True)' is used, handle that as well:
    code_str = re.sub(r'\.reset_index\(drop=True\)', '.sort_index()', code_str)
    code_after_import = re.sub(r"^.*?(import pandas)", r"\1", code_str, flags=re.DOTALL)
    clean_code = re.sub(r"(plt\.show\(\)).*", r"\1", code_after_import, flags=re.DOTALL)
    clean_code = re.sub(r"[:,]\s*None", '', clean_code)

    return clean_code

# Function to execute the code and generate the chart
def execute_chart_code(code, df, prompt):
    with st.spinner("Generating chart..."):
        try:
            # Clean the generated code to ensure it's Streamlit-compatible
            clean_code = clean_generated_code(code)

            # Debugging: Check the DataFrame and generated code
            #st.write("Columns in the DataFrame:", df.columns)
            #st.write("Sample data in DataFrame:", df.head())
            #st.write("Generated Code:", clean_code)

            # Prepare the local environment with necessary libraries and the DataFrame
            local_namespace = {
                "df": df,
                "plt": plt,
                "pd": pd,
                "sns": sns
            }

            # Execute the cleaned code
            exec(clean_code, local_namespace)

            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Store the image and the prompt in session state
            if 'charts' not in st.session_state:
                st.session_state.charts = []
            st.session_state.charts.append((buf, prompt))

            # Display the chart immediately after generation
            #st.pyplot(plt)

            # Clear the current figure
            plt.clf()

        except Exception as e:
            st.error(f"An error occurred while generating the chart: {e}")


# Function to ask Azure OpenAI to answer questions based on the data
def ask_azure_openai_for_data_insight(df, user_question):
    row_count = len(df)
    column_count = len(df.columns)
    columns = df.columns.to_list()
    data_summary = df.describe(include='all').to_string()

    prompt = f"""
    I have a dataset with {row_count} rows and {column_count} columns.
    The column names are: {columns}.
    Here is a summary of the dataset:

    {data_summary}

    Based on this data, please answer the following question:
    User query: {user_question}

    Provide a concise, data-driven response based on the full dataset.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
# Function to merge multiple datasets and handle column name conflicts
def merge_multiple_datasets(dfs, how='inner', on=None):
    with st.spinner("Merging datasets..."):
        if on is None:
            st.error("Merge key not provided.")
            return None
        try:
            merged_df = dfs[0]
            for i in range(1, len(dfs)):
                merged_df = pd.merge(merged_df, dfs[i], how=how, left_on=on[0], right_on=on[i],
                                     suffixes=('', f'_df{i + 1}'))
            st.success("Datasets merged successfully.")
            return merged_df
        except Exception as e:
            st.error(f"An error occurred during merging: {e}")
            return None

# Function to determine if the query is related to chart creation
def is_chart_query(user_query):
    chart_keywords = ['bar chart', 'line chart', 'scatter plot', 'histogram', 'box plot', 'chart', 'plot', 'graph', 'pie', 'Box Plots', 'Heatmap', 'Funnel Charts', 'Waterfall Charts', 'Area Charts']
    return any(keyword in user_query.lower() for keyword in chart_keywords)

# Main Streamlit app
st.markdown(
    "<h1 style='display:inline;'>ClinicalPath Insights<sup style='font-size:0.3em; position: relative; top: -2.0em; font-weight:normal;'> (AI Powered ✨)</sup></h1>",
    unsafe_allow_html=True,
)

# Initialize session state for messages if not already initialized
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for Merge Options before Upload
merge_option = st.sidebar.radio("Do you want to merge multiple datasets?", ("No", "Yes"))

# Sidebar for File Uploads
st.sidebar.title("Upload Datasets or PDFs")
files = st.sidebar.file_uploader("Upload datasets (CSV, Excel) or PDF files", type=['csv', 'xlsx', 'pdf'], accept_multiple_files=True, key="files")

# Display the uploaded file previews in a collapsible format
#st.write("Data Previews")

dfs = []
pdfs = []
if files:
    for file in files:
        with st.expander(f"Preview of {file.name}", expanded=False):
            if file.name.endswith(('csv', 'xlsx')):
                df = load_data(file)
                if df is not None:
                    st.dataframe(df.head())
                    dfs.append(df)
            elif file.name.endswith('pdf'):
                pdf_text = load_data(file)
                if pdf_text is not None:
                    pdfs.append(pdf_text)
                    st.text_area(f"Text extracted from {file.name}:", pdf_text[:500] + "...")

# Handle merging multiple datasets
if merge_option == "Yes" and len(dfs) > 1:
    merge_columns = [st.sidebar.selectbox(f"Select the column to merge on from {file.name}", df.columns) for file, df in zip(files, dfs)]
    merge_type = st.sidebar.selectbox("Select the type of merge:", ["inner", "outer", "left", "right"])

    if st.sidebar.button("Merge Datasets"):
        st.session_state.merged_df = merge_multiple_datasets(dfs, how=merge_type, on=merge_columns)

# Check if merged_df is in session state or use a single dataset if available
if 'merged_df' in st.session_state and st.session_state.merged_df is not None:
    active_df = st.session_state.merged_df
elif len(dfs) == 1:
    active_df = dfs[0]
else:
    active_df = None

# Summarize PDF files
if pdfs:
    for i, pdf_text in enumerate(pdfs):
        if st.button(f"Summarize PDF: {file.name}"):
            summary = summarize_pdf_text(pdf_text)
            if summary:
                st.write(f"Summary of PDF: {file.name}:")
                st.write(summary)

# Ensure the prompt box is always visible for continuous queries using st.chat_input
# Ensure the prompt box is always visible for continuous queries using st.chat_input
if active_df is not None:
    if len(dfs) > 1 and 'merged_df' in st.session_state and st.session_state.merged_df is not None:
        with st.expander("Preview of Merged Dataset:", expanded=False):
            st.dataframe(st.session_state.merged_df.head())
    else:
        # Display the single dataset preview if only one dataset is uploaded
        st.expander("Preview of Uploaded Dataset:", expanded=True)
        #st.dataframe(active_df.head())

    # Single chat_input box for both chart queries and data-related questions
    if prompt := st.chat_input("Enter your query (for chart creation or data-related questions):"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Check if the query is related to chart creation
        if is_chart_query(prompt):
            # Generate chart based on the active dataset (either merged or single)
            chart_code = ask_azure_openai_for_chart(active_df, prompt)
            st.write("Generated Chart Code:")
            st.code(chart_code)

            if chart_code:
                chart_code_clean = clean_generated_code(chart_code)
                execute_chart_code(chart_code_clean, active_df, prompt)

        else:
            # Treat the query as a data-related question
            data_answer = ask_azure_openai_for_data_insight(active_df, prompt)
            if data_answer:
                # Store the question and answer
                if 'data_insights' not in st.session_state:
                    st.session_state.data_insights = []
                st.session_state.data_insights.append((prompt, data_answer))

        # Display all previously generated charts and their prompts
        if 'charts' in st.session_state:
            st.subheader("Generated Charts:")
            for img, prompt in st.session_state.charts:
                st.markdown(f"**Chart Prompt:** {prompt}")
                st.image(img, use_column_width=True)  # Display the chart image

        # Display all data-related questions and their answers
        if 'data_insights' in st.session_state:
            st.subheader("Data Insights:")
            for query, answer in st.session_state.data_insights:
                st.markdown(f"**Data Question:** {query}")
                st.markdown(f"**Data Answer:** {answer}")

# Option for users to reset the interface
if st.sidebar.button("Clear All"):
    st.session_state.clear()
