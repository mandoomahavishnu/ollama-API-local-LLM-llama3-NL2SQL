import streamlit as st
import mysql.connector
import re
import ollama
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from few_shot_examples import few_shot_examples

# Function to download NLTK stopwords if not present
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Ensure NLTK data directories exist
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Call the function to download NLTK resources
download_nltk_resources()

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="your host",
        user="your username",
        password="password",
        database="your database"  # Change this to your database name
    )

# Get database schema for specific tables
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_database_schema(table_names=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if table_names:
        table_names_str = ", ".join(f"'{table}'" for table in table_names)
        query = f"""
            SELECT table_name, column_name, data_type, is_nullable, column_key, column_default, extra, column_comment
            FROM information_schema.columns 
            WHERE table_schema = 'your database' AND table_name IN ({table_names_str})
        """
    else:
        query = """
            SELECT table_name, column_name, data_type, is_nullable, column_key, column_default, extra, column_comment
            FROM information_schema.columns 
            WHERE table_schema = 'your database'
        """
    cursor.execute(query)
    schema = cursor.fetchall()
    conn.close()
    return schema

# Load table descriptions from the new CSV
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_table_descriptions(file_path):
    return pd.read_csv(file_path)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define specific table names to use
table_names = [
    # list of table names used
    'table1', 'table2', 'table3'
]

# Load table descriptions
descriptions = load_table_descriptions('path to your table description file.csv')  # Adjust the file path as needed

# Create FAISS index for few_shot_examples
def create_faiss_index(examples):
    example_texts = [ex['natural_language'] for ex in examples]
    example_embeddings = model.encode(example_texts)
    index = faiss.IndexFlatL2(example_embeddings.shape[1])
    index.add(np.array(example_embeddings, dtype=np.float32))
    return index, example_embeddings

# Create FAISS index
faiss_index, example_embeddings = create_faiss_index(few_shot_examples)

# Select the most relevant examples using FAISS
def select_relevant_examples(query, index, examples, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    return [examples[i] for i in indices[0]]

# Function to identify relevant tables
keyword_table_map = {
    
    "inventory": ["inventory"],
    "sales": ["salesorder", "salesorder1"],
    "crew": ["warehouse"],
    "master qty": ["products"],
    "routing_date": ["warehouse"],
    
}

def identify_tables(query):
    relevant_tables = set()
    for keyword, tables in keyword_table_map.items():
        if keyword in query.lower():
            relevant_tables.update(tables)
    st.write(f"Identified tables: {relevant_tables}")
    return list(relevant_tables)

# Function to select relevant columns from identified tables
def select_columns(tables, schema):
    relevant_columns = []
    for table in tables:
        for row in schema:
            if row[0] == table:
                relevant_columns.append((table, row[1], row[7]))  # Include column comments
    st.write(f"Relevant columns: {relevant_columns}")
    return relevant_columns

# Generate dynamic prompt using selected tables and columns and examples
def generate_dynamic_prompt(query, relevant_columns, examples, history=None, error_message=None):
    guidelines = """
    Guidelines for SQL query generation:
    1. Ensure Efficiency and Performance: Opt for JOINs over subqueries where possible, use indexes effectively, and mention any specific performance considerations to keep in mind.
    2. Adapt to Specific Analytical Needs: Tailor WHERE clauses, JOIN operations, and aggregate functions to precisely meet the analytical question being asked.
    3. Complexity and Variations: Include a range from simple to complex queries, illustrating different SQL functionalities such as aggregate functions, string manipulation, and conditional logic.
    4. Handling Specific Cases: Provide clear instructions on managing NULL values, ensuring date ranges are inclusive, and handling special data integrity issues or edge cases.
    5. Explanation and Rationale: After each generated query, briefly explain why this query structure was chosen and how it addresses the analytical need, enhancing understanding and ensuring alignment with requirements.
    """
    schema_info = "\n".join([f"Table: {table}, Column: {column}, Comment: {comment}" for table, column, comment in relevant_columns])
    examples_info = "\n".join([
        f"Example {i+1}:\nNatural language: {ex['natural_language']}\nSQL query: {ex['sql_query']}" 
        for i, ex in enumerate(examples)
    ])
    
    if history:
        recent_history = history[-2:]  # Only take the last user and bot messages
        history_info = "\n".join([f"User: {item['content']}\nBot: {item.get('bot', '')}\n" for item in recent_history])
    else:
        history_info = ""
    
    prompt = (
        f"Here is the schema information:\n{schema_info}\n\n"
        "Here are some examples:\n" +
        examples_info + "\n\n" +
        guidelines +
        f"\n\nHere is the conversation history:\n{history_info}\n\n"
        f"Now, generate an SQL query for the following natural language request: {query}\n"
    )
    if error_message:
        prompt += f"Note: The previous attempt resulted in the following error: {error_message}\n"
    prompt += "Please generate only the SQL query without any explanation or additional text."
    st.write(f"Generated prompt: {prompt}")
    return prompt

# Generate SQL query from natural language using selected tables and columns
def generate_sql_query_with_tables(query, schema, examples, index, history=None, error_message=None):
    # Identify relevant tables
    relevant_tables = identify_tables(query)
    
    # Select relevant columns
    relevant_columns = select_columns(relevant_tables, schema)
    
    # Select the most relevant examples using FAISS
    relevant_examples = select_relevant_examples(query, index, examples)
    
    # Generate the dynamic prompt using the relevant columns and examples
    prompt = generate_dynamic_prompt(query, relevant_columns, relevant_examples, history, error_message)
    
    # Send the prompt to the API to generate the SQL query
    response_stream = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are an assistant that generates SQL queries for a MySQL database.'},
            {'role': 'user', 'content': prompt}
        ],
        options={
            "temperature": 0.1  # Set the temperature parameter here
        },
        stream=True  # Enable streaming
    )
    
    sql_query = ""
    for response in response_stream:
        if 'message' in response and 'content' in response['message']:
            sql_query += response['message']['content']
    
    sql_query = re.sub(r'```', '', sql_query).strip()  # Clean up backticks and extra spaces
    st.write(f"Generated SQL Query: {sql_query}")
    return sql_query

# Describe table using schema and descriptions
def describe_table(table_name, schema, descriptions):
    table_schema = [row for row in schema if row[0] == table_name]
    table_description = descriptions[descriptions['TABLE'].str.lower() == table_name.lower()]
    
    description_info = []
    for column in table_schema:
        column_name = column[1]
        column_desc = table_description[table_description['COLUMN NAME'].str.lower() == column_name.lower()]
        
        if not column_desc.empty:
            column_description = column_desc.iloc[0]['COLUMN DESCRIPTION']
        else:
            column_description = column[7]  # Use column comment if available
        
        description_info.append({
            'Column Name': column_name,
            'Data Type': column[2],
            'Is Nullable': column[3],
            'Key': column[4],
            'Default': column[5],
            'Extra': column[6],
            'Description': column_description
        })
    
    return description_info

# Display description of table in Streamlit
def display_table_description(table_name, schema, descriptions):
    description_info = describe_table(table_name, schema, descriptions)
    for column_info in description_info:
        st.write(f"Column: {column_info['Column Name']}")
        st.write(f" - Data Type: {column_info['Data Type']}")
        st.write(f" - Is Nullable: {column_info['Is Nullable']}")
        st.write(f" - Key: {column_info['Key']}")
        st.write(f" - Default: {column_info['Default']}")
        st.write(f" - Extra: {column_info['Extra']}")
        st.write(f" - Description: {column_info['Description']}")
        st.write("")

# Execute SQL query
@st.cache_data(ttl=600)  # Cache for 10 minutes to speed up repeated queries
def execute_sql_query(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    columns = cursor.column_names
    conn.close()
    return results, columns

# Translate SQL results to natural language
def results_to_natural_language(results, columns, descriptions):
    if not results:
        return "I couldn't find any results matching your query."

    response = "Here are the results I found:\n\n"
    for row in results:
        row_description = []
        for col, val in zip(columns, row):
            col_description = descriptions[descriptions['COLUMN NAME'].str.lower() == col.lower()]
            if not col_description.empty:
                col_desc = col_description.iloc[0]['COLUMN DESCRIPTION']
            else:
                col_desc = 'No description available'
            row_description.append(f"{col} ({col_desc}): {val}")
        response += " | ".join(row_description) + "\n"
    
    response += "\nYou can download the results as a CSV file using the link below."
    return response

# Create a direct download link in Streamlit
def get_table_download_link(df, file_name="query_results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV file</a>'
    return href

st.title("ASK WAREHOUSE")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []
if "results" not in st.session_state:
    st.session_state.results = None
if "columns" not in st.session_state:
    st.session_state.columns = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ''

# Get database schema
schema = get_database_schema(table_names)

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.context.append(f"User: {user_input}")
        
        max_retries = 5
        error_message = None
        history = st.session_state.messages  # Include the chat history
        for attempt in range(max_retries):
            try:
                # Generate SQL query from natural language using selected tables and columns
                sql_query = generate_sql_query_with_tables(user_input, schema, few_shot_examples, faiss_index, history, error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Generated SQL Query: {sql_query}"})
                st.session_state.context.append(f"Generated SQL Query: {sql_query}")
                
                # Execute SQL query
                results, columns = execute_sql_query(sql_query)
                st.session_state.results = results
                st.session_state.columns = columns
                # Translate results to natural language with descriptions
                natural_language_results = results_to_natural_language(results, columns, descriptions)
                st.session_state.messages.append({"role": "assistant", "content": f"Results in Natural Language: {natural_language_results}"})
                st.session_state.context.append(f"Results in Natural Language: {natural_language_results}")
                break  # Exit the retry loop if successful
            except Exception as e:
                error_message = str(e)
                st.session_state.messages.append({"role": "assistant", "content": f"Attempt {attempt+1} Error: {error_message}"})
                st.session_state.context.append(f"Attempt {attempt+1} Error: {error_message}")
                if attempt >= max_retries - 1:
                    st.session_state.messages.append({"role": "assistant", "content": "Max retries reached. Please refine your request or try again later."})
                    st.session_state.context.append("Max retries reached. Please refine your request or try again later.")
        
        st.session_state.user_input = ''

# Display chat messages in Streamlit
for i, message in enumerate(st.session_state.messages):
    if message['role'] == 'user':
        st.text_area("You:", message['content'], height=50, key=f"user_{i}")
    else:
        st.text_area("Bot:", message['content'], height=200, key=f"bot_{i}")

# Display results and provide download link if results are available
if st.session_state.results is not None and st.session_state.columns is not None:
    df = pd.DataFrame(st.session_state.results, columns=st.session_state.columns)
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

# Add custom CSS to keep the input box at the bottom center
st.markdown(
    """
    <style>
    .stTextInput {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        padding: 10px;
        background-color: white;
        border-top: 1px solid #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input box for the user to type their message
st.text_input("You: ", key="user_input", on_change=send_message, placeholder="Be patient. I am on a slow machine. Be specific. And do not say 'Hello'")
