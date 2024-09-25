import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import streamlit as st
import pandas as pd
import pandasql as ps

checkpoint = "suriya7/Gemma2B-Finetuned-Sql-Generator"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = checkpoint

# Load the model with quantization and use GPU
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

def sanitize_sql(sql):
    # Basic sanitization to ensure there are no malicious inputs
    sql = sql.strip().strip(';')
    return sql

# Define your prompt for SQL generation

def sanitize_sql(sql):
    # Basic sanitization to ensure there are no malicious inputs
    sql = sql.strip().strip(';')
    return sql

def get_gemma_response(prompt):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the output response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=100,  # You can adjust the max_length depending on your needs
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated tokens to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response


# Define your prompt for SQL generation
def prompt_gemma(question):
    start_prompt = f"""
        You are an expert in converting natural language questions into SQL queries. 
        The SQL database schema is provided below:
        Table: players
        Columns:
            - Name (TEXT)
            - Age (INTEGER)
            - Position (TEXT)
            - Nationality (TEXT)
            - Matches_Played (INTEGER)
            - Goals_Scored (INTEGER)
            - Assists (INTEGER)
            - Yellow_Cards (INTEGER)
            - Red_Cards (INTEGER)
            - Height_CM (INTEGER)
            - Weight_KG (INTEGER)
        It is important that the column names used in your SQL query match exactly with the column names in the schema. 
        Ensure that any aliases you use for columns in the SQL query also correctly match the schema names or are clearly defined.
        Here are some examples:
        Question: List all players
        SQL Query: SELECT * FROM players;
        Question: What are the names and ages of players who scored more than 20 goals?
        SQL Query: SELECT Name, Age FROM players WHERE Goals_Scored > 20;
        Question: Find the player with the highest number of assists
        SQL Query: SELECT Name, Assists FROM players ORDER BY Assists DESC LIMIT 1;
        Question: Which players are taller than 180 cm and weigh less than 80 kg?
        SQL Query: SELECT Name FROM players WHERE Height_CM > 180 AND Weight_KG < 80;
        Question: What is the BMI of players?
        SQL Query: SELECT Weight_KG / ((Height_CM / 100.0) * (Height_CM / 100.0)) AS BMI FROM players;
        Guidelines:
        1. Ensure the SQL code is syntactically correct and does not include delimiters like `;`.
        2. Avoid SQL keywords or delimiters in the output.
        3. Handle different variations of questions accurately.
        4. The SQL code should be valid, executable, and not contain unnecessary delimiters.
        quetion:{question}
        sql query:
        """
    return  get_gemma_response(start_prompt)

def generate_conversational_response(question):
    # Example response generation for conversational queries
    custom_prompt = f"""
    You are an assistant that responds to basic conversational queries in a friendly and helpful manner.
    If a user greets you or asks how you are, respond with a polite and friendly reply.
    If the user asks for specific information, a specific topic, a question, or an order, respond with 
    "I am here to chat with you, but I do not have the information you are looking for."
    
    Example interactions:
    User: "Hey"
    Assistant: "Hello! How can I help you today?"

    User: "How are you?"
    Assistant: "I'm doing well, thank you! How can I assist you today?"

    User: "Can you help me?"
    Assistant: "Of course! What do you need help with?"

    User: "Tell me about the latest news."
    Assistant: "I am here to chat with you, but I do not have the information you are looking for."

    User: "What's the weather like?"
    Assistant: "I am here to chat with you, but I do not have the information you are looking for."

    User: "Order a pizza for me."
    Assistant: "I am here to chat with you, but I do not have the information you are looking for."

    User: "Tell me a joke."
    Assistant: "I am here to chat with you, but I do not have the information you are looking for."

    User: "Can you tell me about Python programming?"
    Assistant: "I am here to chat with you, but I do not have the information you are looking for."

    Now, respond to the following question:
    User: {question}
    Assistant:
    """
    sql_query = get_gemma_response(custom_prompt.format(question=question), "")
    return sql_query
def respond_to_question(question, dataframe):
    related_keywords = [
        'players', 'name', 'age', 'position', 'nationality', 'matches played', 'goals scored', 'assists',
        'yellow cards', 'red cards', 'height', 'weight', 'goalkeeper', 'BMI', 'player', 'team', 'club',
        'league', 'season', 'appearances', 'clean sheets', 'minutes played', 'injuries', 'transfers',
        'market value', 'contract', 'retired', 'career', 'debut', 'captain', 'vice-captain', 'defender',
        'midfielder', 'forward', 'winger', 'striker', 'coach', 'manager', 'substitute', 'bench', 'starting XI',
        'training', 'tactics', 'formation', 'penalty', 'free kick', 'corner kick', 'throw-in', 'offside',
        'dribbling', 'passing', 'shooting', 'tackling', 'interceptions', 'saves', 'goal-line clearance',
        'headers', 'crosses', 'fouls', 'penalties', 'own goal', 'hat-trick', 'man of the match', 'stadium',
        'attendance', 'fans', 'supporters', 'rivals', 'derby', 'international', 'domestic', 'cup', 'championship',
        'qualifiers', 'friendly', 'pre-season', 'youth academy', 'scout', 'scouting report', 'transfer window',
        'loan', 'buyout clause', 'release clause', 'agent', 'negotiations', 'signing bonus', 'wages', 'salary',
        'sponsorship', 'endorsement', 'press conference', 'media', 'interview', 'statistic', 'analysis', 'performance'
    ]

    if any(keyword in question.lower() for keyword in related_keywords):
        # Generate the SQL query
        sql_query = prompt_gemma(question)
        sql_query = sql_query.strip('"')

        # Check if the query contains "SELECT"
        if "SELECT" in sql_query.upper():
            try:
                # Execute the SQL query
                result = ps.sqldf(sql_query, {'players': dataframe})

                if len(result) > 0:
                    return result, 'dataframe'
                else:
                    # Convert result to string if the length is 2 or less
                    result_str = "there is no data about this information"
                    return result_str, 'text'
            except Exception as e:
                return f"Sorry I do not have the information you are looking for", 'error'
        else:
            return "Generated SQL query is not a valid SELECT statement.", 'error'
    else:
        # Generate a conversational response
        response = generate_conversational_response(question)
        return response, 'text'
    

one_shot_prompt = f"""
You are an expert in translating SQL query results into clear and understandable responses for users.

Given the following information:
- The question asked by the user: {{question}}
- The result from the SQL query represented in a pandas DataFrame: {{dataframe_result}}

Please generate a concise and easy-to-understand response that directly answers the user's question. Avoid technical jargon and focus on clarity.

Response:
"""

    
# Set the page title and icon
st.set_page_config(page_title="MyTeamBot", page_icon="⚽")

# Title and description
st.title("⚽ MyTeamBot: Your Football Management Assistant")
st.write("""
Welcome to **MyTeamBot**! 🎉 Your personal assistant for all things related to Moroccan Botola league players.
Easily access player statistics, performance insights, and more with just a question. Let's get started! 🚀
""")


# -------------------------------------------------------------------------------------------------------

# Set page configuratio
# Custom CSS for better styling
st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: #262730;
        border: 1px solid #1f77b4;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .css-1aumxhk h4 {
        color: #1f77b4;
    }
    .css-1aumxhk p {
        color: #fafafa;
    }
    .css-1aumxhk input {
        background-color: #0e1117;
        color: #fafafa;
        border: 1px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

players = pd.read_csv("football_players_n.csv")

# Input area for the user to ask a question
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle the question and respond accordingly

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Message MyTeam Bot !"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):

        # Generate response using `respond_to_question` function
        response, response_type = respond_to_question(prompt, players)


        # Update the assistant's message with the response
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response
        if response_type == 'dataframe':
            final_prompt = one_shot_prompt.format(question=prompt, dataframe_result=response)
            st.markdown(get_gemma_response(final_prompt))
        else:
            st.markdown(response)

