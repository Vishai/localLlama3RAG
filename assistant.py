import ollama
import chromadb
import psycopg
import ast
from psycopg.rows import dict_row
from tqdm import tqdm
from colorama import Fore

client = chromadb.Client()
system_prompt = (
    'You are an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)
convo = [{'role': 'system', 'content': system_prompt}]

DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'vishai',
    'password': 'BJKPEa090!',
    'host': 'localhost',
    'port': '5432'
}

def connect_db():
    try:
        # print("Attempting to connect with these parameters:")
        # for key, value in DB_PARAMS.items():
        #     if key != 'password':
        #         print(f"{key}: {value}")
        #     else:
        #         print("password: [REDACTED]")
        
        conn = psycopg.connect(**DB_PARAMS)
        print(f"Successfully connected to database: {conn.info.dbname} on {conn.info.host}")
        print(f"Connected as user: {conn.info.user}")
        return conn
    except Exception as e:
        print(f"Failed to connect to database. Error: {e}")
        return None

def test_db_connection():
    try:
        conn = connect_db()
        if conn:
            print("Database connection successful.")
            conn.close()
    except Exception as e:
        print(f"Database connection failed: {e}")

def check_postgres_db():
    postgres_params = DB_PARAMS.copy()
    postgres_params['dbname'] = 'postgres'
    postgres_params['user'] = 'postgres'
    try:
        conn = psycopg.connect(**postgres_params)
        with conn.cursor() as cursor:
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases = cursor.fetchall()
            print("Databases in postgres:")
            for db in databases:
                print(f"- {db[0]}")
            
            if 'memory_agent' in [db[0] for db in databases]:
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_catalog = 'memory_agent';")
                tables = cursor.fetchall()
                print("Tables in memory_agent database:")
                for table in tables:
                    print(f"- {table[0]}")
        conn.close()
    except Exception as e:
        print(f"Failed to connect to postgres database. Error: {e}")

def fetch_conversations():
    try:
        conn = connect_db()
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM conversations')
            conversations = cursor.fetchall()
        conn.close()
        return conversations
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        return []

def create_vector_db(conversations):
    vector_db_name = "conversations"

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]
        
        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def create_queries(prompt):
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations you have ever had with the user. '
        'With first principles create a Python list of queries to search the embeddings database for any data that would be necessary '
        'to have access to in order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursuasive request for them to lower my monthly rate.'},
        {'role': 'assistant', 'content': '["What is the users name?", "What is the users current auto insurance provider?", "What is the monthly rate the user currently pays for auto insurance?"]'},
        {'role': 'user', 'content': 'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3 instead of the openai tts api?'},
        {'role': 'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "OpenAI TTS", "openai speak"]'},
        {'role': 'user', 'content': prompt}
    ]
    
    response = ollama.chat(model='llama3', messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]} \n')
    
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]

def classify_embedding(query, context):
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly related respond "no". '
        'Do not respond "yes" unless the content is highly relevant to the search query.'
    )
    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are Ai Austin. How can I help today Austin?'},
        {'role': 'assistant', 'content': 'yes'},
        {'role': 'user', 'content': f'SEARCH QUERY: Llama3 Python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant on Apple iOS and Mac OS. The voice assistant is designed to take voice prompts and help the user complete simple tasks on the device.'},
        {'role': 'assistant', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    
    response = ollama.chat(model='llama3', messages=classify_convo)
    
    return response['message']['content'].strip().lower()

def retrieve_embeddings(queries, results_per_query=2, confidence_threshold=0.8):
    embeddings = set()
    
    for query in tqdm(queries, desc="Processing queries to vector database"):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']

        collection = client.get_collection(name='conversations')
        results = collection.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]
        distances = results['distances'][0]
        
        for best, distance in zip(best_embeddings, distances):
            confidence = 1 - distance  # Assuming distance is normalized between 0 and 1
            if confidence >= confidence_threshold and 'yes' in classify_embedding(query=query, context=best) and best not in embeddings:
                embeddings.add(best)
                
    return embeddings

def store_conversation(prompt, response):
    try:
        conn = connect_db()
        if conn is None:
            print("Failed to connect to database. Cannot store conversation.")
            return
        with conn.cursor() as cursor:
            cursor.execute('INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s) RETURNING id', (prompt, response))
            new_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Conversation stored successfully with ID: {new_id}")
    except Exception as e:
        print(f"Error storing conversation: {e}")
    finally:
        if conn:
            conn.close()

def remove_last_conversation():
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()
        print("Last conversation removed successfully.")
    except Exception as e:
        print(f"Error removing last conversation: {e}")
    finally:
        if conn:
            conn.close()

def stream_response(prompt):
    # Always try to retrieve relevant information
    queries = create_queries(prompt)
    embeddings = retrieve_embeddings(queries)
    
    if embeddings:
        relevant_info = f"Retrieved information: {embeddings}\n\n"
        system_message = f"Use the following retrieved information to inform your response. If the information is not relevant or sufficient, state that you don't have enough information to answer accurately: {relevant_info}"
    else:
        system_message = "You don't have any specific information about this query. If you can't answer accurately, please state that you don't have enough information."

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': prompt}
    ]

    stream = ollama.chat(model='llama3', messages=messages, stream=True)
    response = ''
    print(Fore.LIGHTGREEN_EX + '\nASSISTANT:')
    
    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    
    print('\n')
    store_conversation(prompt=prompt, response=response)
    convo.append({'role': 'assistant', 'content': response})

def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append({'role': 'user', 'content': f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(Fore.YELLOW + f'\n{len(embeddings)} message:response embeddings added for context.')

def view_recent_conversations(limit=5):
    try:
        conn = connect_db()
        print(f"Connected to database: {conn.info.dbname} on {conn.info.host}")
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM conversations ORDER BY timestamp DESC LIMIT %s', (limit,))
            conversations = cursor.fetchall()
        print(f"Retrieved {len(conversations)} conversations")
        for conv in conversations:
            print(f"ID: {conv['id']}, Timestamp: {conv['timestamp']}")
            print(f"Prompt: {conv['prompt']}")
            print(f"Response: {conv['response']}")
            print("-" * 50)
    except Exception as e:
        print(f"Error viewing conversations: {e}")
    finally:
        if conn:
            conn.close()

def show_table_structure():
    try:
        conn = connect_db()
        print(f"Connected to database: {conn.info.dbname} on {conn.info.host}")
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns
                WHERE table_name = 'conversations'
            """)
            columns = cursor.fetchall()
            print("Table structure for 'conversations':")
            for column in columns:
                print(f"{column[0]}: {column[1]}")
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            print(f"\nTotal number of rows in 'conversations': {count}")
    except Exception as e:
        print(f"Error showing table structure: {e}")
    finally:
        if conn:
            conn.close()
            
def check_db_details():
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_schema(), current_user")
            db, schema, user = cursor.fetchone()
            print(f"Current database: {db}")
            print(f"Current schema: {schema}")
            print(f"Current user: {user}")
            
            cursor.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema != 'pg_catalog' AND table_schema != 'information_schema'")
            tables = cursor.fetchall()
            print("Tables in current database:")
            for schema, table in tables:
                print(f"- {schema}.{table}")
        conn.close()
    except Exception as e:
        print(f"Error checking database details: {e}")

# Main execution
if __name__ == "__main__":
    test_db_connection()
    check_postgres_db()
    conversations = fetch_conversations()
    create_vector_db(conversations)

    while True:
        prompt = input(Fore.WHITE + 'USER: \n')
        
        if prompt[:7].lower() == '/recall':
            recall(prompt = prompt[7:].strip())
            stream_response(prompt=prompt[7:].strip())
        elif prompt[:9].lower() == '/memorize':
            store_conversation(prompt = prompt[9:].strip(), response='Memory stored.')
            print('\n')
        elif prompt[:7].lower() == '/forget':
            remove_last_conversation()
            convo = convo[:-2]
            print('\n')
        elif prompt[:5].lower() == '/view':
            view_recent_conversations()
        elif prompt[:10].lower() == '/structure':
            show_table_structure()
        else:
            stream_response(prompt=prompt)