import ollama
import chromadb
import psycopg
import ast
from psycopg.rows import dict_row
from tqdm import tqdm
from colorama import Fore, init
import sys
import requests
from bs4 import BeautifulSoup

# Initialize colorama for cross-platform colored output
init()

client = chromadb.Client()
system_prompt = (
    'You are an AI assistant for an e-commerce agency managing Amazon sellers. '
    'You have access to client information, store performance, product data, and case histories. '
    'Provide insightful answers and recommendations based on this data.'
)
convo = [{'role': 'system', 'content': system_prompt}]

DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'Vishai',
    'password': 'BJKPEa090!',
    'host': 'localhost',
    'port': '5432'
}

def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn

def fetch_conversations():
    conn = connect_db()
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM conversations')
            conversations = cursor.fetchall()
    finally:
        conn.close()
    return conversations

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
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    
    response = ollama.chat(model='llama3', messages=classify_convo)
    result = response['message']['content'].strip().lower()
    print(f"Classification for query '{query}': {result}")
    return result

def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()
    
    for query in tqdm(queries, desc="Processing queries to vector database"):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']

        collection = client.get_collection(name='conversations')
        results = collection.query(query_embeddings=[query_embedding], n_results=results_per_query)
        
        if not results['documents']:
            continue
        
        best_embeddings = results['documents'][0]
        
        for best in best_embeddings:
            if 'yes' in classify_embedding(query=query, context=best) and best not in embeddings:
                embeddings.add(best)
                
    return embeddings

def store_conversation(prompt, response=None):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
                (prompt, response if response else "User memorized information")
            )
            conn.commit()
    finally:
        conn.close()

def remove_last_conversation():
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
            conn.commit()
    finally:
        conn.close()

def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    stream = ollama.chat(model='llama3', messages=convo, stream=True)
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

def web_search(query):
    # This is a simplified web search function. In practice, you'd use a more robust solution.
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='g')
    return [result.text for result in results[:3]]

def get_similar_cases(case_type, description):
    conn = connect_db()
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('''
                SELECT c.*, cr.resolution_steps, cr.outcome
                FROM cases c
                LEFT JOIN case_resolutions cr ON c.id = cr.case_id
                WHERE c.case_type = %s AND c.status = 'Resolved'
                ORDER BY similarity(c.description, %s) DESC
                LIMIT 3
            ''', (case_type, description))
            similar_cases = cursor.fetchall()
    finally:
        conn.close()
    return similar_cases

def formulate_resolution_steps(case_type, description):
    similar_cases = get_similar_cases(case_type, description)
    web_results = web_search(f"Amazon seller {case_type} resolution")
    
    context = f"Case Type: {case_type}\nDescription: {description}\n\n"
    context += "Similar Resolved Cases:\n"
    for case in similar_cases:
        context += f"- {case['title']}: {case['resolution_steps']}\n"
    context += "\nWeb Search Results:\n"
    for result in web_results:
        context += f"- {result}\n"
    
    prompt = f"{context}\n\nBased on the above information, provide a step-by-step resolution plan for this case."
    
    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])
    
    return response['message']['content']

def create_case(client_id, store_id, case_type, priority, title, description):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO cases (client_id, store_id, case_type, status, priority, title, description)
                VALUES (%s, %s, %s, 'Open', %s, %s, %s)
                RETURNING id
            ''', (client_id, store_id, case_type, priority, title, description))
            case_id = cursor.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return case_id

def update_case(case_id, update_type, description):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO case_updates (case_id, update_type, description)
                VALUES (%s, %s, %s)
            ''', (case_id, update_type, description))
            cursor.execute('UPDATE cases SET updated_at = CURRENT_TIMESTAMP WHERE id = %s', (case_id,))
        conn.commit()
    finally:
        conn.close()
        
        def resolve_case(case_id, resolution_steps, outcome):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO case_resolutions (case_id, resolution_steps, outcome)
                VALUES (%s, %s, %s)
            ''', (case_id, resolution_steps, outcome))
            cursor.execute('UPDATE cases SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s', ('Resolved', case_id))
        conn.commit()
    finally:
        conn.close()

def handle_case_query(prompt, client_id=None):
    if "create case" in prompt.lower():
        case_details = prompt.split(",")
        case_id = create_case(client_id, case_details[0], case_details[1], case_details[2], case_details[3], case_details[4])
        return f"Case created with ID: {case_id}"
    
    elif "update case" in prompt.lower():
        update_details = prompt.split(",")
        update_case(update_details[0], update_details[1], update_details[2])
        return "Case updated successfully"
    
    elif "resolve case" in prompt.lower():
        resolution_details = prompt.split(",")
        resolve_case(resolution_details[0], resolution_details[1], resolution_details[2])
        return "Case resolved successfully"
    
    elif "suggest resolution" in prompt.lower():
        case_details = prompt.split(",")
        resolution_steps = formulate_resolution_steps(case_details[0], case_details[1])
        return f"Suggested Resolution Steps:\n{resolution_steps}"
    
    else:
        return stream_response(prompt)

def print_feedback(message, color=Fore.YELLOW):
    print(f"\n{color}{message}{Fore.RESET}")

def handle_command(command, content):
    if command == '/memorize':
        try:
            store_conversation(prompt=content, response='Memory stored.')
            print_feedback("Memory successfully stored.")
        except Exception as e:
            print_feedback(f"Error storing memory: {str(e)}", Fore.RED)
    elif command == '/recall':
        try:
            recall(prompt=content)
            stream_response(prompt=content)
        except Exception as e:
            print_feedback(f"Error recalling memory: {str(e)}", Fore.RED)
    elif command == '/forget':
        try:
            remove_last_conversation()
            convo.pop()  # Remove the last user message
            if convo[-1]['role'] == 'assistant':
                convo.pop()  # Remove the last assistant message if present
            print_feedback("Last conversation removed from memory.")
        except Exception as e:
            print_feedback(f"Error removing last conversation: {str(e)}", Fore.RED)
    else:
        print_feedback(f"Unknown command: {command}", Fore.RED)

def main():
    print_feedback("Welcome to your AI assistant. Type '/help' for available commands.")
    
    conversations = fetch_conversations()
    create_vector_db(conversations)
    
    while True:
        try:
            user_input = input(Fore.WHITE + 'USER: \n')
            
            if user_input.lower() == '/exit':
                print_feedback("Goodbye!")
                sys.exit(0)
            
            if user_input.lower() == '/help':
                print_feedback("Available commands:\n"
                               "/memorize [content]: Store a new memory\n"
                               "/recall [query]: Recall and respond based on stored memories\n"
                               "/forget: Remove the last stored conversation\n"
                               "/exit: Exit the program")
                continue
            
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                content = parts[1] if len(parts) > 1 else ""
                handle_command(command, content)
            else:
                handle_case_query(user_input)
                
        except Exception as e:
            print_feedback(f"An error occurred: {str(e)}", Fore.RED)

if __name__ == "__main__":
    main()