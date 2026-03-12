'''Chat bot demo using a llama.cpp server with the OpenAI-compatible API.

Start llama-server with the GPT-OSS-120B MoE model, splitting expert layers to CPU:

$ llama-server \
    -m model.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 0 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "$GPT_API_KEY"
'''

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
api_key = os.environ.get('PERDRIZET_API_KEY', 'dummy')
server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

# Build the base URL: use HTTPS for remote servers, HTTP for localhost
if server.startswith('localhost') or server.startswith('127.'):
    base_url = f'http://{server}/v1'
else:
    base_url = f'https://{server}:9472/v1'
temperature = 0.7

system_prompt = (
    'Reasoning: low\n\n'
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible.'
)

# Initialize the OpenAI client pointing at the llama.cpp server
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Get the model name from the server
models = client.models.list()
model = models.data[0].id

# Start conversation history with system prompt
history = [{'role': 'system', 'content': system_prompt}]


def main():
    '''Main conversation loop.'''

    print(f'Connected to GPT server at {base_url}')
    print(f'Model: {model}')
    print('Type "exit" to quit.\n')

    # Loop until user types 'exit'
    while True:

        # Get text input from the user
        user_input = input('User: ')

        # If the user types 'exit', break the loop and end the conversation
        if user_input == 'exit':
            break

        # Add the user's message to the conversation history
        history.append({'role': 'user', 'content': user_input})

        # Stream the response so tokens appear as they are generated
        stream = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            stream=True,
        )

        print(f'\n{model}: ', end='', flush=True)

        assistant_message = ''

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end='', flush=True)
                assistant_message += token

        print('\n')

        # Add the model's response to the conversation history
        history.append({'role': 'assistant', 'content': assistant_message})


# Main entry point
if __name__ == '__main__':
    main()
