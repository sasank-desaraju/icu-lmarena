import sys
import os
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

SYSTEM_PROMPT = """The user's favorite color is purple."""
# USER_PROMPT = """What is my favorite color?"""
USER_PROMPT = """Write a short poem about my favorite color."""

gemini_key = os.environ.get("GEMINI_API_KEY")

# Test Gemini conversation


# Use this to only test the part that we want
# sys.exit(0)
conversation_history = None
system_prompt = SYSTEM_PROMPT
user_prompt = USER_PROMPT
MAX_TOKENS = 200
print(f"conversation_history: {conversation_history}")
print(f"system_prompt: {system_prompt}")
print(f"user_prompt: {user_prompt}")
client = genai.Client(api_key=gemini_key)
print("made client")

# Build conversation history for Gemini
test_contents = []
if conversation_history:
    print(f"type of conversation_history: {type(conversation_history)}")
    print(f"conversation history:\n", conversation_history)
    for step in conversation_history:
        test_contents.append(types.Content(role="user", parts=[types.Part(text=step['user_prompt'])]))
        test_contents.append(types.Content(role="model", parts=[types.Part(text=step['assistant_response'])]))

# contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))
test_contents = [
    types.Content(role="user", parts=[types.Part(text=user_prompt)])
]
print(f"contents are:\n", test_contents)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=MAX_TOKENS,
        thinking_config=types.ThinkingConfig(
            thinking_budget=100,
            include_thoughts=True)
    ),
    contents=test_contents
)
print(response)
print(response.text)

sys.exit(0)

# Test Gemini with conversation history
print("\nGemini with conversation:")
key = os.environ.get("GEMINI_API_KEY")
if key:
    client = genai.Client(api_key=key)
    
    # Simulate conversation history
    # contents = [
    #     types.Content(role="user", parts=[types.Part(text="What is my favorite color? I love acrostic poems, by the way.")]),
    #     types.Content(role="model", parts=[types.Part(text="Your favorite color is purple.")]),
    #     types.Content(role="user", parts=[types.Part(text=USER_PROMPT)])
    # ]
    contents = []
    contents.append(types.Content(role="user", parts=[types.Part(text="What is my favorite color? I love acrostic poems, by the way.")]))
    # contents.append(types.Content(role="model", parts=[types.Part(text="Your favorite color is purple.")]))
    # contents.append(types.Content(role="user", parts=[types.Part(text=USER_PROMPT)]))
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=200
        ),
        contents=contents
    )
    print(response.text)
else:
    print("GEMINI_API_KEY not set")

sys.exit(0)

# ChatGPT-5 (using OpenAI)
print("ChatGPT-5:")
key = os.environ.get("OPENAI_API_KEY")
if key:
    client = OpenAI(api_key=key)
    # response = client.chat.completions.create(
    #     model="gpt-5",  # Use available model; adjust as needed
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": USER_PROMPT}
    #     ],
    #     temperature=0.2,
    #     max_completion_tokens=700,
    # )
    # print(response.choices[0].message.content)
    response = client.responses.create(
        model="gpt-5",  # Use available model; adjust as needed
        reasoning={"effort": "low"},
        input=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    )
    print(response.output_text)
else:
    print("OPENAI_API_KEY not set")

print("\nClaude:")
key = os.environ.get("ANTHROPIC_API_KEY")
if key:
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    print(response.content[0].text)
else:
    print("ANTHROPIC_API_KEY not set")

print("\nGemini:")
key = os.environ.get("GEMINI_API_KEY")
if key:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
        contents=USER_PROMPT
    )
    print(response.text)
else:
    print("GEMINI_API_KEY not set")

print("\nGrok:")
key = os.environ.get("XAI_API_KEY")
if key:
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-4",  # Adjust model name
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.2,
        max_tokens=700,
    )
    print(response.choices[0].message.content)
else:
    print("XAI_API_KEY not set")