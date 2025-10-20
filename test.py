import sys
import os
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

SYSTEM_PROMPT = """The user's favorite color is purple."""
# USER_PROMPT = """What is my favorite color?"""
USER_PROMPT = """Write a poem about my favorite color."""

print("\nGemini:")
key = os.environ.get("GEMINI_API_KEY")
if key:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=200
            # maxOutputTokens=200
        ),
        contents=USER_PROMPT
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