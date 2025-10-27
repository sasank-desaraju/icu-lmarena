# Instructions for what this app should be


## Old Instructions from a Previous Day
Create a Streamlit app similar to LMArena.
I want to compare how 4 frontier LLMs compare in response to a single user input prompt.
ChatGPT-5, Claude 4.5 Sonnet, Gemini 2.0, and Grok 4.
All 4 side by side but blinded and randomly ordered every time.
I want 4 5-point Likert scales under each LLM's response (Accuracy, Instruction-following, Style, Helpfulness).
I want to deploy this on Streamlit Cloud and have a few things set up in the project folder already like API keys.
I want a single password to allow use of the app (I'll share with my colleagues) and I want the user to put in their name.
I want to record evey input, set of 4 outputs, Likert scores, user name.
I want user to be able to do multiple of these "one question and evaluate answers" per session so I want a reset button that refreshes chat history.

## Newest Instructions and Goals
- Create Dummy Runners that return fixed text for each model instead of calling the APIs. This is to allow testing the app without incurring API costs. If the flag DUMMY_RUNNERS is set to True, use the dummy runners. DONE
- Fix bug where Comments are not recorded DONE
- Fix bug where Likert scales are not recorded when left at their default value of 3 DONE
- Make the system prompt visible to the user and editable by them. Keep the default system prompt as is but the user can change it before submitting the prompt. DONE