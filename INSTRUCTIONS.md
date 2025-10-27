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
- Create Dummy Runners that return fixed text for each model instead of calling the APIs. This is to allow testing the app without incurring API costs. If the flag DUMMY_RUNNERS is set to True, use the dummy runners. DONE
- Fix bug where Comments are not recorded DONE
- Fix bug where Likert scales are not recorded when left at their default value of 3 DONE
- Make the system prompt visible to the user and editable by them. Keep the default system prompt as is but the user can change it before submitting the prompt. DONE

- Create a new Streamlit app at classification_app.py
    - This app should take inspiration from streamlit_app.py.
    - The context is that a research will input a simple "generic" question about a classification task (e.g. "make a classification question about the epidemiology of ARDS"). This app will help a researcher guide an LLM to generate actual sample classification questions from this generic prompt.
    - The app should use the same 4 LLMs (ChatGPT-5, Claude 4.5 Sonnet, Gemini 2.0, and Grok 4) and offer the user a choice of which single LLM to use in a dropdown. Only 1 LLM per session/prompt.
    - This will be modelled as a guided conversation with the LLM. System prompt will be prepopulated but editable by the user. All user prompts will also be editable by the user.
    - First user prompt will be just the generic prompt.
    - The LLM will respond with a set of classification dimensions and options that are appropriate for that question/prompt. e.g. for ARDS epidemiology, the dimensions could be age(<18, 18-40, 40-60, >60), sex(male, female), comorbidities (none, diabetes, hypertension, etc), severity (mild, moderate, severe).
    - The second prompt (prepopulated but editable) will ask the LLM to generate the sample classification questions based on the dimensions and options it just generated. e.g. one of them would be "Classify whether ARDS is most common in people age <18, 18-40, 40-60, or >60".
    - The third prompt (prepopulated but editable) will ask the LLM to generate answers for each classification question it just generated. e.g. for the question above, the answer might be "ARDS is most common in people aged 40-60". Ofc each answer will be based on actual data we'll request that in the prompt.
    - Each prompt will be shown in the app with the LLM response below it. It will be modeled as a chat conversation in the back end.
    - This will all be saved to a database similar to streamlit_app.py with rater name, prompt, system prompt, LLM used, and all 3 prompts and responses saved.
    - We might add evaluation later but for now just focus on getting the generation flow working.

## Newest Instructions and Goals
- Add temperature and top p controls for each model in the sidebar.
- Maybe record reasoning traces of each model response if feasible.
- Add max tokens and max reasoning tokens controls for each model in the sidebar.