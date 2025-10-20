import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit app title
st.title("LMArena: LLM Comparison App")

# Description
st.markdown("""
Compare ChatGPT-5, Gemini 2.5, Claude Sonnet 4.5, and Grok 4 across four dimensions: 
**Accuracy**, **Helpfulness**, **Instruction-following**, and **Style**. 
Rate each model using a Likert scale (1-5) and submit to see the results.
""")

# Define models and evaluation criteria
models = ["ChatGPT-5", "Gemini 2.5", "Claude Sonnet 4.5", "Grok 4"]
criteria = ["Accuracy", "Helpfulness", "Instruction-following", "Style"]
likert_scale = [1, 2, 3, 4, 5]  # 1 = Poor, 5 = Excellent

# Initialize session state to store ratings
if 'ratings' not in st.session_state:
    st.session_state.ratings = []

# Input form for ratings
st.subheader("Rate the Models")
with st.form(key="rating_form"):
    ratings = {}
    for model in models:
        st.markdown(f"### {model}")
        ratings[model] = {}
        for criterion in criteria:
            ratings[model][criterion] = st.selectbox(
                f"{criterion} for {model}",
                options=likert_scale,
                index=2,  # Default to 3 (neutral)
                key=f"{model}_{criterion}"
            )
    submit_button = st.form_submit_button(label="Submit Ratings")

# Process form submission
if submit_button:
    # Store ratings
    rating_entry = {"Model": [], "Criterion": [], "Rating": []}
    for model in models:
        for criterion in criteria:
            rating_entry["Model"].append(model)
            rating_entry["Criterion"].append(criterion)
            rating_entry["Rating"].append(ratings[model][criterion])
    st.session_state.ratings.append(pd.DataFrame(rating_entry))
    st.success("Ratings submitted!")

# Display and visualize results if ratings exist
if st.session_state.ratings:
    st.subheader("Evaluation Results")
    
    # Combine all ratings into a single DataFrame
    all_ratings = pd.concat(st.session_state.ratings, ignore_index=True)
    
    # Display raw ratings
    st.markdown("### Raw Ratings")
    st.dataframe(all_ratings)
    
    # Calculate average ratings per model and criterion
    avg_ratings = all_ratings.groupby(["Model", "Criterion"])["Rating"].mean().unstack()
    
    # Display average ratings table
    st.markdown("### Average Ratings")
    st.dataframe(avg_ratings)
    
    # Visualize ratings with a bar chart
    st.markdown("### Visualization")
    fig = px.bar(
        all_ratings,
        x="Model",
        y="Rating",
        color="Criterion",
        barmode="group",
        title="Model Comparison Across Criteria",
        labels={"Rating": "Average Rating (1-5)"},
        height=500
    )
    st.plotly_chart(fig)

    # Radar chart for a comprehensive view
    st.markdown("### Radar Chart Comparison")
    radar_data = avg_ratings.reset_index().melt(id_vars="Model", value_vars=criteria, var_name="Criterion", value_name="Rating")
    fig_radar = px.line_polar(
        radar_data,
        r="Rating",
        theta="Criterion",
        color="Model",
        line_close=True,
        title="Model Performance (Radar Chart)",
        height=500
    )
    st.plotly_chart(fig_radar)

# Instructions for users
st.markdown("""
### Instructions
1. Rate each model for **Accuracy**, **Helpfulness**, **Instruction-following**, and **Style** using the 1-5 Likert scale (1 = Poor, 5 = Excellent).
2. Submit your ratings to see the results in a table and visualizations.
3. Multiple submissions are averaged for display.
""")