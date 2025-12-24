import streamlit as st
import langchain
from langchain.chat_models import init_chat_model
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="Chess AI Assistant", page_icon="♟️")
st.title("♟️ Chess Analysis AI")

# 2. Securely get API Key
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("API key not found. Please set it in .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

# 3. User Input
# We use a form so the app doesn't reload on every keystroke, only on Submit
with st.form("my_form"):
    default_prompt = """I played Black in Chess. Here are the first few moves. 
d4 - Nf6, f3 - Nc6, e3 - e5, d5 - Na5
What is the incorrect move that I made? 
Can you explain how I should approach this game"""
    
    user_prompt = st.text_area("Enter your prompt:", value=default_prompt, height=150)
    submitted = st.form_submit_button("Submit")

# 4. Logic & Response
if submitted and user_prompt:
    with st.spinner("AI is thinking..."):
        try:
            # Initialize Model
            model = init_chat_model(model="gpt-5-nano", 
                                    model_provider="openai", 
                                    api_key=OPENAI_API_KEY)
            
            # Get Response
            response = model.invoke(user_prompt)
            output_text = response.content
            
            # Display Response
            st.success("Analysis Complete!")
            st.markdown("### AI Response:")
            st.write(output_text)
            
            # 5. Create CSV for Download
            # We create a simple DataFrame with the prompt and response
            df = pd.DataFrame({
                "Prompt": [user_prompt],
                "Response": [output_text]
            })
            
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Response as CSV",
                data=csv,
                file_name="chess_analysis.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")