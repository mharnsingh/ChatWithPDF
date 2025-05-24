import streamlit as st
import requests
import os

# Use the API URL from env variable or default to localhost
agent_api_url = os.getenv("AGENT_API_URL", "http://localhost:8000/query")
clear_memory_url = os.getenv("AGENT_CLEAR_MEMORY_URL", "http://localhost:8000/clear_memory")

st.title("Chat with PDF Demo")

query = st.text_area("Enter your query:", "")

if st.button("Submit Query"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            response = requests.post(agent_api_url, json={"query": query})
            if response.status_code == 200:
                result = response.json().get("result", {})
                st.success("Response:")
                st.json(result)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
            

if st.button("Clear Memory"):
    try:
        response = requests.post(clear_memory_url)
        if response.status_code == 200:
            st.success("Memory cleared successfully.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

