import streamlit as st
import sys
import subprocess
from langchain_openai import ChatOpenAI

st.title("🦜🔗 LM Studio Quickstart App")

def get_host_ip():
    if sys.platform == "darwin":
        return "localhost"
    else:
        try:
            return subprocess.check_output(
                "ip route | grep default", shell=True
            ).decode().split()[2]
        except Exception:
            return "127.0.0.1"

def generate_response(input_text):
    model = ChatOpenAI(
        base_url=f"http://{get_host_ip()}:1234/v1",
        api_key="lm-studio", 
        model_name="gemma-4-e4b",
        temperature=0.7
    )
    
    # --- スピナー ---
    with st.spinner("AIが回答を作成中..."):
        try:
            response = model.invoke(input_text)
            st.info(response.content)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    # --------------------------

with st.form("my_form"):
    text = st.text_area("Enter text:", "プログラムを学ぶ上で、3つのポイントは？")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        generate_response(text)
