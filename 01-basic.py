import streamlit as st
import sys
import subprocess
from langchain_openai import ChatOpenAI

st.title("🦜🔗 LM Studio Quickstart App")

# LM Studio側のホストIPを動的に取得する関数
def get_host_ip():
    if sys.platform == "darwin":  # macOSの場合
        return "localhost"
    else:
        try:
            # Linux/WSL2などでホストマシンのIPを取得
            return subprocess.check_output(
                "ip route | grep default", shell=True
            ).decode().split()[2]
        except Exception:
            return "127.0.0.1"

# レスポンス生成用の関数
def generate_response(input_text):
    model = ChatOpenAI(
        base_url=f"http://{get_host_ip()}:1234/v1",
        api_key="lm-studio", 
        model_name="gemma-4-e4b",
        temperature=0.7
    )
    
    try:
        # シンプルに invoke で呼び出し
        response = model.invoke(input_text)
        # st.info で青いボックスに結果を表示
        st.info(response.content)
    except Exception as e:
        st.error(f"接続エラー: {e}")

# Streamlit のフォーム機能
with st.form("my_form"):
    # デフォルトの質問
    text = st.text_area("Enter text:", "プログラムを学ぶ上で、3つのポイントは？")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # 送信ボタンが押されたら実行
        generate_response(text)
