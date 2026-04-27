import streamlit as st
import sys
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage # メッセージ形式の導入

st.title("🤖 Chat with History (WSL)")

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

# 1. 履歴の初期化
# session_stateの中に"messages"というリストがなければ作成する
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 保存されている履歴を画面に表示する
# これにより過去のメッセージが上に表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. ユーザー入力 (チャット専用の入力バー)
if user_input := st.chat_input("何か聞いてください"):
    
    # ユーザーの入力を表示 & 履歴に追加
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 4. AIの応答生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            model = ChatOpenAI(
                base_url=f"http://{get_host_ip()}:1234/v1",
                api_key="lm-studio", 
                model_name="gemma-4-e4b",
                temperature=0.7
            )
            
            # --- 履歴をAIに渡すために ---
            # これまでの履歴をLangChainのメッセージ形式に変換
            history = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                else:
                    history.append(AIMessage(content=m["content"]))
            
            try:
                # 文字列ではなく、履歴（リスト）をそのまま渡す
                response = model.invoke(history)
                answer = response.content
                
                # 画面に表示
                st.markdown(answer)
                # 履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"エラー: {e}")
