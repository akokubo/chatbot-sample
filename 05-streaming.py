import streamlit as st
import sys
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.title("🌊 Streaming LangChain Bot")

# --- 1. 環境設定 & IP取得 ---
def get_host_ip():
    if sys.platform == "darwin":
        return "localhost"
    else:
        try:
            return subprocess.check_output("ip route | grep default", shell=True).decode().split()[2]
        except Exception: return "127.0.0.1"

# --- 2. モデル・プロンプト・チェインの定義 ---
llm = ChatOpenAI(
    base_url=f"http://{get_host_ip()}:1234/v1",
    api_key="lm-studio", 
    model_name="gemma-4-e4b",
    temperature=0.7,
    streaming=True  # ストリーミングを有効化（明示的に設定）
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切で優秀なAIアシスタントです。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# --- 3. 履歴の管理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 画面への履歴表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. メイン処理 ---
if user_input := st.chat_input("メッセージを入力..."):
    # ユーザー入力を表示 & 保存
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AIの応答生成（ストリーミング表示）
    with st.chat_message("assistant"):
        # 履歴をLangChain形式に変換
        history_for_chain = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                history_for_chain.append(HumanMessage(content=m["content"]))
            else:
                history_for_chain.append(AIMessage(content=m["content"]))

        try:
            # 1. 実行を stream() に変更
            # これはジェネレータ（逐次形式のデータ）を返してくる
            stream = chain.stream({
                "history": history_for_chain,
                "input": user_input
            })

            # 2. st.write_stream で表示
            # ジェネレータを渡すだけで、表示が終わると「全テキスト」を返す
            full_response = st.write_stream(stream)
            
            # 3. 最終的な回答を履歴に保存
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
