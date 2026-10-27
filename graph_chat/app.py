from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_engine import GraphChatEngine


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR.parent / "graph_preprocess"))
ARTIFACTS_DIR = Path(
    os.getenv(
        "ARTIFACTS_DIR",
        DATA_DIR / "graphrag_preprocessed_rubert" / "output",
    )
)
CHECKPOINT_PATH = Path(
    os.getenv("CHECKPOINT_PATH", DATA_DIR / "C3_A4_posweight_model.pt")
)
TOP_K = 12

st.set_page_config(
    page_title="GraphRAG Chat",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1050px; padding-top: 2rem;}
      [data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.2); border-radius: 14px; padding: .35rem .7rem;}
      .model-badge {display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#eef4ff; color:#234; font-size:.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> GraphChatEngine:
    return GraphChatEngine(
        artifacts_dir=ARTIFACTS_DIR,
        checkpoint_path=CHECKPOINT_PATH,
    )


st.title("🕸️ GraphRAG Chat")
st.caption(
    "Поиск по графу выполняет обученная GAT-модель "
    "`C3_A4_posweight_model.pt`, ответ формирует локальная LLM."
)

try:
    with st.spinner("Загружаю граф и GAT-модель…"):
        engine = load_engine()
except Exception as error:
    st.error(f"Не удалось запустить движок: {error}")
    st.stop()

with st.sidebar:
    st.subheader("Настройки")
    st.markdown(
        f'<span class="model-badge">{engine.model_config["name"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"LLM: `{engine.llm_model}`")
    st.caption(f"API: `{engine.llm_base_url}`")

    if st.button("Проверить LLM", use_container_width=True):
        health = engine.llm_health()
        if health["ok"]:
            st.success(health["message"])
        else:
            st.error(health["message"])

    if st.button("Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Задайте вопрос по материалам графа. Например: "
            "**«Где Сергей Скородумов защищал диссертацию?»**"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Введите вопрос по графу…")
if prompt:
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Ищу узлы и формирую ответ…", expanded=False):
                result = engine.answer(prompt, history=history, top_k=TOP_K)
            st.markdown(result["answer"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                }
            )
        except Exception as error:
            error_message = f"Не удалось получить ответ: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
