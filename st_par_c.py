import os
import shutil
from glob import glob
import streamlit as st
from dotenv import load_dotenv

# LangChain ドキュメントローダーとスプリッター
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# 埋め込みとベクターストア
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# LLM モデル
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# .envファイルから環境変数を読み込む
load_dotenv()

# ======== 設定値をまとめて管理 ========
def get_config():
    return {
        # LLMサービス: 'azure', 'openai', 'ollama'
        "LLM_SERVICE": "ollama",
        "AZURE_DEPLOYMENT_NAME": "gpt-4o",
        "AZURE_API_VERSION": "2024-04-01",
        "OPENAI_MODEL": "gpt-4o",
        "OLLAMA_MODEL": "gemma3:4b-it-qat",
        "OLLAMA_BASE_URL": "http://localhost:11434",

        # 埋め込みサービス: 'azure', 'openai', 'hf'
        "EMBEDDING_SERVICE": "hf",
        "AZURE_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_EMBEDDING_API_VERSION": "2024-04-01",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "HF_EMBEDDING_MODEL": "intfloat/multilingual-e5-small",

        # その他
        "PDF_FOLDER": "pdfs",
        "CHROMA_PATH": "chroma_db",
    }

# ======== モデルと埋め込み設定 ========
@st.cache_resource
def load_embeddings(config):
    service = config.get("EMBEDDING_SERVICE", "hf")
    if service == "azure":
        try:
            return OpenAIEmbeddings(
                deployment=config["AZURE_EMBEDDING_DEPLOYMENT"],
                openai_api_version=config["AZURE_EMBEDDING_API_VERSION"]
            )
        except Exception:
            st.error("Azure OpenAI埋め込みモデルの初期化に失敗しました。")
            st.stop()
    elif service == "openai":
        try:
            return OpenAIEmbeddings(
                model=config["OPENAI_EMBEDDING_MODEL"]
            )
        except Exception:
            st.error("OpenAI埋め込みモデルの初期化に失敗しました。")
            st.stop()
    elif service == "hf":
        try:
            return HuggingFaceEmbeddings(model_name=config["HF_EMBEDDING_MODEL"])
        except Exception:
            st.error("HuggingFace埋め込みモデルの初期化に失敗しました。")
            st.stop()
    else:
        st.error("埋め込みサービスの設定が不正です。")
        st.stop()

def load_llm(config):
    service = config.get("LLM_SERVICE", "ollama")
    if service == "azure":
        return AzureChatOpenAI(
            deployment_name=config["AZURE_DEPLOYMENT_NAME"],
            openai_api_version=config["AZURE_API_VERSION"]
        )
    elif service == "openai":
        return ChatOllama(model=config["OPENAI_MODEL"])
    elif service == "ollama":
        return ChatOllama(
            model=config["OLLAMA_MODEL"],
            base_url=config["OLLAMA_BASE_URL"]
        )
    else:
        st.error("LLMサービスの設定が不正です。")
        st.stop()

# ======== PDFフォルダ読み込み & ChromaDB 構築 ========
def build_chroma(embeddings, config):
    loader = PyPDFDirectoryLoader(
        path=config["PDF_FOLDER"],
    )
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter=splitter)
    st.write(f"PDF chunk 数: {len(docs)} 件")
    if len(docs) == 0:
        st.error("PDFが1件も読み込めませんでした。pdfsフォルダにPDFが存在するか確認してください。")
        return None
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=config["CHROMA_PATH"]
    )
    return vectorstore

# ======== Plan ステップ ========
from pydantic import BaseModel, Field

class PlanSteps(BaseModel):
    steps: list[str] = Field(description="質問に答えるための段階的な検索ステップ")

def get_planner_chain(llm):
    plan_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
ユーザーの質問に答えるために、段階的に検索ステップを計画してください。
質問: {question}
"""
    )
    return plan_prompt | llm.with_structured_output(PlanSteps)

# ======== Act & Answer ステップ ========
def retrieve_and_answer(vectorstore, steps, llm, max_retry=2):
    class StepReview(BaseModel):
        result: str = Field(description="OKなら'OK'、不十分・誤りがあれば'NG'")
        reason: str = Field(description="NGの場合の理由。OKなら空文字")
        new_query: str = Field(description="NGの場合の再検索クエリ案。OKなら空文字")

    results = []
    for i, step in enumerate(steps, start=1):
        current_step = step
        for retry in range(max_retry):
            st.markdown(f"### Step {i} {step}- Try: {retry+1}")
            docs = vectorstore.similarity_search(current_step, k=3)
            context = "\n".join([d.page_content for d in docs])
            step_prompt = PromptTemplate(
                input_variables=["step", "context"],
                template="""
以下の情報に基づいて中間回答を生成してください。
ステップ: {step}
参照:
{context}

回答:
"""
            )
            step_chain = step_prompt | llm
            answer = step_chain.invoke({"step": current_step, "context": context})
            st.info(f"**中間回答:** {answer.content}")

            # --- RAGによる中間回答の正確性チェック ---
            review_prompt = PromptTemplate(
                input_variables=["step", "answer", "context"],
                template="""
以下は、ある検索ステップに対する中間回答です。

ステップ: {step}
回答: {answer}
参照情報:
{context}

この回答は十分に正確・妥当ですか？
- 十分なら result: OK, reason: , new_query: 
- 不十分・誤りがあれば result: NG, reason: (理由), new_query: (再検索クエリ案)
出力は必ずJSON形式で:
{{
  \"result\": ...,
  \"reason\": ...,
  \"new_query\": ...
}}
"""
            )
            review_chain = review_prompt | llm.with_structured_output(StepReview)
            review_result = review_chain.invoke({"step": current_step, "answer": answer, "context": context})

            st.markdown(f"**レビュー判定:** {review_result.result}")
            if review_result.result.strip().upper() == "OK":
                st.success("OK: 十分な回答です。")
                results.append((current_step, answer))
                break
            else:
                st.warning(f"NG: {review_result.reason}")
                if review_result.new_query and review_result.new_query != current_step:
                    st.info(f"再検索クエリ案: {review_result.new_query}")
                    current_step = review_result.new_query
                if retry == max_retry - 1:
                    st.error("リトライ上限に達したため、この回答を採用します。")
                    results.append((current_step, answer))
    return results

# ======== Review ステップ ========
def review_steps(step_results, llm):
    review_prompt = PromptTemplate(
        input_variables=["steps"],
        template="""
以下は各ステップの中間回答です。整合性や不明点をレビューしてください。

{steps}

レビュー結果:
"""
    )
    steps_text = "\n".join([f"{i}. {s} → {a}" for i, (s, a) in enumerate(step_results, start=1)])
    review_chain = review_prompt | llm
    return review_chain.invoke({"steps": steps_text})

# ======== Aggregate ステップ ========
def aggregate_answer(question, step_results, review_notes, llm):
    agg_prompt = PromptTemplate(
        input_variables=["question", "steps", "review"],
        template="""
質問: {question}

各ステップ回答:
{steps}

レビュー:
{review}

最終的に統合した回答:
"""
    )
    steps_text = "\n".join([f"{i}. {s} → {a}" for i, (s, a) in enumerate(step_results, start=1)])
    agg_chain = agg_prompt | llm
    return agg_chain.invoke({"question": question, "steps": steps_text, "review": review_notes})

# ======== Streamlit UI & メイン処理 ========
def main():
    config = get_config()
    embeddings = load_embeddings(config)
    llm = load_llm(config)
    planner_chain = get_planner_chain(llm)

    st.title("📚 Steps-RAG")

    if st.button("🔄 ChromaDB 再構築"):
        build_chroma(embeddings, config)
        st.success("ChromaDBを再構築しました")

    question = st.text_input("💬 質問を入力:")
    if st.button("🧠 回答生成") and question:
        vectorstore = Chroma(persist_directory=config["CHROMA_PATH"], embedding_function=embeddings)
        # Step 1: Plan
        st.subheader("📝 Step 1: Plan")
        plan_obj = planner_chain.invoke({"question": question})
        st.write(plan_obj)
        steps = plan_obj.steps if hasattr(plan_obj, "steps") else []
        # Step 2: Retrieve & Answer
        st.subheader("🔍 Step 2: Retrieve & Answer")
        step_results = retrieve_and_answer(vectorstore, steps, llm)
        # for i, (s, a) in enumerate(step_results, start=1):
        #     st.markdown(f"**Step {i}: {s}**")
        #     st.write(a)
        # Step 3: Review
        st.subheader("🔎 Step 3: Review")
        review = review_steps(step_results, llm)
        st.write(review.content)
        # Step 4: Aggregate
        st.subheader("✅ Step 4: Final Answer")
        final = aggregate_answer(question, step_results, review, llm)
        st.write(final.content)

if __name__ == "__main__":
    main()
