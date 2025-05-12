import os
import shutil
from glob import glob
import streamlit as st

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

# ======== 設定値をまとめて管理 ========
def get_config():
    return {
        "USE_AZURE": False,  # True にすると Azure OpenAI を使用
        "PDF_FOLDER": "pdfs",
        "CHROMA_PATH": "chroma_db",
        "AZURE_DEPLOYMENT_NAME": "gpt-4o",
        "AZURE_API_VERSION": "2024-04-01",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "gemma3:4b-it-qat",
        "HF_EMBEDDING_MODEL": "intfloat/multilingual-e5-small"
    }

# ======== モデルと埋め込み設定 ========
@st.cache_resource
def load_embeddings(config):
    if config["USE_AZURE"]:
        try:
            embeddings = OpenAIEmbeddings(
                deployment="text-embedding-ada-002",
                openai_api_version=config["AZURE_API_VERSION"]
            )
        except Exception as e:
            st.error("Azure OpenAI埋め込みモデルの初期化に失敗しました。")
            st.stop()
        return embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config["HF_EMBEDDING_MODEL"])
    except Exception as e:
        st.error("HuggingFace埋め込みモデルの初期化に失敗しました。")
        st.stop()
    return embeddings

def load_llm(config):
    if config["USE_AZURE"]:
        return AzureChatOpenAI(
            deployment_name=config["AZURE_DEPLOYMENT_NAME"],
            openai_api_version=config["AZURE_API_VERSION"]
        )
    else:
        return ChatOllama(model=config["OLLAMA_MODEL"], base_url=config["OLLAMA_BASE_URL"])

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
厳密に判定し、不明な場合はNGとしてください。
出力は必ずJSON形式で:
{{
  \"result\": ...,
  \"reason\": ...,
  \"new_query\": ...
}}
"""
            )
            review_chain = review_prompt | llm.with_structured_output(StepReview)
            review_result = review_chain.invoke({"step": current_step, "answer": answer.content, "context": context})

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

# ======== LangGraphによるワークフロー定義 ========
from langgraph.graph import StateGraph, END

def plan_node(state):
    llm = state["llm"]
    planner_chain = get_planner_chain(llm)
    plan_obj = planner_chain.invoke({"question": state["question"]})
    steps = plan_obj.steps if hasattr(plan_obj, "steps") else []
    return {"steps": steps, "step_index": 0, "retry_count": 0, "step_results": []}

def retrieve_node(state):
    vectorstore = state["vectorstore"]
    steps = state["steps"]
    idx = state["step_index"]
    step = steps[idx]
    docs = vectorstore.similarity_search(step, k=3)
    context = "\n".join([d.page_content for d in docs])
    return {"context": context}

def answer_node(state):
    llm = state["llm"]
    steps = state["steps"]
    idx = state["step_index"]
    step = steps[idx]
    context = state["context"]
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
    answer = step_chain.invoke({"step": step, "context": context})
    return {"answer": answer.content}

def review_node(state):
    llm = state["llm"]
    steps = state["steps"]
    idx = state["step_index"]
    step = steps[idx]
    answer = state["answer"]
    context = state["context"]
    class StepReview(BaseModel):
        result: str = Field(description="OKなら'OK'、不十分・誤りがあれば'NG'")
        reason: str = Field(description="NGの場合の理由。OKなら空文字")
        new_query: str = Field(description="NGの場合の再検索クエリ案。OKなら空文字")
    review_prompt = PromptTemplate(
        input_variables=["step", "answer", "context"],
        template="""
以下は、ある検索ステップに対する中間回答です。

ステップ: {step}
回答: {answer}
参照情報:
{context}

この回答はステップに対して十分に正確・妥当ですか？
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
    review_result = review_chain.invoke({"step": step, "answer": answer, "context": context})
    return {"review_result": review_result}

def aggregate_node(state):
    llm = state["llm"]
    question = state["question"]
    step_results = state["step_results"]
    review_notes = state.get("review_notes", "")
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
    steps_text = "\n".join([f"{i+1}. {s} → {a}" for i, (s, a) in enumerate(step_results)])
    agg_chain = agg_prompt | llm
    final = agg_chain.invoke({"question": question, "steps": steps_text, "review": review_notes})
    return {"final_answer": final.content}

def review_steps_node(state):
    llm = state["llm"]
    step_results = state["step_results"]
    review_prompt = PromptTemplate(
        input_variables=["steps"],
        template="""
以下は各ステップの中間回答です。整合性や不明点をレビューしてください。

{steps}

レビュー結果:
"""
    )
    steps_text = "\n".join([f"{i+1}. {s} → {a}" for i, (s, a) in enumerate(step_results)])
    review_chain = review_prompt | llm
    review = review_chain.invoke({"steps": steps_text})
    return {"review_notes": review.content}

# ======== LangGraphグラフ構築 ========
def build_graph():
    graph = StateGraph()
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("review", review_node)
    graph.add_node("review_steps", review_steps_node)
    graph.add_node("aggregate", aggregate_node)
    # Plan → Retrieve
    graph.add_edge("plan", "retrieve")
    # Retrieve → Answer
    graph.add_edge("retrieve", "answer")
    # Answer → Review
    graph.add_edge("answer", "review")
    # Review → (OK: 次のステップ or review_steps, NG: リトライ or終了)
    def review_cond(state):
        review_result = state["review_result"]
        steps = state["steps"]
        idx = state["step_index"]
        retry = state["retry_count"]
        max_retry = 2
        if review_result.result.strip().upper() == "OK":
            if idx + 1 < len(steps):
                return "retrieve"  # 次のステップへ
            else:
                return "review_steps"  # 全ステップ終了
        else:
            if retry + 1 < max_retry:
                return "retrieve"  # リトライ
            else:
                if idx + 1 < len(steps):
                    return "retrieve"  # 次のステップへ
                else:
                    return "review_steps"
    graph.add_conditional_edges("review", review_cond, {"retrieve": "retrieve", "review_steps": "review_steps"})
    # review_steps → aggregate
    graph.add_edge("review_steps", "aggregate")
    # aggregate → END
    graph.add_edge("aggregate", END)
    return graph.compile()

# ======== Streamlit UI & メイン処理（LangGraph版） ========
def main():
    config = get_config()
    embeddings = load_embeddings(config)
    llm = load_llm(config)
    st.title("📚 Steps-RAG (LangGraph版)")
    if st.button("🔄 ChromaDB 再構築"):
        build_chroma(embeddings, config)
        st.success("ChromaDBを再構築しました")
    question = st.text_input("💬 質問を入力:")
    if st.button("🧠 回答生成 (LangGraph)") and question:
        vectorstore = Chroma(persist_directory=config["CHROMA_PATH"], embedding_function=embeddings)
        workflow = build_graph()
        # 初期状態
        state = {
            "question": question,
            "llm": llm,
            "vectorstore": vectorstore,
            "step_index": 0,
            "retry_count": 0,
            "step_results": []
        }
        for result in workflow.stream(state):
            node = result["__node__"]
            st.markdown(f"### ノード: {node}")
            if node == "plan":
                st.write(f"**Plan結果:** {result.get('steps', [])}")
            elif node == "retrieve":
                st.write(f"**検索コンテキスト:** {result.get('context', '')[:300]}...")
            elif node == "answer":
                st.info(f"**中間回答:** {result.get('answer', '')}")
            elif node == "review":
                review_result = result.get('review_result')
                st.markdown(f"**レビュー判定:** {review_result.result}")
                if review_result.result.strip().upper() == "OK":
                    # ステップ結果を蓄積
                    idx = result["step_index"]
                    steps = result["steps"]
                    answer = result["answer"]
                    step_results = result["step_results"]
                    step_results.append((steps[idx], answer))
                    result["step_results"] = step_results
                    result["step_index"] = idx + 1
                    result["retry_count"] = 0
                else:
                    st.warning(f"NG: {review_result.reason}")
                    if review_result.new_query and review_result.new_query != result["steps"][result["step_index"]]:
                        st.info(f"再検索クエリ案: {review_result.new_query}")
                        result["steps"][result["step_index"]] = review_result.new_query
                    result["retry_count"] += 1
            elif node == "review_steps":
                st.subheader("🔎 Step 3: Review")
                st.write(result.get("review_notes", ""))
            elif node == "aggregate":
                st.subheader("✅ Step 4: Final Answer")
                st.write(result.get("final_answer", ""))

if __name__ == "__main__":
    main()
