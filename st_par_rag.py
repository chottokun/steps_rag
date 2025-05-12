import streamlit as st
import os
import json
import uuid
from typing import List, Dict, Any, TypedDict, Annotated

# Langchain/Langgraph imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAIをコメントアウト
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama # Ollamaをインポート
from langchain_openai import OpenAIEmbeddings # Chromadb用にOpenAI Embeddingsは残す
# from langchain_community.embeddings import OllamaEmbeddings # Ollama Embeddingsを使用する場合
from langgraph.graph import StateGraph, END

# Chromadb imports
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
# LLMプロバイダーの選択
LLM_PROVIDER = st.sidebar.selectbox("LLM Provider", ["Ollama", "OpenAI"])

# Ollama設定
if LLM_PROVIDER == "Ollama":
    ollama_base_url = st.sidebar.text_input("Ollama Base URL", "http://localhost:11434")
    ollama_model_name = st.sidebar.text_input("Ollama Model Name", "gemma3:4b-it-qat") # 使用するOllamaモデル名
    if not ollama_base_url or not ollama_model_name:
        st.error("Ollama Base URL と Model Name を設定してください。")
        st.stop()
    # Ollama Chat Modelの初期化
    llm = ChatOllama(base_url=ollama_base_url, model=ollama_model_name, temperature=0)
    st.sidebar.success(f"Ollama ({ollama_model_name}) をLLMとして使用します。")

# OpenAI設定
elif LLM_PROVIDER == "OpenAI":
    # 環境変数からOpenAI APIキーを読み込む
    # Streamlit Secretsを使用する場合は st.secrets["OPENAI_API_KEY"] に変更
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")
        st.stop()
    # OpenAI Chat Modelの初期化
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0) # デフォルトモデルを指定
    st.sidebar.success("OpenAI をLLMとして使用します。")

# Embeddingモデルの指定 (Chromadb用)
# 現在はOpenAI Embeddingsを使用。Ollama Embeddingsに変更する場合は以下をコメントアウトし、
# langchain_community.embeddings.OllamaEmbeddings をインポートして使用する。
# EMBEDDING_MODEL = "text-embedding-ada-002"
# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_api_key) # OpenAI APIキーが必要

# Ollama Embeddingsを使用する場合の例:
# EMBEDDING_MODEL = "nomic-embed-text" # 使用するOllama Embeddingモデル名
# embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=EMBEDDING_MODEL)

# HuggingFace Embeddingsを使用する場合の例:
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small") # HuggingFaceの埋め込みモデルを指定

# Chromadbの初期化
# インメモリデータベースを使用。永続化する場合はディレクトリを指定
client = chromadb.Client() # または chromadb.PersistentClient(path="./chroma_db")

# コレクション名
COLLECTION_NAME = "par_rag_documents"

# --- Data Preparation (Sample Data) ---
# RAGに使用するサンプル文書データ
sample_documents = [
    "PAR RAGは、マルチホップ質問応答の推論経路のずれやエラー伝播を軽減するために提案されました。",
    "PAR RAGはPlan, Action, Reviewの3つの主要な段階で構成されます。",
    "Plan Moduleは、複雑な問題を分解し、実行可能な多段階の計画をトップダウン方式で生成します。",
    "Action Moduleは、計画の各ステップを順序通りに実行し、情報検索と暫定回答生成を行います。",
    "Review Moduleは、実行段階で生成された暫定的な回答を検証し、必要に応じて修正します。",
    "Review Moduleは、細粒度の情報検索にベクトルインデックスとナレッジグラフを利用します。",
    "ナレッジグラフは、文書から抽出されたエンティティとその関係で構築されます。",
    "多粒度検索アルゴリズムとして、ベクトル類似性検索とPersonalized PageRank (PPR) が使用されます。",
    "PAR RAGの設計思想は、人間のPDCAサイクルから着想を得ています。",
    "計画は、各ステップの思考プロセス (thought) とサブ質問 (question) を含む構造で定義されます。",
    "Action Moduleでは、暫定回答生成時に引用に基づく証拠選択が行われます。",
    "Action Moduleでは、現在の暫定回答を用いて次ステップの質問を洗練します。",
    "各ステップの実行完了後、軌跡 (trajectory) が生成され、軌跡チェーンに追加されます。",
    "全ての計画ステップ完了後、軌跡チェーンを用いて最終的な回答が生成されます。",
    "PAR RAGは、マルチホップQAタスクでEMおよびF1スコアの向上を達成しました。",
    "PAR RAGの課題として、応答時間（RTPQ）とコスト（CTPQ）の増加が挙げられます。"
]

# ドキュメントをChromadbに投入する関数
def setup_chromadb(docs: List[str], collection_name: str, embedding_func):
    # コレクションが存在すれば削除
    try:
        client.delete_collection(name=collection_name)
    except:
        pass # コレクションが存在しない場合は何もしない

    # コレクションを作成
    # embedding_func パラメータで渡されたEmbeddingFunctionを使用
    collection = client.create_collection(name=collection_name, embedding_function=embedding_func)

    # ドキュメントとIDを準備
    ids = [str(uuid.uuid4()) for _ in docs]
    # メタデータとして元のテキストを保存 (引用時に使用)
    metadatas = [{"text": doc} for doc in docs]

    # ドキュメントをコレクションに追加
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )
    st.sidebar.success(f"Chromadbコレクション '{collection_name}' に {len(docs)} 件のドキュメントをロードしました。")
    return collection

# Chromadbコレクションの取得またはセットアップ
# EmbeddingFunctionを渡す必要があるため、HuggingFaceEmbeddingFunctionをここで定義
# chroma_embedding_func = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=openai_api_key, model_name="text-embedding-ada-002"
# )  # OpenAI Embeddingsを使用する場合の例 (無効化)

chroma_embedding_func = embedding_functions.HuggingFaceEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"  # 先に定義したHuggingFace Embeddingsと一致させる
)


try:
    collection = client.get_collection(name=COLLECTION_NAME)
    st.sidebar.info(f"既存のChromadbコレクション '{COLLECTION_NAME}' を使用します。")
except:
    st.sidebar.info(f"Chromadbコレクション '{COLLECTION_NAME}' をセットアップします。")
    # setup_chromadb 関数に embedding_func を渡す
    collection = setup_chromadb(sample_documents, COLLECTION_NAME, chroma_embedding_func)


# --- Langgraph State ---
# グラフの状態を定義
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's initial question.
        plan: The multi-step plan generated by the Plan Module.
        current_step_index: The index of the current step being executed in the plan.
        trajectory_chain: A list of dictionaries, where each dictionary contains
                          the question, provisional answer, and citations for a step.
        final_answer: The final answer generated after executing all steps.
        error: Any error message encountered during execution.
    """
    question: str
    plan: List[Dict[str, Any]]
    current_step_index: int
    trajectory_chain: List[Dict[str, Any]]
    final_answer: str
    error: str

# --- Langgraph Nodes (Modules) ---

# 1. Plan Module
def plan_module(state: GraphState) -> GraphState:
    """
    Generates a multi-step plan based on the user's question.
    """
    st.subheader("🚀 プランニング段階")
    question = state["question"]
    st.write(f"質問: {question}")

    plan_prompt = PromptTemplate(
        template="""あなたは複雑な質問を解決するための専門家です。
ユーザーの質問に対して、それを解決するための多段階の計画をトップダウン方式で生成してください。
計画はJSON形式で出力し、各ステップには 'thought' (そのステップの思考プロセス) と 'question' (そのステップで解決すべきサブ質問) を含めてください。
計画は実行可能で、最終的に元の質問に回答できるよう導くものである必要があります。
質問の複雑さに応じてステップ数を調整してください。

質問: {question}

計画のJSON形式例:
[
  {{
    "thought": "まず、最初のサブ問題を特定します。",
    "question": "最初のサブ質問"
  }},
  {{
    "thought": "次に、前のステップの結果を使って次のサブ問題を解決します。",
    "question": "次のサブ質問"
  }},
  ...
]
""",
        input_variables=["question"],
    )

    # LLMはグローバル変数として使用
    chain = plan_prompt | llm | JsonOutputParser()

    try:
        plan = chain.invoke({"question": question})
        st.write("生成された計画:")
        st.json(plan)
        return {**state, "plan": plan, "current_step_index": 0, "trajectory_chain": [], "final_answer": "", "error": ""}
    except Exception as e:
        st.error(f"プラン生成中にエラーが発生しました: {e}")
        return {**state, "error": f"プラン生成エラー: {e}"}


# 2. Action Module
def action_module(state: GraphState) -> GraphState:
    """
    Executes a single step of the plan: retrieves information, generates a provisional answer,
    selects citations, and refines the next question.
    """
    st.subheader(f"🔬 実行段階 (ステップ {state['current_step_index'] + 1})")
    question = state["question"]
    plan = state["plan"]
    current_step_index = state["current_step_index"]
    trajectory_chain = state["trajectory_chain"]

    if current_step_index >= len(plan):
        st.warning("計画のステップ数が超過しました。")
        return {**state, "error": "計画ステップ超過"}

    current_step = plan[current_step_index]
    sub_question = current_step["question"]
    st.write(f"現在のサブ質問: {sub_question}")

    # 粗粒度検索 (Chromadbを使用)
    st.info("情報を検索中...")
    try:
        # 検索クエリとしてサブ質問を使用
        # collection はグローバル変数として使用
        results = collection.query(
            query_texts=[sub_question],
            n_results=5, # 取得するドキュメント数
            include=['metadatas'] # メタデータ（元のテキスト）を含める
        )
        retrieved_docs = [res['text'] for res in results['metadatas'][0]]
        st.write("検索結果:")
        for i, doc in enumerate(retrieved_docs):
            st.write(f"- {doc[:100]}...") # ドキュメントの冒頭を表示
    except Exception as e:
        st.error(f"情報検索中にエラーが発生しました: {e}")
        return {**state, "error": f"情報検索エラー: {e}"}

    # 暫定回答生成と引用選択
    st.info("暫定回答を生成中...")
    answer_prompt = PromptTemplate(
        template="""以下の情報と質問に基づいて、暫定的な回答を生成してください。
回答には、提供された情報の中から回答の根拠となる部分を引用として含めてください。
引用は、元のテキストをそのまま使用し、必要に応じて文脈に合わせて調整してください。
もし情報が不足している場合は、その旨を回答に含めてください。

情報:
{context}

質問: {question}

暫定回答と引用:
""",
        input_variables=["context", "question"],
    )

    context = "\n".join(retrieved_docs)
    # LLMはグローバル変数として使用
    chain = answer_prompt | llm

    try:
        provisional_answer_with_citations = chain.invoke({"context": context, "question": sub_question}).content
        st.write("暫定回答と引用:")
        st.write(provisional_answer_with_citations)

        # 暫定回答から引用部分と回答部分を分離する（簡易的な方法）
        # より厳密な引用抽出はLLMにJSON形式で出力させるなど工夫が必要
        provisional_answer = provisional_answer_with_citations # 一旦全体を暫定回答とする
        citations = retrieved_docs # 検索されたドキュメント全体を引用として扱う（簡易化）

    except Exception as e:
        st.error(f"暫定回答生成中にエラーが発生しました: {e}")
        return {**state, "error": f"暫定回答生成エラー: {e}"}

    # 次ステップの質問洗練 (もし次のステップがあれば)
    refined_next_question = None
    if current_step_index + 1 < len(plan):
        st.info("次ステップの質問を洗練中...")
        next_step = plan[current_step_index + 1]
        original_next_question = next_step["question"]

        refine_prompt = PromptTemplate(
            template="""あなたは質問洗練の専門家です。
現在の暫定的な回答と次のステップの元の質問に基づいて、次のステップの質問を洗練してください。
洗練された質問は、現在の回答の文脈を考慮し、より具体的で効果的な情報検索や推論を可能にするように調整してください。
洗練された質問のみを出力してください。

現在の暫定回答: {provisional_answer}

次のステップの元の質問: {original_next_question}

洗練された次の質問:
""",
            input_variables=["provisional_answer", "original_next_question"],
        )
        # LLMはグローバル変数として使用
        chain = refine_prompt | llm

        try:
            refined_next_question = chain.invoke({"provisional_answer": provisional_answer, "original_next_question": original_next_question}).content
            st.write(f"洗練された次の質問: {refined_next_question}")
            # 計画の次のステップの質問を更新
            plan[current_step_index + 1]["question"] = refined_next_question
        except Exception as e:
            st.warning(f"次ステップの質問洗練中にエラーが発生しました: {e}")
            # エラーが発生しても処理は続行
            refined_next_question = original_next_question # 洗練に失敗したら元の質問を使用

    # 軌跡の生成と追加
    trajectory = {
        "step": current_step_index + 1,
        "sub_question": sub_question,
        "provisional_answer": provisional_answer,
        "citations": citations
    }
    trajectory_chain.append(trajectory)
    st.write("軌跡を更新しました。")

    # 次のステップへ進む
    next_step_index = current_step_index + 1

    return {
        **state,
        "plan": plan, # 洗練された質問で更新された計画
        "current_step_index": next_step_index,
        "trajectory_chain": trajectory_chain,
        "error": ""
    }

# 3. Review Module
def review_module(state: GraphState) -> GraphState:
    """
    Reviews the provisional answer generated in the Action Module,
    validates its accuracy, and potentially revises it.
    """
    st.subheader(f"🧐 レビュー段階 (ステップ {state['current_step_index']})") # Actionの後に呼ばれるため、indexはActionでインクリメントされた後
    question = state["question"]
    plan = state["plan"]
    current_step_index = state["current_step_index"] - 1 # レビュー対象は直前のステップ
    trajectory_chain = state["trajectory_chain"]

    if not trajectory_chain:
        st.warning("レビュー対象の軌跡がありません。")
        return state # レビューするものがない場合はそのまま返す

    last_trajectory = trajectory_chain[-1]
    sub_question = last_trajectory["sub_question"]
    provisional_answer = last_trajectory["provisional_answer"]
    st.write(f"レビュー対象のサブ質問: {sub_question}")
    st.write(f"レビュー対象の暫定回答: {provisional_answer}")

    # 細粒度検索 (ここでは暫定回答をクエリとしてChromadbを使用)
    # 本来はKGも利用するが、簡易化のためChromadbのみ
    st.info("レビューのための情報を検索中...")
    try:
        # 検索クエリとして暫定回答を使用
        # collection はグローバル変数として使用
        results = collection.query(
            query_texts=[provisional_answer],
            n_results=3, # レビューのための少数の関連ドキュメントを取得
            include=['metadatas']
        )
        review_docs = [res['text'] for res in results['metadatas'][0]]
        st.write("レビュー用検索結果:")
        for i, doc in enumerate(review_docs):
            st.write(f"- {doc[:100]}...")
    except Exception as e:
        st.warning(f"レビュー用情報検索中にエラーが発生しました: {e}")
        review_docs = [] # エラーが発生してもレビューは続行

    # 暫定回答の検証と修正
    st.info("暫定回答を検証・修正中...")
    review_prompt = PromptTemplate(
        template="""あなたは回答検証の専門家です。
以下の元の質問、暫定的な回答、そしてレビューのための関連情報を考慮して、暫定回答の正確性を検証してください。
もし暫定回答が不正確、不完全、または提供された情報と矛盾している場合、関連情報に基づいて回答を修正してください。
検証の結果、回答が正確であれば、元の暫定回答をそのまま出力してください。
修正が必要な場合は、修正後の回答のみを出力してください。

元の質問: {original_question}

レビュー対象のサブ質問: {sub_question}

レビュー対象の暫定回答: {provisional_answer}

レビューのための関連情報:
{review_context}

検証と修正後の回答:
""",
        input_variables=["original_question", "sub_question", "provisional_answer", "review_context"],
    )

    review_context = "\n".join(review_docs)
    # LLMはグローバル変数として使用
    chain = review_prompt | llm

    try:
        revised_answer = chain.invoke({
            "original_question": question,
            "sub_question": sub_question,
            "provisional_answer": provisional_answer,
            "review_context": review_context
        }).content
        st.write("検証・修正後の回答:")
        st.write(revised_answer)

        # 軌跡チェーンの最後の回答を更新
        trajectory_chain[-1]["provisional_answer"] = revised_answer

    except Exception as e:
        st.warning(f"暫定回答の検証・修正中にエラーが発生しました: {e}")
        # エラーが発生しても軌跡は更新しない

    return {**state, "trajectory_chain": trajectory_chain, "error": ""}


# 4. Final Answer Module
def final_answer_module(state: GraphState) -> GraphState:
    """
    Generates the final answer based on the initial question and the trajectory chain.
    """
    st.subheader("✨ 最終回答生成段階")
    question = state["question"]
    trajectory_chain = state["trajectory_chain"]

    st.write("軌跡チェーン全体を考慮して最終回答を生成中...")

    # 軌跡チェーンを整形してプロンプトに含める
    trajectory_text = ""
    for traj in trajectory_chain:
        trajectory_text += f"--- ステップ {traj['step']} ---\n"
        trajectory_text += f"サブ質問: {traj['sub_question']}\n"
        trajectory_text += f"暫定回答: {traj['provisional_answer']}\n"
        # 引用は冗長になるためここでは含めないが、必要に応じて追加
        # trajectory_text += f"引用: {', '.join(traj['citations'])}\n"
        trajectory_text += "\n"

    final_answer_prompt = PromptTemplate(
        template="""あなたは最終回答生成の専門家です。
以下の元の質問と、それを解決するために実行された多段階の思考プロセス（軌跡チェーン）を考慮して、最終的な回答を生成してください。
軌跡チェーンには各ステップでのサブ質問と暫定回答が含まれています。
これらの情報をもとに、元の質問に対する包括的で正確な最終回答をまとめてください。

元の質問: {question}

軌跡チェーン:
{trajectory_chain}

最終回答:
""",
        input_variables=["question", "trajectory_chain"],
    )

    # LLMはグローバル変数として使用
    chain = final_answer_prompt | llm

    try:
        final_answer = chain.invoke({"question": question, "trajectory_chain": trajectory_text}).content
        st.write("最終回答:")
        st.success(final_answer)
        return {**state, "final_answer": final_answer, "error": ""}
    except Exception as e:
        st.error(f"最終回答生成中にエラーが発生しました: {e}")
        return {**state, "error": f"最終回答生成エラー: {e}"}

# --- Langgraph Graph Definition ---

# グラフを構築
workflow = StateGraph(GraphState)

# ノードを追加
workflow.add_node("plan", plan_module)
workflow.add_node("action", action_module)
workflow.add_node("review", review_module)
workflow.add_node("final_answer", final_answer_module)

# エッジ（遷移）を定義

# 開始ノード
workflow.set_entry_point("plan")

# plan -> action
workflow.add_edge("plan", "action")

# action -> review
workflow.add_edge("action", "review")

# review からの条件付き遷移
# 次のステップがあるか、またはエラーが発生したかで遷移先を決定
def should_continue(state: GraphState) -> str:
    """
    Determines whether to continue to the next action step or finish.
    """
    if state.get("error"):
        return "end" # エラーが発生したら終了
    plan = state["plan"]
    current_step_index = state["current_step_index"]
    # current_step_index は action_module の最後でインクリメントされている
    if current_step_index < len(plan):
        return "continue" # 次のステップがあれば action へ
    else:
        return "end" # なければ final_answer へ

workflow.add_conditional_edges(
    "review",
    should_continue,
    {
        "continue": "action",
        "end": "final_answer"
    }
)

# final_answer -> END
workflow.add_edge("final_answer", END)

# グラフをコンパイル
app = workflow.compile()

# --- Streamlit App UI ---

st.title("PAR RAG デモ")
st.markdown("""
PAR RAG (Plan-Action-Review Retrieval-Augmented Generation) のデモンストレーションアプリです。
複雑な質問に対して、Plan, Action, Reviewの段階を経て回答を生成します。
サイドバーでLLMプロバイダーを選択できます。
""")

# 質問入力
user_question = st.text_input("質問を入力してください:", "PAR RAGの設計思想と主要な機能要件は何ですか？")

# 実行ボタン
if st.button("実行"):
    if not user_question:
        st.warning("質問を入力してください。")
    else:
        st.info("処理を開始します...")
        # グラフの実行
        # 状態の初期化
        initial_state = {
            "question": user_question,
            "plan": [],
            "current_step_index": 0,
            "trajectory_chain": [],
            "final_answer": "",
            "error": ""
        }

        # グラフの実行と結果の表示
        # ストリーム表示には対応していない簡易的な実行
        try:
            # 実行中に各ノードからの出力がStreamlitに表示される
            # LLMとcollectionはグローバル変数としてアクセスされる
            final_state = app.invoke(initial_state)

            # 最終結果の確認 (final_answer_moduleで表示済みだが、念のため)
            # if final_state.get("final_answer"):
            #     st.subheader("最終回答:")
            #     st.success(final_state["final_answer"])
            if final_state.get("error"):
                 st.error(f"処理中にエラーが発生しました: {final_state['error']}")

            st.balloons() # 完了時にバルーンを表示

        except Exception as e:
            st.error(f"グラフ実行中に予期せぬエラーが発生しました: {e}")

st.markdown("---")
st.write("サイドバーでLLMプロバイダーとChromadbのロード状況を確認できます。")
