# PAR RAGフレームワークを用いたLangChainシステムの構築

PAR RAG（Plan-Act-Review Retrieval Augmented Generation）は、推論経路の逸脱や中間結果のエラーが蓄積して最終的な回答精度を低下させる問題に対応するための新しいフレームワークです。従来のRAGシステムの弱点を克服し、より信頼性の高い回答を生成するための構造化されたアプローチを提供します。本レポートでは、LangChainを使用してPAR RAGシステムを実装するPythonプログラムの構築方法について解説します。

## RAGの進化と課題

従来のRetrieval Augmented Generation（RAG）は、大規模言語モデル（LLM）の知識を外部データで拡張する手法として広く利用されてきました。しかし、単純なRAGでは複雑なクエリへの対応や、推論経路の逸脱が課題となっています[^6]。特に、検索と生成のプロセスが単一ステップで行われる場合、誤った情報の取得や中間結果の誤りが最終回答に大きく影響します[^8]。

PlanRAGやAdaptive-RAGなどの発展的アプローチがこれらの課題への対応として提案されていますが[^2][^6]、PAR RAGフレームワークはこれらをさらに発展させ、計画（Plan）、実行と検証（Act \& Review）のサイクルを明示的に取り入れた手法です。

## PAR RAGフレームワークの実装

### 必要なライブラリのインストール

まず、必要なライブラリをインストールします。

```python
# 必要なライブラリをインストール
!pip install langchain langchain_openai faiss-cpu
```


### 基本的なコンポーネントの構築

PAR RAGシステム実装に必要な基本的なコンポーネントを準備します。

```python
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import os

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = "あなたのAPIキー"

# LLMの初期化
llm = OpenAI(temperature=0.7)
embeddings = OpenAIEmbeddings()
```


### ステップ1: データの準備とベクトルストアの構築

検索対象となる知識ベースを準備し、ベクトル化してインデックスを作成します。

```python
# サンプルデータの準備
documents = [
    Document(page_content="RAGシステムは情報検索と生成モデルを組み合わせたシステムです。", metadata={"source": "RAG概要"}),
    Document(page_content="PlanRAGは、まず計画を立ててから検索と回答を行う手法です。", metadata={"source": "新しいRAG手法"}),
    Document(page_content="推論経路の逸脱は、最終的な回答精度に大きく影響します。", metadata={"source": "RAGの課題"}),
    Document(page_content="Pythonは機械学習やAIアプリケーション開発に広く使われるプログラミング言語です。", metadata={"source": "プログラミング言語"}),
    Document(page_content="LangChainは、大規模言語モデルを使ったアプリケーション開発のためのフレームワークです。", metadata={"source": "AIフレームワーク"})
]

# ベクトルストアの作成
vector_store = FAISS.from_documents(documents, embeddings)
```


### ステップ2: Plan - 計画立案コンポーネント

ユーザーの質問から、どのような順序で検索と回答生成を行うかの計画を立てるコンポーネントを実装します。これはPlanRAGの概念に基づいており[^2]、複雑な質問を適切なステップに分解します。

```python
# 計画立案のためのプロンプトテンプレート
plan_template = """
あなたは質問応答システムのプランナーです。
ユーザーの質問から、回答を導くために必要な調査ステップを計画してください。
各ステップには順番と、そのステップで調べるべき具体的な質問を含めてください。

ユーザーの質問: {question}

計画:
"""

plan_prompt = PromptTemplate(template=plan_template, input_variables=["question"])
plan_chain = LLMChain(llm=llm, prompt=plan_prompt)

def create_plan(question):
    """ユーザーの質問から計画を生成する関数"""
    return plan_chain.run(question=question)
```


### ステップ3: Act \& Review - 実行と検証コンポーネント

計画の各ステップに対して、情報検索、中間回答の生成、正確性の検証を行うコンポーネントを実装します。

```python
# 検索用の関数
def retrieve_information(query, top_k=3):
    """クエリに関連する情報を検索する関数"""
    docs = vector_store.similarity_search(query, k=top_k)
    return docs

# 中間回答生成のためのプロンプトテンプレート
answer_template = """
以下の質問に関連する情報に基づいて回答を生成してください。

質問: {question}

関連情報:
{context}

回答:
"""

answer_prompt = PromptTemplate(template=answer_template, input_variables=["question", "context"])
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

# 回答検証のためのプロンプトテンプレート
review_template = """
以下の質問と回答を評価し、回答が正確かどうかを判断してください。
回答に不足や誤りがある場合は、どのように修正すべきかを指摘してください。

質問: {question}
回答: {answer}
関連情報: {context}

評価:
"""

review_prompt = PromptTemplate(template=review_template, input_variables=["question", "answer", "context"])
review_chain = LLMChain(llm=llm, prompt=review_prompt)

def act_and_review(step_question):
    """計画のステップに対して検索、回答生成、検証を行う関数"""
    # 情報検索
    docs = retrieve_information(step_question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 中間回答の生成
    initial_answer = answer_chain.run(question=step_question, context=context)
    
    # 回答の検証
    review = review_chain.run(question=step_question, answer=initial_answer, context=context)
    
    # 検証結果に基づいて、必要に応じて回答を修正
    if "修正" in review or "不足" in review:
        # 追加の検索と回答の再生成
        additional_docs = retrieve_information(step_question, top_k=5)
        additional_context = "\n".join([doc.page_content for doc in additional_docs])
        improved_answer = answer_chain.run(
            question=step_question, 
            context=f"{context}\n{additional_context}\n検証結果: {review}"
        )
        return improved_answer
    
    return initial_answer
```


### ステップ4: 最終回答の生成コンポーネント

各ステップの中間結果を統合して最終的な回答を生成するコンポーネントを実装します[^8]。

```python
# 最終回答生成のためのプロンプトテンプレート
final_answer_template = """
以下の質問に対する複数のステップの回答結果を統合して、最終的な回答を生成してください。

元の質問: {original_question}

ステップごとの回答:
{step_answers}

最終回答:
"""

final_answer_prompt = PromptTemplate(template=final_answer_template, 
                                     input_variables=["original_question", "step_answers"])
final_answer_chain = LLMChain(llm=llm, prompt=final_answer_prompt)

def generate_final_answer(original_question, step_results):
    """各ステップの結果を統合して最終回答を生成する関数"""
    step_answers = "\n".join([f"ステップ{i+1}: {result}" for i, result in enumerate(step_results)])
    return final_answer_chain.run(original_question=original_question, step_answers=step_answers)
```


### ステップ5: PAR RAGフレームワークの統合

これまでに実装した各コンポーネントを統合して、完全なPAR RAGシステムを構築します。SimpleSequentialChainを使用することも検討できます[^5]。

```python
def par_rag_system(user_question):
    """PAR RAGフレームワークを実行する主要関数"""
    # ステップ1: Plan - 計画立案
    plan = create_plan(user_question)
    print(f"計画:\n{plan}\n")
    
    # 計画からステップごとの質問を抽出
    # この実装では、計画が「ステップ1: 質問内容」の形式で書かれていることを想定
    step_questions = [line.split(": ", 1)[^1] for line in plan.strip().split("\n") 
                      if line.startswith("ステップ")]
    
    # ステップ2: Act &amp; Review - 各ステップの実行と検証
    step_results = []
    for i, step_question in enumerate(step_questions):
        print(f"\nステップ{i+1}の実行: {step_question}")
        step_answer = act_and_review(step_question)
        step_results.append(step_answer)
        print(f"ステップ{i+1}の回答: {step_answer}\n")
    
    # ステップ3: 最終回答の生成
    final_answer = generate_final_answer(user_question, step_results)
    print(f"最終回答:\n{final_answer}")
    
    return {
        "plan": plan,
        "step_results": step_results,
        "final_answer": final_answer
    }
```


### 完全なPAR RAGシステムの使用例

以下は、完成したPAR RAGシステムの使用例です。

```python
# PAR RAGシステムの実行例
user_question = "RAGシステムの課題と、新しいアプローチであるPlanRAGの利点について教えてください。"
results = par_rag_system(user_question)
```


## PAR RAGフレームワークの利点

このPAR RAGフレームワークには、以下のような利点があります：

1. **構造化された推論**: 計画を立ててから実行することで、推論経路の逸脱を防ぎます[^2]。
2. **エラーの検出と修正**: 各ステップで回答を検証することで、中間結果のエラーを早期に検出して修正できます[^6]。
3. **柔軟性**: クエリの複雑さに応じて検索戦略を適応させることができます[^6]。
4. **追跡可能性**: 各ステップでの決定と根拠が明示的に記録されるため、結果の検証が容易になります[^8]。

## 結論

PAR RAGフレームワークは、計画、実行と検証のサイクルを取り入れることで、従来のRAGシステムの限界を克服し、より信頼性の高い回答を提供します。LangChainを活用した実装により、柔軟で拡張性の高いシステムが構築できました。このアプローチは、特に複雑なクエリや高い正確性が求められる分野での応用が期待されます。

今後の発展方向としては、ドメイン固有の指示（Domain Specific Instruction）による精度向上[^1]や、Adaptive-RAGのように質問の複雑さに応じて自動的に適切な戦略を選択する機能の実装[^6]が考えられます。さらに、LangChainの提供する多様なコンポーネントを活用することで、より洗練されたPAR RAGシステムを構築することができるでしょう。

<div style="text-align: center">⁂</div>

[^1]: https://www.semanticscholar.org/paper/9fef9293e363ac4af5e72444c63823abae00d263

[^2]: https://aclanthology.org/2024.naacl-long.364.pdf

[^3]: https://python.langchain.com/docs/tutorials/rag/

[^4]: https://indepa.net/archives/4361

[^5]: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sequential.SimpleSequentialChain.html

[^6]: https://arxiv.org/abs/2403.14403

[^7]: https://bestoutcome.com/knowledge-centre/how-many-rags/

[^8]: https://python.langchain.com/v0.2/docs/tutorials/rag/

[^9]: https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langchain_chains

[^10]: https://arxiv.org/abs/2410.01782

[^11]: https://github.com/langchain-ai/rag-from-scratch

[^12]: https://www.restack.io/docs/langchain-knowledge-simple-sequential-chain-example-cat-ai

[^13]: https://arxiv.org/abs/2502.11175

[^14]: https://www.semanticscholar.org/paper/240856fa122cc30a0beb33b9ca24545eb75fc1a3

[^15]: https://arxiv.org/abs/2503.15879

[^16]: https://arxiv.org/abs/2504.16787

[^17]: https://zenn.dev/knowledgesense/articles/6d4d9bf52690a4

[^18]: https://arxiv.org/html/2502.14902v1

[^19]: https://cloud.google.com/use-cases/retrieval-augmented-generation

[^20]: https://www.youtube.com/watch?v=ztJrSBUuBGI

[^21]: https://www.semanticscholar.org/paper/23ad9714939aebb659959d6cc68bc691279c6c48

[^22]: https://arxiv.org/abs/2501.14998

[^23]: https://www.semanticscholar.org/paper/4c404a9d739c52abffa33b55a2fb0cef25e06cae

[^24]: https://arxiv.org/abs/2409.02361

[^25]: https://labelstud.io/blog/seven-ways-your-rag-system-could-be-failing-and-how-to-fix-them/

[^26]: https://zenn.dev/yuzame/articles/c3ed4063260f97

[^27]: https://aws.amazon.com/what-is/retrieval-augmented-generation/

[^28]: https://www.arxiv.org/pdf/2504.16787.pdf

[^29]: https://cloudsecurityalliance.org/blog/2023/11/22/mitigating-security-risks-in-retrieval-augmented-generation-rag-llm-applications

[^30]: https://qiita.com/ysv/items/82dd14ae7a1328ef5ee2

[^31]: https://research.ibm.com/blog/retrieval-augmented-generation-RAG

[^32]: https://openreview.net/pdf/3534d8616c60894945d89aa9d890a6dd5df97cb6.pdf

[^33]: https://deconvoluteai.com/blog/rag/failure-modes

[^34]: https://python.langchain.com/v0.2/docs/tutorials/rag/

[^35]: https://enterprisezine.jp/article/detail/21714

[^36]: https://www.projectmanager.com/blog/rag-status

[^37]: https://arxiv.org/html/2410.07176v1

[^38]: https://note.com/mizutory/n/ncc5e2f11b2dc

[^39]: https://arxiv.org/abs/2411.07021

[^40]: https://arxiv.org/abs/2410.09662

[^41]: https://arxiv.org/abs/2407.19619

[^42]: https://arxiv.org/abs/2412.05159

[^43]: https://milvus.io/ai-quick-reference/how-does-langchain-handle-multistep-reasoning-tasks

[^44]: https://cheatsheet.md/ja/langchain-tutorials/load-qa-chain-langchain

[^45]: https://www.youtube.com/watch?v=tcqEUSNCn8I

[^46]: https://qiita.com/t-hashiguchi/items/21ad182d448c3b5dff75

[^47]: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.router.multi_prompt.MultiPromptChain.html

[^48]: https://note.com/npaka/n/ncf1dbb190caf

[^49]: https://github.com/pixegami/langchain-rag-tutorial

[^50]: https://www.creationline.com/tech-blog/author/higuchi/77039

[^51]: https://wp-kyoto.net/langchain-js-develop-multi-step-llm-invocation/

[^52]: https://python.langchain.com/v0.1/docs/use_cases/question_answering/

[^53]: https://developer.ibm.com/tutorials/awb-create-langchain-rag-system-python-watsonx/

[^54]: https://qiita.com/tinymouse/items/4d359674f6b2494bb22d

[^55]: https://apxml.com/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/practice-multi-step-chain

[^56]: https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb

[^57]: https://www.semanticscholar.org/paper/2f58d0e43f43c0ade16645119c067a7f8cb56ed3

[^58]: https://arxiv.org/abs/2403.14258

[^59]: https://www.semanticscholar.org/paper/cbdbbe22ed60de1a58ea14cc0e6cd4daa24ced1c

[^60]: https://arxiv.org/abs/2308.01990

[^61]: https://nuco.co.jp/blog/article/7si1H9bM

[^62]: https://github.com/hwchase17/langchain/issues/3638

[^63]: https://www.youtube.com/watch?v=b2mNsbqYRs8

[^64]: https://www.youtube.com/watch?v=J7n9e0eSoKg

[^65]: https://github.com/langchain-ai/langchain/discussions/18456

[^66]: https://note.com/nice_phlox322/n/n49bcac4d83c7

[^67]: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sequential.SequentialChain.html

[^68]: https://zenn.dev/takanao/articles/9e3d0fc9f88008

[^69]: https://labelstud.io/blog/seven-ways-your-rag-system-could-be-failing-and-how-to-fix-them

[^70]: https://vectorize.io/building-fault-tolerant-rag-pipelines-strategies-for-dealing-with-api-failures/

[^71]: https://arxiv.org/html/2504.16787v1

[^72]: https://www.lasso.security/blog/rag-security

[^73]: https://arxiv.org/html/2504.16787

[^74]: https://www.semanticscholar.org/paper/54f85de4a086d7bb42d03342db9d3e5d1aeae25e

[^75]: https://arxiv.org/abs/2412.12447

[^76]: https://www.semanticscholar.org/paper/0da66fdf7e5095fc4c74b376fb404b37dad97380

[^77]: https://www.semanticscholar.org/paper/1d0b68891243777f05e391e5d075f0ffbdd3831f

[^78]: https://www.semanticscholar.org/paper/51504cd322c8524d08e1419e9a5ffdcd290a1e8a

[^79]: https://www.semanticscholar.org/paper/c0aec04ee86c0724d61c976f19590fbe9c615723

[^80]: https://zenn.dev/yumefuku/articles/llm-langchain-rag

[^81]: https://hackernoon.com/comprehensive-tutorial-on-building-a-rag-application-using-langchain

[^82]: https://python.langchain.com/docs/tutorials/rag/

[^83]: https://arxiv.org/abs/2409.04181

[^84]: https://www.semanticscholar.org/paper/107c21885aa6d957d02881ba0aaafaa463472ca6

[^85]: https://www.semanticscholar.org/paper/2ad8dbc163ce289e23e9c02b324c82d9c2fe8190

[^86]: https://api.python.langchain.com/en/latest/chains/langchain.chains.sequential.SimpleSequentialChain.html

[^87]: https://tecelit.hashnode.dev/lang-chain-chaining-simple-sequential-chain

[^88]: https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html

