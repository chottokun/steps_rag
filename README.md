# Steps-RAG: PDFベース段階的RAG QAシステムのお試し

## 概要

Steps-RAGは、PDFファイルから知識ベースを自動構築し、段階的な検索・回答・レビュー・統合を行うRAG（Retrieval Augmented Generation）型QAシステムです。StreamlitベースのUIで、ユーザーの質問に対し、LLM（大規模言語モデル）とベクトル検索を組み合わせて、根拠のある回答を生成を目指します。本レポジトリは、原理確認の実験的なものです。

- PDFを分割・埋め込み・ChromaDBへ格納
- ユーザー質問を複数の検索ステップに分解
- 各ステップごとに根拠付きで回答
- 回答の妥当性をLLMでレビュー
- 全体を統合し最終回答を生成

## 考え方
以下を参考にしました。ただし、論文中の実装を必ずしも正しく実装するものではなく、フローを参考にした程度です。
- [CREDIBLE PLAN-DRIVEN RAG METHOD FOR MULTI-HOP
QUESTION ANSWERING](https://arxiv.org/pdf/2504.16787)


## ディレクトリ構成

```
par_rag/
├── main.py
├── st_par_graph.py   # Streamlitアプリ本体 langgraph
├── st_par_chain.py   # langchain バージョン
├── par._samplepy
├── pdfs/            # PDFファイル格納ディレクトリ
├── chroma_db/       # ChromaDBデータ
├── pyproject.toml
├── uv.lock
└── README.md
```

## 必要環境

- Python 3.10以上
- Linux（他OSでも動作する可能性あり）
- pip

### 主要ライブラリ
- streamlit
- langchain
- langchain_community
- langchain_chroma
- langchain_ollama
- langchain_huggingface
- chromadb
- torch

## セットアップ手順

1. リポジトリをクローン

```bash
git clone <このリポジトリのURL>
cd par_rag
```

2. Python仮想環境の作成・有効化（推奨）

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. 依存パッケージのインストール

#### uvを使う場合（推奨）
```bash
uv pip install -r requirements.txt
```

#### pipを使う場合
```bash
pip install -r requirements.txt
```

4. PDFファイルを`pdfs/`ディレクトリに配置

5. LLM/埋め込みモデルの設定

- デフォルトはOllamaローカルLLM（`gemma3:4b-it-qat`）とHuggingFace埋め込み（`intfloat/multilingual-e5-small`）
- Azure OpenAIを使いたい場合は`st_par_rev2.py`の`get_config()`で`USE_AZURE=True`に変更
- Ollamaを使う場合は[Ollama公式](https://ollama.com/)でモデルをダウンロード・起動しておく

6. ChromaDBの初期化（初回またはPDF追加時）

Streamlitアプリ上で「ChromaDB再構築」ボタンを押すと自動で構築されます。

## 実行方法

```bash
streamlit run st_par_rev2.py
```

ブラウザで`http://localhost:8501`にアクセス。

## 使い方

1. PDFを`pdfs/`に入れる
2. 必要なら「ChromaDB再構築」ボタンを押す
3. 質問を入力し「回答生成」ボタンを押す
4. 各ステップの検索・回答・レビュー・最終回答が順に表示されます

## 機能詳細

- **Plan**: 質問を複数の検索ステップに分解
- **Retrieve & Answer**: 各ステップごとにベクトル検索→LLMで根拠付き回答
- **Review**: 各中間回答の妥当性をLLMで自己レビュー
- **Aggregate**: 全体を統合し最終回答を生成

## カスタマイズ

- `get_config()`でモデルやパスを変更可能
- ステップ分割やレビューのプロンプトは調整して利用します。
- PDF分割サイズや検索件数も適宜調整してください。

## 注意事項
- streamlit空のpytorch呼び出しでエラーが発生したため、対策を実施しています。環境によっては不要です。
- LLMや埋め込みモデルのAPIキー・ローカルサーバー起動は各自でご用意ください
- 大きなPDFや複雑な質問では処理に時間がかかる場合があります
- 本システムはテストの一環です。


## 参考

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)
