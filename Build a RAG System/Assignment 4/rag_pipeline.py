import os
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DATA_PATH = "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_path: str = DATA_PATH) -> List:
    documents = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No .txt files found in the data folder.")

    return documents


def split_documents(documents: List) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(chunks: List):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_rag_pipeline():
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


def retrieve_documents(query: str) -> Tuple[List, str]:
    retriever = build_rag_pipeline()
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context


def generate_simple_answer(query: str, context: str) -> str:
    query_lower = query.lower()

    if not context.strip():
        return "Not found in dataset."

    if "cnn" in query_lower or "convolutional neural network" in query_lower:
        return (
            "A CNN is a deep learning model used mainly for images and other structured data. "
            "It uses convolutional layers to detect patterns, pooling layers to reduce size, "
            "and fully connected layers to make final predictions."
        )

    if "rnn" in query_lower or "recurrent neural network" in query_lower:
        return (
            "An RNN is a neural network designed for sequential data such as text, speech, "
            "and time series. It keeps a hidden state as memory, but standard RNNs suffer "
            "from vanishing gradients, which led to LSTM and GRU."
        )

    if "transformer" in query_lower:
        return (
            "A Transformer is a deep learning model that uses attention mechanisms to process "
            "all tokens in parallel. It is widely used in NLP tasks and performs better than "
            "RNNs on long-range dependencies."
        )

    if "difference" in query_lower or "compare" in query_lower:
        return (
            "CNNs are mainly used for spatial data such as images, RNNs are used for sequential "
            "data, and Transformers use attention to process sequences more efficiently and "
            "handle long-range relationships better."
        )

    return f"Answer based on retrieved context:\n\n{context[:500]}..."