import argparse
from loguru import logger
from dotenv import load_dotenv
import os
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from myembedder import MyEmbedder
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
api_key = os.getenv("GEMINI_API_KEY")

def main():
    # argparse là thư viện chuẩn của Python (có sẵn, không cần cài) — dùng để xử lý các tham số dòng lệnh (command-line arguments).
    # python main.py --input data.txt --verbose ==> thì argparse chính là công cụ giúp bạn đọc và quản lý mấy cái --input, --verbose đó trong code.
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function= MyEmbedder()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Debug: Print similarity scores
    print(f"\nFound {len(results)} results:")
    for i, (doc, score) in enumerate(results):
        print(f"  Result {i+1}: score={score:.4f}")

    if len(results) == 0 or results[0][1] < 0.05: 
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        api_key=api_key
    )
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()