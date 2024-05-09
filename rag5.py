import streamlit as st
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import TextLoader


import os
openai_api_key = os.environ['OPENAI_API_KEY'] 
# Define the Streamlit app
def main():
    st.title("Question Answering about Innova")

    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://en.wikipedia.org/wiki/Toyota_Innova",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("mw-content-ltr mw-parser-output")
            )
        ),
    )

    # file_path = "/Users/raafid_mv/Documents/tesr/ragg.txt"
    # loader = TextLoader(file_path)

    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key = openai_api_key)

    # Define the format_docs function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Question input
    question = st.text_input("Ask your question here:")

    if st.button("Get Answer"):
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Question
        answer = rag_chain.invoke(question)

        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
