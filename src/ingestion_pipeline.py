import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
# from langchain_community.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Ingestion pipeline module loaded.")

if __name__ == "__main__":
    main()

# Load Files

# Chunk Files

# Embed and store in Vector DB


# def setup_ingestion_pipeline(documents_path: str, persist_directory: str):
#     """
#     Set up a basic RAG ingestion pipeline with LangChain.
    
#     Args:
#         documents_path: Path to documents directory
#         persist_directory: Path to store vector database
#     """
#     # Load documents
#     loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=TextLoader)
#     documents = loader.load()
    
#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # Create embeddings and vector store
#     embeddings = OpenAIEmbeddings()
#     vector_store = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=persist_directory
#     )
    
#     return vector_store

# if __name__ == "__main__":
#     vector_store = setup_ingestion_pipeline(
#         documents_path="./documents",
#         persist_directory="./chroma_db"
#     )
#     print("Ingestion pipeline completed!")