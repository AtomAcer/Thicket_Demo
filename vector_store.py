# Package imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever

# Import the read module
from read import read

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=1250,
    chunk_overlap=250
)

# Initialize SentenceTransformer embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Set a base directory for all file operations
base_directory = "data"  # You can change this to any directory you prefer

def create_new_collection_streamlit(collection_name_str=None, pdf_file=None):
    # Ensure file paths are within the base directory
    pdf_path = f"{base_directory}/{pdf_file}.pdf"
    text_path = f"{base_directory}/{pdf_file}.txt"

    # Convert PDF to text, saving in the same directory
    read([pdf_path])

    # Load the document and split it into chunks
    loader = TextLoader(text_path)
    documents = loader.load()

    # Apply the text splitter to the documents
    splits = text_splitter.split_documents(documents)

    return splits

def load_BM25Retriever(filepath):
    # Ensure the filepath is within the base directory
    text_path = f"{base_directory}/{filepath}"

    # Load the document and split it into chunks
    loader = TextLoader(text_path)
    documents = loader.load()

    # Apply the text splitter to the documents
    splits = text_splitter.split_documents(documents)

    return BM25Retriever.from_documents(splits)
