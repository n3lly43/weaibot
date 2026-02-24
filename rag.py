import glob
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def docs_return(directory_path, flag):
    docx_file_pattern = '*.docx'
    pdf_file_pattern = '*.pdf'
    txt_file_pattern = '*.txt'

    docx_file_paths = glob.glob(directory_path + docx_file_pattern)
    pdf_file_paths = glob.glob(directory_path + pdf_file_pattern)
    txt_file_paths = glob.glob(directory_path + txt_file_pattern)

    all_doc, all_doc2 = [], []

    for x in docx_file_paths:
        loader = Docx2txtLoader(x)
        documents = loader.load()
        all_doc.extend(documents)
        all_doc2.append(str(documents[0].page_content))

    for x in pdf_file_paths:
        loader = PyPDFLoader(x, extract_images=True)
        docs_lazy = loader.lazy_load()
        documents = []
        for doc in docs_lazy:
            documents.append(doc)
        all_doc.extend(documents)
        all_doc2.append(str(documents[0].page_content))

    for x in txt_file_paths:
        loader = TextLoader(x)
        documents = loader.load()
        all_doc.extend(documents)
        all_doc2.append(str(documents[0].page_content))

    docs = '\n\n'.join(all_doc2)

    return all_doc if flag == 0 else docs

def get_text_splitter(splitter_type='character',
                      chunk_size=500,
                      chunk_overlap=30,
                      separator="\n",
                      max_tokens=1000):
    if splitter_type == 'character':
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    elif splitter_type == 'recursive':
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'token':
        return TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Unsupported splitter type. Choose from 'character', 'recursive', or 'token'.")
    
# Function to get or download the embedding model
def get_embedding_model(model_name):
    model_kwargs = {"device": "cpu"}

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Retriever using Chroma and HuggingFace embeddings
def retrieve_docs_chroma(docs_path, 
                          embeddings, 
                           splitter_type='character', 
                           chunk_size=1500, 
                           chunk_overlap=30, 
                           separator="\n", 
                           max_tokens=1000):
    """retrieve WEAI reference docs to use as context for the agents"""
    all_doc = docs_return(docs_path, 0)

    # Use the splitter parameters
    text_splitter = get_text_splitter(splitter_type=splitter_type,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    separator=separator,
                                    max_tokens=max_tokens)

    # Split the documents using the text splitter
    docs = text_splitter.split_documents(documents=all_doc)

    # Create a Chroma vector database
    return Chroma.from_documents(docs, embeddings, persist_directory="chroma_data")

def retrieve_refs_chroma(refs_path, embeddings):    
    loader = CSVLoader(file_path=refs_path,
                    source_column="Description (what it contains and what it's useful for)")
    refs = loader.load()

    return Chroma.from_documents(
        refs, embeddings, persist_directory='chroma_data'
    )

def create_rag_db(docs_path,
                  refs_path,
                  model_name="sentence-transformers/all-mpnet-base-v2"):    
    # Load or download the embedding model
    embeddings = get_embedding_model(model_name)

    docs_chroma = retrieve_docs_chroma(docs_path, embeddings)
    refs_chroma = retrieve_refs_chroma(refs_path, embeddings)

    return docs_chroma, refs_chroma
