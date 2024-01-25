from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp
# https://python.langchain.com/docs/integrations/text_embedding/llamacpp
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# https://python.langchain.com/docs/integrations/vectorstores/faiss
from langchain_community.vectorstores import FAISS
# https://python.langchain.com/docs/integrations/llms/llamacpp#installation-with-windows

# https://python.langchain.com/docs/integrations/text_embedding/gpt4all

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

embeddings = LlamaCppEmbeddings(model_path="G:\AI\mistral-7b-instruct-v0.1.Q5_K_M.gguf", n_gpu_layers=20, n_threads=3, verbose=True, n_ctx=1000) # must be > chunk_size split doc

def save_rags_embeddings(index_filename) : 
    loader =  TextLoader(file_path="G:\AI\state_of_the_union2.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs = docs, embeddings = embeddings.embed_documents(docs))
    db.save_local("faiss_index")
    return db


def load_rags_embeddings(index_filename) :
    return FAISS.load_local("faiss_index", embeddings)

db = save_rags_embeddings("G:\AI\state_of_the_union2.txt")
#db = load_rags_embeddings("faiss_index")
query = "How much the federal government spends to keep the country safe?"
'''docs = db.similarity_search(query)
print(docs)'''

#retriever = db.as_retriever()
#docs = retriever.invoke(query)
#print(docs[0].page_content)