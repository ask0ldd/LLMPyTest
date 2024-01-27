from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp
# https://python.langchain.com/docs/integrations/text_embedding/llamacpp
# from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import DocArrayHnswSearch
# https://python.langchain.com/docs/integrations/vectorstores/faiss
from langchain_community.vectorstores import FAISS
from torch import cuda
# https://python.langchain.com/docs/integrations/llms/llamacpp#installation-with-windows
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# https://python.langchain.com/docs/integrations/text_embedding/gpt4all

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

'''embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)'''

def get_retriever():
    documents = TextLoader(file_path="G:\AI\state_of_the_union.txt", encoding="utf-8").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    db = DocArrayHnswSearch.from_documents(docs, embeddings, work_dir="hnswlib_store/", n_dim=1536)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    return retriever

def get_retriever2():
    documents = TextLoader(file_path="G:\AI\state_of_the_union.txt", encoding="utf-8").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embed_model)
    db.save_local("faiss_index")
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})


def save_rags_embeddings(index_filename) : 
    documents = TextLoader(file_path="G:\AI\state_of_the_union.txt", encoding="utf-8").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embed_model)
    db.save_local("faiss_index")
    return db


def load_rags_embeddings(index_filename) :
    return FAISS.load_local("faiss_index", embed_model)

def new_llmchain(_retriever) :
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path="G:\AI\mistral-7b-instruct-v0.1.Q5_K_M.gguf",
        temperature=0.1,
        max_tokens=2048,
        n_threads=3,
        n_gpu_layers=16,
        n_ctx=2048,
        n_batch = 2048, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        streaming=True,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=_retriever, memory=memory, verbose=True
    )

    return qa_chain



# db = save_rags_embeddings("G:\AI\state_of_the_union.txt")

# db = load_rags_embeddings("faiss_index")

query = "How much does the federal government spend to keep the country safe?"

'''docs = db.similarity_search(query)
print('similarity :')
print(docs)'''

'''retriever = db.as_retriever() # search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
docs2 = retriever.invoke(query)
print('retriever : ')
print(docs2[0].page_content)'''

retriever = get_retriever()

llm_chain = new_llmchain(retriever)

template = "{question}"
# prompt = PromptTemplate(template=template, input_variables=["which american forces have been mobilized to protect the NATO countries?"])
prompt = """which american forces have been mobilized to protect the NATO countries?"""

response = llm_chain.invoke(prompt)

print(response)