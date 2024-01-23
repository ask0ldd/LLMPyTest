from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.embeddings.llamacpp import LlamaCppEmbeddings

embeddings = LlamaCppEmbeddings(model_path="G:\AI\mistral-7b-instruct-v0.1.Q5_K_M.gguf", n_threads=3)

doc_result = embeddings.embed_documents(["cat", "dog", "car"])

'''text = "This is a test document."

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])

print(query_result)'''
print(doc_result)