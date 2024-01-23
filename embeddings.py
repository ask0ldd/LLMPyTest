from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.evaluation import load_evaluator, EmbeddingDistance
import numpy as np

embeddings = LlamaCppEmbeddings(model_path="G:\AI\mistral-7b-instruct-v0.1.Q5_K_M.gguf", n_threads=3, verbose=False)

evaluator = load_evaluator("pairwise_embedding_distance", distance_metric=EmbeddingDistance.COSINE, embeddings=embeddings)

documents = ["cat", "dog", "car"]

text = "This is a test document."

def getEmbeddingsQuery(text) : 
    return embeddings.embed_query(text)

def getEmbeddingsDocuments(texts) :
    return embeddings.embed_documents(texts)

def getEmbeddingsDistance(pairA, pairB) :
    return evaluator.evaluate_string_pairs(prediction=pairA, prediction_b=pairB)

def getEmbeddingsDistanceNP(pairA, pairB) :
    return np.linalg.norm(np.array(getEmbeddingsQuery(pairA)) - np.array(getEmbeddingsQuery(pairB)))

print(getEmbeddingsQuery(text))
print(getEmbeddingsDocuments(documents))
print(getEmbeddingsDistance("cat", "car"))
print(getEmbeddingsDistance("cat", "because"))
print(getEmbeddingsDistance("man", "king"))
print(getEmbeddingsDistanceNP("cat", "car"))
print(getEmbeddingsDistanceNP("cat", "because"))
print(getEmbeddingsDistanceNP("man", "king"))
print(np.linalg.norm(np.array([0,1]) - np.array([1,0])))