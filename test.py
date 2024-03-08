import torch
import pinecone

from pinecone import Pinecone, ServerlessSpec

# from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.indexes import VectorstoreIndexCreator
import os

from dotenv import load_dotenv

load_dotenv()

directory = "Datasets"


def get_data():
    try:
        files = os.listdir("Datasets")
        loaders = [CSVLoader("Datasets/" + x, encoding="utf-8") for x in files]
        documents = [loader.load() for loader in loaders]
        all_documents = []
        for doc in documents:
            all_documents += doc
        return all_documents

    except Exception as e:
        print(e)
        loader1 = CSVLoader("Datasets\output.csv", encoding="utf-8")
        loader2 = CSVLoader("Datasets\Graduate.csv", encoding="utf-8")

        documents1 = loader1.load()
        documents2 = loader2.load()
        all_documents = documents1 + documents2
        return all_documents


documents = get_data()

print("Data Loaded Successfully !!")


def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)
print(len(docs))

torch.cuda.empty_cache()
# Set device to your GPU (assuming index 0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create embeddings for both documents
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# spec = ServerlessSpec(cloud="gcp-starter", region="us-central1")
# pinecone.init(
#     api_key=os.environ.get("PINE_CONE_API_KEY"),  # find at app.pinecone.io
#     environment="gcp-starter",  # next to api key in console
# )
index_name = "careerdrishti"

# index = pinecone.Index(
#     index_name, host="https://careerdrishti-uls0qwc.svc.gcp-starter.pinecone.io"
# )

# index = pc.from_documents(docs, embeddings, index_name=index_name)

from langchain_pinecone import PineconeVectorStore


index = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


def get_similiar_docs(query, k=4, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs


message = "I want to pursue law, can you connect me to a counselor"
# print(get_similiar_docs(message))
print("Embeddings were created Successfully!!!")

# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferWindowMemory,
    ConversationBufferMemory,
)
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import openai
import streamlit as st
from streamlit_chat import message

# from test import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=1, return_messages=True
    )

# memory = ConversationBufferMemory()

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""You are a world class career guidance expert coversational bot that gives best career advice.
Answer the question as truthfully as possible using the provided context,
and if the answer is not contained within the text below, try to mimic the knowledgebase and generate responsenes'.
"""
)


human_msg_template = HumanMessagePromptTemplate.from_template(
    template="""{input} 
                here is some context about the current conversation : """
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        system_msg_template,
        human_msg_template,
        MessagesPlaceholder(variable_name="history"),
    ]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")


st.title("Vriddhi - Your personal career counselor")

conversation = ConversationChain(
    memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True
)
# conversation = LLMChain(
#     memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True
# )

# from InstructorEmbedding import INSTRUCTOR

# model = INSTRUCTOR("hkunlp/instructor-base")

# from openai import OpenAI

# client = OpenAI()


# def query_refiner(conversation, query):
#     print(conversation)
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#         temperature=0.7,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#     )
#     print(response)
#     try:
#         return response["choices"][0]["text"]
#     except Exception as e:
#         print(e)
#         return query


# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return (
#         result["matches"][0]["metadata"]["text"]
#         + "\n"
#         + result["matches"][1]["metadata"]["text"]
#     )


# def get_conversation_string():
#     conversation_string = ""
#     for i in range(len(st.session_state["responses"]) - 1):
#         conversation_string += "Human: " + st.session_state["requests"][i] + "\n"
#         conversation_string += "Bot: " + st.session_state["responses"][i + 1] + "\n"
#     return conversation_string


response_container = st.container()
textcontainer = st.container()

with textcontainer:

    def clear_input():
        st.session_state["input"] = ""

    query = st.text_input("Query: ", key="input")

    # submit_button = st.button( "", type="hidden", on_click=lambda x: st.session_state["input"]="", key="submit")

    if query:
        with st.spinner("typing..."):
            # conversation_string = get_conversation_string()
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query: ")
            # st.write(refined_query)

            knowledge = get_similiar_docs(query)
            Template = f"""You are a world class career guidance expert coversational bot that gives best career advice. I will share a prospect's message
1/ Response should be very similar or even identical to the past best practies, in terms
2/ If the best practice are irrelevant, then try to mimic the style of the best practice
3/ Also provide some relevant point of contact information if possible
Below is a message I received from the prospect:
{query}
Here is a list of best practies of how we normally respond to prospect in similar scenario
{knowledge}
Below is the context of the conversation happening between ai and user undrstand it and then Please write the best response which is accurate to what user have asked."""
            response = conversation.predict(input=Template)
            # response = conversation.run(knowledge=knowledge, input=query)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        # st.session_state["input"] = ""


with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(
                    st.session_state["requests"][i], is_user=True, key=str(i) + "_user"
                )
