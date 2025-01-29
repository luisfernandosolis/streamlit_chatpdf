import streamlit as st
from streamlit_chat import message
## gestionar variables de entorno
from dotenv import load_dotenv
import os
import base64


load_dotenv()

## langchain modules

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit_pdf_viewer import pdf_viewer

## groq

from langchain_groq import ChatGroq


## create session state variables
if "questions" not in st.session_state:
    st.session_state["questions"] = []

if "answers" not in st.session_state:
    st.session_state["answers"] = []

if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None



def displayPDF(upl_file, width):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8", 'ignore')

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)



def vectordb_from_file (pdf_file):
    pass
    ## read the file
    path_pdf_file = f"{pdf_file.name}"
    pdf_loader = PyPDFLoader(path_pdf_file)
    documents = pdf_loader.load()

    ## split text in chunks

    spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = spliter.split_documents(documents=documents)

    ## load embedding model
    embedding_model = OpenAIEmbeddings()

    ## save in vector database

    vector_db = FAISS.from_documents(chunks,embedding_model)
    return vector_db



with st.sidebar:
    st.title("RAG WITH OPENAI FAISS AND STREAMLIT")

    pdf_file = st.file_uploader("Uppload your PDF file", type=["pdf"])

    load_button = st.button(label="Let's go!", type="primary")
    clear_button = st.button(label="clear clat", type="secondary")

    if clear_button:
        st.session_state["questions"]=[]
        st.session_state["answers"]=["Hi, I'm your assistant, how can I help you?"]

    if load_button and pdf_file is not None:
        vector_db = vectordb_from_file(pdf_file)

        
        if vector_db:
            st.session_state["answers"].append("Hi, I'm your assistant, how can I help you?")
            st.session_state["vector_db"] = vector_db
            print("vector db created!")





## show conversation
chat_container = st.container()



## form for text input
input_container = st.container()

with input_container:
    with st.form(key="my_form", clear_on_submit=True):
        query = st.text_area("Message RAG", key="input", height=80)
        submit_button = st.form_submit_button(label="Submit")
    
    if query and submit_button:
        ## send questions to llm

        vector_db = st.session_state["vector_db"]


        prompt = """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                        Question: {question} 
                        Context: {context} 
                        Answer:
        """
        prompt_template = PromptTemplate(
            template=prompt
        )

        llm_groq = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            api_key="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"
        )

        retriever_db = vector_db.as_retriever()

        retrieve_qa = RetrievalQA.from_chain_type(
            llm=llm_groq,
            retriever =retriever_db,
            chain_type="stuff" # https://js.langchain.com/v0.1/docs/modules/chains/document/stuff/
        )

        answer = retrieve_qa.run(query)




        st.session_state["questions"].append(query)
        st.session_state["answers"].append(answer)


with chat_container:
    st.header("chat with your pdf!")

    question_messages = st.session_state["questions"]
    answers_messages = st.session_state["answers"]

    if answers_messages:
        for i in range(len(answers_messages)):
            message(answers_messages[i], key=str(i)+"_bot")
            if i<len(question_messages):
                message(question_messages[i],is_user=True, key=str(i)+"_user")



