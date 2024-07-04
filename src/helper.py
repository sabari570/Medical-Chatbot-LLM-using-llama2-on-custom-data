# This file is used to implement the modular coding, such that we keep all the functions like these all in one file
# for keeping proper file management

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from src.prompt import *
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from transformers import TextStreamer

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# A helper function to return qa chain
def set_qa_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    embeddings= download_hugging_face_embeddings()
    streamer = TextStreamer(embeddings, skip_prompt=True, skip_special_tokens=True)
    loaded_VDB = FAISS.load_local("research\saved_VDB", embeddings, allow_dangerous_deserialization=True)
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={"max_new_tokens": 1000, "temperature": 0.5},
        streamer=streamer
    )
    llama_prompt= PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template_with_chat_history
    )
    qa_chain= ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=loaded_VDB.as_retriever(search_kwargs={'k': 2}),
        memory=memory, 
        combine_docs_chain_kwargs={'prompt': llama_prompt}
    )

    return qa_chain
