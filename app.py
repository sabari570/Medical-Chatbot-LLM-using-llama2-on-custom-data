import streamlit as st
from streamlit_chat import message
from flask import Flask, jsonify, request
from src.helper import set_qa_chain
from src.prompt import *
from werkzeug.exceptions import ClientDisconnected

app= Flask(__name__)


# ----------------------------------------------------------------
# CODE BELOW IS FOR RETRIEVALQA A SIMPLE CHATBOT WITHOUT MEMORY
# ----------------------------------------------------------------
# embeddings = download_hugging_face_embeddings()

# loaded_VDB = FAISS.load_local("savedVDB\saved_FAISS_VDB", embeddings, allow_dangerous_deserialization=True)

# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Prompt for integrating memory into langchain
# # MEMORY_PROMPT = PromptTemplate(template=prompt_template2, input_variables=["context", "question", "chat_history"])

# chain_type_kwargs={"prompt": PROMPT}

# memory= ConversationBufferMemory(output_key="result", return_messages=True)

# llm = CTransformers(
#     model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
#     model_type="llama",
#     config={"max_new_tokens": 512, "temperature": 0.5}
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever= loaded_VDB.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True,
#     chain_type_kwargs=chain_type_kwargs,
#     memory=memory
# )
# ----------------------------------------------------------------

qa_chain = set_qa_chain()

# ------------------------------------------------
# CODE BELOW IS FOR API INTEGRATION
# -------------------------------------------------
# @app.route("/")
# def index():
#     return jsonify({"message": "API running successfully"})

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     try:
#         msg= request.form.get("msg")
#         query= msg
#         print(f"User: {query}")
#         result = qa_chain.invoke({"question": query})
#         print("Bot: ", result)
#         return jsonify({
#             "user": query,
#             "bot": result["answer"],
#         }), 200
    
#     except ClientDisconnected:
#         # Handle the client disconnection
#         app.logger.warning("Client disconnected during request processing.")
#         return jsonify({"error": "Client disconnected"}), 499
    
#     except Exception as e:
#         app.logger.error(f"An error occurred: {str(e)}")
#         return jsonify({"error": "An internal error occurred"}), 500


# if __name__ == "__main__":
#     app.run(debug=True, use_reloader=False)

# -------------------------------------------------


# ------------------------------------------------
# CODE BELOW IS FOR STREAMLIT APP
# -------------------------------------------------

st.title("Medical Chatbot 🧑🏽‍⚕️")
def conversation_chat(query):
    result= qa_chain.invoke({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input= st.text_input("Question:", placeholder="Ask about your anything related to health", key="input")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output= conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


initialize_session_state()
display_chat_history()
