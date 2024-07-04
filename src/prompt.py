# This file is made for writing templates in this
prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt_template_with_chat_history = """You are a helpful, respectful and honest assistant.
Answer the question in your own words as truthfully as possible from the context given to you.
If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
If questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"

Context: {context}


Chat history: {chat_history}

Human: {question}
Assistant:"""