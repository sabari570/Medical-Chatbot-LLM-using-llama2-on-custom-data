from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Creating Embeddings for Each of The Text Chunks & storing
db=FAISS.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# Saving the vector database locally
db.save_local("saved_VDB")