import os
import fitz
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)
CORS(app)

# Load PDFs
def load_pdfs(folder_path):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    doc = fitz.open(path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    docs.append({"content": text, "source": path})
                    print(f"✅ Loaded: {file}")
                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")
    return docs

# Prepare Chroma Vector Store
def prepare_vectorstore(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        sub_docs = splitter.create_documents([doc["content"]])
        for chunk in sub_docs:
            chunk.metadata = {"source": doc["source"]}
            chunks.append(chunk)

    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

# Load LLM - TinyLlama
def load_tinyllama_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.5, do_sample=True)
    return HuggingFacePipeline(pipeline=pipe)

# Load and prepare
print("📄 Loading and embedding PDFs...")
pdf_folder = r"C:\Users\srkra\OneDrive\Desktop\pdfs"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("./chroma_db/index"):
    print("📦 Found existing Chroma vector DB. Skipping re-embedding.")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    print("🔄 No existing Chroma DB. Processing and embedding...")
    docs = load_pdfs(pdf_folder)
    vectordb = prepare_vectorstore(docs, embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("🚀 Loading TinyLlama model...")
llm = load_tinyllama_model()

# Define Custom Prompt
prompt_template = """{context}

Q: {question}
A:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create custom QA chain
qa_llm_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
qa_chain = RetrievalQA(combine_documents_chain=qa_llm_chain, retriever=retriever)

# API Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"response": "Please provide a valid question."})
    
    response = qa_chain.invoke(question)
    response_text = response['result'].strip()
    formatted_response = f"🤖: {response_text}"
    return jsonify({"response": formatted_response})

if __name__ == '__main__':
    app.run(debug=True)
