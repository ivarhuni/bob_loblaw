"""
ask_law.py

A CLI tool for asking questions about Icelandic law using the vector store and local LLM.
The tool uses the previously created vector store to find relevant legal documents
and uses them as context for answering questions.

Usage:
    python ask_law.py
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTOR_DIR = "vector_output"
MODEL_DIR = "models"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
LAW_HTML_DIR = "law_html"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Mapping of import names to pip install names
required_packages = {
    "requests": "requests",
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "bs4": "beautifulsoup4",
    "tqdm": "tqdm"
}

def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"Package '{import_name}' not found. Installing '{pip_name}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

for import_name, pip_name in required_packages.items():
    ensure_package(import_name, pip_name)

def check_dependencies():
    """Check and install required Python packages."""
    print("\n🔍 Checking dependencies...")
    
    missing_packages = []
    for package, requirement in required_packages.items():
        try:
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(requirement)
    
    if missing_packages:
        print("\n📦 Installing missing dependencies...")
        for package in missing_packages:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Dependencies installed successfully!")

    # Install llama-cpp-python separately with specific options
    try:
        import llama_cpp
        print("✓ llama-cpp-python is installed")
    except ImportError:
        print("\n📦 Installing llama-cpp-python...")
        print("This might take a few minutes...")
        try:
            # Try to install the CPU-only version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "llama-cpp-python",
                "--no-cache-dir",
                "--verbose",
                "--force-reinstall",
                "--prefer-binary"
            ])
            print("✅ llama-cpp-python installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing llama-cpp-python: {e}")
            print("\nIf you see build errors, you may need to install Visual Studio Build Tools:")
            print("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("During installation, select 'Desktop development with C++'")
            sys.exit(1)

def download_model():
    """Download the Mistral model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model found at {MODEL_PATH}")
        return True

    print(f"\n📥 Model not found. Downloading {MODEL_NAME}...")
    print("This is a large file (about 4GB) and may take a while to download")
    print("Download speed will depend on your internet connection")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            # Using a longer timeout for the initial connection
            response = requests.get(MODEL_URL, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            print(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
            print("Starting download...")
            
            with open(MODEL_PATH, 'wb') as f, tqdm(
                desc="Downloading model",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            
            print("✅ Model downloaded successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("\n❌ All download attempts failed.")
                print("\nPlease try these solutions:")
                print("1. Check your internet connection")
                print("2. Try using a VPN if you're having connection issues")
                print("3. Download the model manually from:")
                print("   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
                print("   Then place it in the 'models' directory")
                return False

def check_vectorstore():
    """Check if vector store exists and is valid, if not run build_law_vectors.py."""
    vectorstore_valid = False
    
    if os.path.exists(VECTOR_DIR):
        # Check if directory is empty or contains required files
        files = os.listdir(VECTOR_DIR)
        if files and any(f.endswith('.faiss') for f in files):
            try:
                # Try to load the vector store to verify it's valid
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                FAISS.load_local(VECTOR_DIR, embedding_model)
                print(f"✓ Vector store found and validated at {VECTOR_DIR}")
                vectorstore_valid = True
            except Exception as e:
                print(f"⚠️ Vector store appears to be corrupted: {e}")
                vectorstore_valid = False

    if not vectorstore_valid:
        print("\n🔨 Vector store not found or invalid. Building it now...")
        print("This will process all HTML files and create embeddings")
        try:
            # Import and run build_law_vectors directly instead of using subprocess
            import build_law_vectors
            build_law_vectors.build_vectorstore()
            print("✅ Vector store built successfully!")
            return True
        except Exception as e:
            print(f"❌ Error building vector store: {e}")
            return False
    
    return True

def load_llm():
    """Load the local LLM model."""
    print("\n🤖 Loading language model...")
    if not os.path.exists(MODEL_PATH):
        if not download_model():
            return None

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=2000,
            top_p=1,
            verbose=False,
            n_ctx=4096,
        )
        print("✅ Language model loaded")
        return llm
    except Exception as e:
        print(f"❌ Error loading language model: {e}")
        return None

def load_vectorstore():
    """Load the FAISS vector store."""
    print("\n📚 Loading vector store...")
    if not os.path.exists(VECTOR_DIR):
        if not check_vectorstore():
            return None

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = FAISS.load_local(VECTOR_DIR, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded")
        return vectorstore
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def create_qa_chain(vectorstore, llm):
    """Create a QA chain with custom prompt."""
    template = """You are a specialized AI assistant for Icelandic law. Your role is to provide accurate, 
    well-structured legal information based on the provided context. Follow these guidelines:

    1. ALWAYS respond in Icelandic
    2. Use formal Icelandic legal terminology
    3. Structure your response with:
       - A clear, direct answer
       - Relevant legal provisions
       - Any important conditions or exceptions
    4. If the context doesn't contain enough information, say "Ég get ekki gefið nákvæma svarið út frá þessum gögnum"
    5. Always cite your sources using the article titles and file names provided
    6. If there are multiple relevant laws, explain how they relate to each other

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def format_source_link(file_name, article_title):
    """Format a source link for the terminal."""
    file_path = os.path.join(LAW_HTML_DIR, file_name)
    if os.path.exists(file_path):
        # Use PowerShell's file:/// protocol for clickable links
        abs_path = os.path.abspath(file_path)
        return f"[{article_title}] (file:///{abs_path})"
    return f"{article_title} ({file_name})"

def main():
    print("\n🚀 Starting Icelandic Law Q&A System")
    print("Initializing system components...")
    
    check_dependencies()
    
    vectorstore = load_vectorstore()
    if not vectorstore:
        return

    llm = load_llm()
    if not llm:
        return

    print("\n🔗 Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore, llm)
    print("✅ System ready!")

    print("\n🔍 Icelandic Law Q&A System")
    print("Type 'exit' or 'quit' to end the session")
    print("Type your question in Icelandic and press Enter\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ['exit', 'quit']:
            break

        if not question:
            continue

        print("\n🔎 Searching for relevant documents...")
        result = qa_chain.invoke({"query": question})
        
        print("\n📝 Answer:")
        print(result["result"])
        
        print("\n📚 Sources:")
        for doc in result["source_documents"]:
            link = format_source_link(doc.metadata['source_file'], doc.metadata['article'])
            print(f"- {link}")
        print()

if __name__ == "__main__":
    main() 