"""
build_law_vectors.py

Parses Icelandic law HTML files from the `law_html/` folder and builds a FAISS vector store
that can be used for AI-assisted legal question answering (e.g. using LangChain RAG).

Installation:
1. Place HTML files in the folder: law_html/
2. Install dependencies:
   pip install langchain sentence-transformers faiss-cpu bs4
3. Run the script:
   python build_law_vectors.py
"""

import os
import re
import sys
import time
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = "law_html"
VECTOR_DIR = "vector_output"
BATCH_SIZE = 100  # Process documents in batches to show progress

def print_usage():
    print(__doc__)

def clean_text(text):
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Icelandic characters
    text = re.sub(r'[^\w\sáéíóúýþæöÁÉÍÓÚÝÞÆÖ.,;:!?()\-–—]', '', text)
    return text.strip()

def extract_articles_from_html(filepath: Path):
    """Extract articles from an HTML file with proper error handling."""
    try:
        logger.info(f"Processing file: {filepath.name}")
        
        # Read file with proper encoding
        with open(filepath, 'r', encoding='iso-8859-1') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, "html.parser")
        articles = []

        # Find all spans with IDs starting with 'G'
        g_spans = soup.find_all("span", id=re.compile(r'^G\d+$'))
        g_numbers = sorted([int(span['id'][1:]) for span in g_spans])
        
        if not g_numbers:
            logger.warning(f"No G-numbered spans found in {filepath.name}")
            return articles

        logger.info(f"Found {len(g_numbers)} G-numbered spans in {filepath.name}")
        
        for i in g_numbers:
            span_id = f"G{i}"
            next_span_id = f"G{i+1}"
            span = soup.find("span", id=span_id)
            if not span:
                continue

            article_title = None
            article_text = ""
            
            # Find the article title (usually in a <b> tag after the span)
            title_elem = span.find_next("b")
            if title_elem:
                article_title = clean_text(title_elem.get_text())
            
            # Get the article content
            current = span.next_sibling
            while current and not (hasattr(current, 'get') and current.get('id') == next_span_id):
                if isinstance(current, str):
                    article_text += current.strip() + " "
                elif current.name not in ['img', 'br']:  # Skip images and line breaks
                    article_text += clean_text(current.get_text()) + " "
                current = current.next_sibling

            if article_title and article_text.strip():
                articles.append(Document(
                    page_content=clean_text(article_text),
                    metadata={
                        "article": article_title,
                        "source_file": filepath.name,
                        "article_number": i
                    }
                ))

        logger.info(f"Found {len(articles)} articles in {filepath.name}")
        return articles
        
    except Exception as e:
        logger.error(f"Error processing {filepath.name}: {str(e)}")
        return []

def build_vectorstore():
    """Build the vector store from HTML files with progress tracking."""
    try:
        # Delete old vector store data if it exists
        if os.path.exists(VECTOR_DIR):
            logger.info(f"Deleting old vector store data from {VECTOR_DIR}...")
            for file in os.listdir(VECTOR_DIR):
                file_path = os.path.join(VECTOR_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {str(e)}")
            logger.info("Old vector store data deleted successfully")

        logger.info("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}  # Explicitly use CPU to avoid GPU memory issues
        )

        all_docs = []
        html_files = list(Path(DATA_DIR).glob("*.html"))
        total_files = len(html_files)
        
        if not html_files:
            logger.error(f"No HTML files found in {DATA_DIR}")
            return
        
        logger.info(f"\nFound {total_files} HTML files to process")
        
        # Process files with progress bar
        for idx, file in enumerate(tqdm(html_files, desc="Processing files"), 1):
            docs = extract_articles_from_html(file)
            all_docs.extend(docs)
            if idx % 10 == 0:  # Log progress every 10 files
                logger.info(f"Processed {idx}/{total_files} files, collected {len(all_docs)} articles")

        if not all_docs:
            logger.error("⚠️ No articles found in any files")
            return

        logger.info(f"\nCreating vector store with {len(all_docs)} articles...")
        
        # Process documents in batches with progress bar
        vectorstore = None
        start_time = time.time()
        
        for i in tqdm(range(0, len(all_docs), BATCH_SIZE), desc="Creating embeddings"):
            batch = all_docs[i:i + BATCH_SIZE]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embedding_model)
            else:
                vectorstore.add_documents(batch)
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nEmbedding creation took {elapsed_time:.2f} seconds")
        
        # Create output directory if it doesn't exist
        os.makedirs(VECTOR_DIR, exist_ok=True)
        
        logger.info(f"Saving vector store to {VECTOR_DIR}...")
        vectorstore.save_local(VECTOR_DIR)
        logger.info(f"✅ Vector store saved to: {VECTOR_DIR}")
        
    except Exception as e:
        logger.error(f"Error building vector store: {str(e)}")
        raise

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
    else:
        build_vectorstore()
