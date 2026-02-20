import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from Bio import Entrez
import time
import tempfile

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    .source-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.title("🏥 Medical Research Assistant")
st.markdown("*AI-powered research tool using RAG (Retrieval-Augmented Generation)*")
st.markdown("---")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    # api_key = st.text_input(
    #     "🔑 Groq API Key",
    #     type="password",
    #     help="Get your free API key from console.groq.com"
    # )
    
    st.markdown("---")
    
    # Input method selection
    st.subheader("📚 Data Source")
    input_method = st.radio(
        "Choose your data source:",
        ["📁 Upload PDFs", "🔗 Download from URLs", "🔬 Search PubMed"],
        help="Select how you want to provide research papers"
    )
    
    st.markdown("---")
    
    # Help section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Steps:**
        1. Enter your Groq API key above
        2. Choose a data source
        3. Provide your research papers
        4. Ask your question
        5. Get AI-powered answers!
        
        **Tips:**
        - Upload multiple PDFs for better results
        - Be specific with your questions
        - Check the sources in the answer
        """)
    
    st.markdown("---")
    
    # Links
    st.markdown("### 🔗 Quick Links")
    st.markdown("[Get Groq API Key](https://console.groq.com)")
    st.markdown("[PubMed Database](https://pubmed.ncbi.nlm.nih.gov/)")
    st.markdown("[GitHub Repo](#)")

# ========== COPY YOUR FUNCTIONS HERE ==========

def load_pdfs_from_uploaded_files(uploaded_files):
    """Load PDFs from Streamlit uploaded files"""
    document_sentences = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📄 Processing: {uploaded_file.name}")
        
        try:
            reader = PdfReader(uploaded_file)
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                
                if text:
                    sentences = text.split('.')
                    
                    for sentence in sentences:
                        sentence_clean = sentence.strip()
                        
                        if len(sentence_clean) > 20:
                            document_sentences.append({
                                'text': sentence_clean,
                                'source': uploaded_file.name,
                                'page': page_num
                            })
            
            st.success(f"✓ Loaded {uploaded_file.name}: {len(reader.pages)} pages")
            
        except Exception as e:
            st.error(f"✗ Error processing {uploaded_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.empty()
    progress_bar.empty()
    
    return document_sentences


def search_pubmed(query, max_results=10):
    """Search PubMed for papers"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"🔍 Searching PubMed for: '{query}'")
    
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            status_text.error("❌ No papers found")
            return []
        
        status_text.text(f"✅ Found {len(id_list)} papers, fetching details...")
        
        papers = []
        
        for i, pmid in enumerate(id_list):
            time.sleep(0.5)
            
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=pmid,
                    rettype="abstract",
                    retmode="xml"
                )
                
                records = Entrez.read(handle)
                handle.close()
                
                if not records.get('PubmedArticle'):
                    continue
                
                article = records['PubmedArticle'][0]['MedlineCitation']['Article']
                
                title = article.get('ArticleTitle', 'No title')
                abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                abstract = ' '.join([str(part) for part in abstract_parts]) if abstract_parts else 'No abstract'
                
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                papers.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'link': link,
                    'text': f"{title}\n\n{abstract}"
                })
                
                st.success(f"✓ Fetched: {title[:50]}...")
                
            except Exception as e:
                st.warning(f"✗ Error fetching paper {pmid}: {e}")
            
            progress_bar.progress((i + 1) / len(id_list))
        
        status_text.empty()
        progress_bar.empty()
        
        return papers
        
    except Exception as e:
        st.error(f"❌ PubMed search failed: {e}")
        return []


def create_sentence_from_pubmed(papers):
    """Convert PubMed papers to sentences"""
    document_sentences = []
    
    for paper in papers:
        text = paper['text']
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            
            if len(sentence_clean) > 20:
                document_sentences.append({
                    'text': sentence_clean,
                    'source': f"PubMed: {paper['title'][:50]}...",
                    'page': f"PMID: {paper['pmid']}",
                    'link': paper['link']
                })
    
    return document_sentences


def tfidf_search(question, document_sentences, top_n=5):
    """TF-IDF search"""
    texts = [doc['text'] for doc in document_sentences]
    
    with st.spinner(f"🔍 Analyzing {len(texts)} sentences with TF-IDF..."):
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        all_documents = [question] + texts
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        
        question_vector = tfidf_matrix[0]
        sentence_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(question_vector, sentence_vectors)[0]
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'text': document_sentences[idx]['text'],
                    'score': round(similarities[idx], 4),
                    'source': document_sentences[idx]['source'],
                    'page': document_sentences[idx]['page']
                })
        
        return results


def generate_research_answer(question, retrieved_sentences):
    """Generate AI answer"""
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    context_parts = []
    for i, sentence in enumerate(retrieved_sentences, 1):
        context_parts.append(
            f"[Source {i}: {sentence['source']}, Page {sentence['page']}]\n"
            f"{sentence['text']}\n"
        )
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are a medical research assistant. Based on the following research excerpts, provide a clear, evidence-based answer to the question.

QUESTION: {question}

RESEARCH EXCERPTS:
{context}

INSTRUCTIONS:
1. Synthesize the information from all sources
2. Provide a comprehensive answer
3. Cite sources using [Source 1], [Source 2], etc.
4. If the excerpts don't fully answer the question, say so
5. Keep the answer focused and relevant

ANSWER:"""
    
    with st.spinner("🤖 Generating AI answer..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
    
    return answer


# ========== MAIN UI ==========

# Question input (always visible)
question = st.text_input(
    "💬 Enter your research question:",
    placeholder="e.g., What is heat surge in India?",
    help="Ask any medical research question"
)

# Input method-specific UI
document_sentences = None

if input_method == "📁 Upload PDFs":
    st.subheader("📁 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more research papers in PDF format"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} file(s) uploaded")

elif input_method == "🔗 Download from URLs":
    st.subheader("🔗 Enter PDF URLs")
    st.info("⚠️ This feature requires server-side processing. Use 'Upload PDFs' instead for now.")
    urls_text = st.text_area(
        "Enter URLs (one per line):",
        placeholder="https://example.com/paper1.pdf\nhttps://example.com/paper2.pdf",
        height=150
    )

elif input_method == "🔬 Search PubMed":
    st.subheader("🔬 PubMed Search Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pubmed_email = st.text_input(
            "📧 Your Email (required by PubMed)",
            help="PubMed requires an email for API access"
        )
    
    with col2:
        max_papers = st.number_input(
            "📊 Number of papers",
            min_value=1,
            max_value=20,
            value=5,
            help="How many papers to retrieve from PubMed"
        )

# Search button
st.markdown("---")

if st.button("🔍 Search & Analyze", type="primary", use_container_width=True):
    
    # Validation
    
    if not question:
        st.error("❌ Please enter a research question")
    else:
        # Process based on input method
        try:
            if input_method == "📁 Upload PDFs":
                if not uploaded_files:
                    st.error("❌ Please upload at least one PDF file")
                else:
                    with st.spinner("Processing PDFs..."):
                        document_sentences = load_pdfs_from_uploaded_files(uploaded_files)
            
            elif input_method == "🔬 Search PubMed":
                if not pubmed_email:
                    st.error("❌ Please enter your email for PubMed")
                else:
                    Entrez.email = pubmed_email
                    papers = search_pubmed(question, max_results=max_papers)
                    
                    if papers:
                        document_sentences = create_sentence_from_pubmed(papers)
                        st.success(f"✅ Extracted {len(document_sentences)} sentences from {len(papers)} papers")
            
            # Continue if we have documents
            if document_sentences and len(document_sentences) > 0:
                
                # TF-IDF Search
                st.markdown("---")
                st.subheader("🔍 Retrieval Results")
                
                retrieved_sentences = tfidf_search(question, document_sentences, top_n=5)
                
                if not retrieved_sentences:
                    st.warning("❌ No relevant content found")
                else:
                    # Show retrieved context
                    with st.expander("📄 View Retrieved Context", expanded=False):
                        for i, result in enumerate(retrieved_sentences, 1):
                            st.markdown(f"**Source {i}:** {result['source']} (Page {result['page']})")
                            st.markdown(f"*Relevance Score: {result['score']:.4f}*")
                            st.text(result['text'][:200] + "...")
                            st.markdown("---")
                    
                    # Generate answer
                    st.markdown("---")
                    st.subheader("🤖 AI-Generated Answer")
                    
                    try:
                        answer = generate_research_answer(question, retrieved_sentences)
                        
                        # Display answer in a nice box
                        st.markdown(f"""
                        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;color:#000000'>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sources
                        st.markdown("---")
                        st.subheader("📚 Sources Used")
                        
                        for i, result in enumerate(retrieved_sentences, 1):
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"**{i}.** {result['source']}")
                                    st.caption(f"Page: {result['page']}")
                                with col2:
                                    st.metric("Score", f"{result['score']:.4f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {e}")
            
            elif document_sentences is not None:
                st.error("❌ No text could be extracted from the documents")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🏥 Medical Research Assistant | Built with Streamlit & Groq AI | Powered by TF-IDF Retrieval</p>
    <p><small>⚠️ For research purposes only. Always verify information with healthcare professionals.</small></p>
</div>
""", unsafe_allow_html=True)