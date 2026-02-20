import os                                #importing this lib by which code can get accees to pdfs in system
from groq import Groq
from PyPDF2 import PdfReader             #this library for reading pdf(python inbuilt)
import nltk                              #natural lang toolkit for call its toll stopword lib which will be predifined set for filtering
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import requests
from Bio import Entrez
import time


def search_pubmed(query, max_results=10):
    """
    Search PubMed for papers
    
    Parameters:
    - query: Search terms (e.g., "heat wave India health")
    - max_results: How many papers to find
    
    Returns:
    - List of paper metadata (title, abstract, link)
    """
    print(f"🔎Searching PubMed for: '{query}")

    try:
        handle=Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record=Entrez.read(handle)
        handle.close()

        id_list =record["IdList"]
        if not id_list:
            print("❌ No papers found")
            return[]
        print(f"✅ Found {len(id_list)} papers")

        papers=[]

        for pmid in id_list:
            time.sleep(0.5)

            try:
                handle=Entrez.efetch(
                    db="pubmed",
                    id=pmid,
                    rettype="abstract",
                    retmode="xml"
                )

                records= Entrez.read(handle)
                handle.close()

                article=records['PubmedArticle'][0]['MedlineCitation']['Article']

                title= article.get('ArticleTitle','No title')

                abstract_parts= article.get('Abstract', {}).get('AbstractText', [])
                abstract=' '.join([str(part) for part in abstract_parts])

                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                papers.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'link': link,
                    'text': f"{title}\n\n{abstract}"  # Combined for processing
                })

                print(f"  ✓ {title[:60]}...")

            except Exception as e:
                print(f"  ✗ Error fetching paper {pmid}: {e}")

        return papers
    except Exception as e:
        print(f"❌ PubMed search failed: {e}")
        return []
    

def create_sentence_from_pubmed(papers):
    """Convert PubMed papers to sentence format for TF-IDF"""
    document_sentences=[]


    for paper in papers:

        text = paper['text']
        sentences = text.replace('?','.').replace('!','.').split('.')

        for sentence in sentences:
            sentence_clean=sentence.strip()

            if len(sentence_clean)>20:
                document_sentences.append({
                    'text':sentence_clean,
                    'source':f"PubMed: {paper['title'][:50]}...",
                    'page': f"PMID: {paper['pmid']}",
                    'link':paper['link']
                }) 

    return document_sentences




def download_pdf_from_url(url, save_folder="downloads_papers"):
    """
    Download a pdf with url
    parameters:
    -url:Direct link to pdf
    -save_folder: Where to save pdf
    
    Returns:
    -Path to download file or None if failed
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    try: 
        print(f" Downloading from :{url}")
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': url.rsplit('/', 1)[0] + '/',
        }
        response =requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        content_type=response.headers.get('content-type','')

        if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
            print(f" Warning :This might not be a PDF file")

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if not filename or not filename.endswith('.pdf'):
            # Try to get from Content-Disposition header
            content_disp = response.headers.get('content-disposition', '')
            if 'filename=' in content_disp:
                filename = content_disp.split('filename=')[1].strip('"')
            else:
                filename = f"downloaded_paper_{hash(url) % 10000}.pdf"


        filepath = os.path.join(save_folder,filename)

        with open(filepath,'wb') as f:
            f.write(response.content)

        print(f"✅ Downloaded: {filename}")
        return filepath
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') else None
        
        if status_code == 403:
            print(f"❌ 403 Forbidden: Website blocking automated access")
            print(f"   💡 Solution: Download manually from browser")
        elif status_code == 404:
            print(f"❌ 404 Not Found: File doesn't exist at this URL")
        else:
            print(f"❌ HTTP Error {status_code}: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None
 


def download_multiple_pdfs(urls):
    """download multiple PDFs from a list of URLs"""

    downloaded_files=[]
    print(f"Downloading {len(urls)} PDFs...\n")

    for i, url in enumerate(urls,1):
        print(f"[{i}/{len(urls)}]",end=" ")
        filepath = download_pdf_from_url(url)

        if filepath:
            downloaded_files.append(filepath)
        print()

    print(f"\n✅ Successfully downloaded {len(downloaded_files)}/{len(urls)} files")

    return downloaded_files




def generate_research_answer(question, retrieved_sentences, api_key):
    """
    Using this function to rephrase the answer from the content received from TF-IDF

    Parameters:
    -questions: users research question
    -retrieved_sentences: List of top sentences from TF-IDF,
    -api_key: groq API key

    Returns:
    an good phrase summary.
    """

    client=Groq(api_key= api_key)
    context_parts=[]
    for i, sentence in enumerate(retrieved_sentences, 1):
        context_parts.append(
            f"[Source {i}: {sentence['source']}, Page {sentence['page']}]\n"
            f"{sentence['text']}\n"
        )
    
    context ="\n".join(context_parts)

    prompt=f"""You are a medical research assistant. Based on the following research excerpts, provide a clear, evidence-based answer to the question.

QUESTION: {question}

RESEARCH EXCERPTS:
{context}

INSTRUCTONS:
1. Synthesize the information from all sources
2. Provide a comprehensive answer
3. Cite sources using [Sources 1], [Sources 2], etc.
4. If the excerpts don't fully answer the question, say so
5. Keep the answer focused and relevant

ANSWER:"""
    
    print("🤖 Generating AI answer...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    answer=response.choices[0].message.content

    return answer






def tfidf_search(question, document_sentences, top_n=10):                    #beter scoring system which tell how much an word is important with all document
    """
    Use TF-IDF to find most relevant sentences
    Parameters:
    - question: User's question (string)
    - document_sentences: List of sentence dictionaries
    - top_n: How many results to return
    
    Returns:
    - List of top N most relevant sentences with scores

    """
    texts= [doc['text'] for doc in document_sentences]

    print(f"🔍 Analyzing {len(texts)} sentences with TF-IDF...")

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1,2)
    )

    all_documents = [question] + texts

    tfidf_matrix = vectorizer.fit_transform(all_documents)

    question_vector= tfidf_matrix[0]
    
    sentence_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(question_vector, sentence_vectors)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]

    results=[]
    for idx in top_indices:
        if similarities[idx]>0:
            results.append({
                'text': document_sentences[idx]['text'],
                'score': round(similarities[idx],4),
                'source': document_sentences[idx]['source'],
                'page' : document_sentences[idx]['page']
            })
    return results



def load_all_pdfs_with_tracking(folder_path):                    #this function is build by which program is capable 
                                                                 #to read multiple pdf in a folder.
    document_sentences=[]

    if not os.path.exists(folder_path):              
        print(f"❌ Folder not found: {folder_path}")
        return None
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    if len(pdf_files)==0:
        print(f"❌ No pdfs found in {folder_path}")
        return None
    
    print(f"✅ Found {len(pdf_files)} PDF files\n")
    

    for filename in pdf_files:
        filepath= os.path.join(folder_path, filename)
        print(f"📄 Processing: {filename}")

        try:
            reader = PdfReader(filepath)


            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()

                if text:
                   
                   sentences = text.split('.')

                   for sentence in sentences:
                       sentence_clean = sentence.strip()

                       if len(sentence_clean)>20:
                        document_sentences.append({
                          'text': sentence_clean,
                          'source': filename,
                          'page': page_num
                    }) 
                        
            print(f"   ✓ Loaded {len(reader.pages)} pages")
    
        except Exception as e:
            print(f"   ✗ Error: {e}")

    print(f"\n✅ Total sentences : {len(document_sentences)}")
    return document_sentences


def extract_keywords_for_display(question):
    """Extract keywords just for showing the user (optional)"""
    words = question.split()
    stop_words = set(stopwords.words('english'))
    keywords = []
    
    for word in words:
        word_clean = word.lower().strip('.,!?')
        if word_clean not in stop_words and len(word_clean) > 2:
            keywords.append(word_clean)
    
    return keywords
                    

#  ======MAIN program=======

print("="*70)
print("MEDICAL RESEARCH ASSISTANT V1.4 - WITH SOURCE TRACKING")
print("="*70)

API_KEY = input("\n 🔑 Enter your Groq API key: ").strip()



print("\n📚 Choose input method:")
print("1. Local folder (PDFs already on your computer)")
print("2. Download from URLs (provide links to PDFs)")
print("3. Search PubMed (automatic)")

choice = input("Enter choice (1/2/3): ").strip()

question=input("\n💭Ask your research question: ")


keywords = extract_keywords_for_display(question)
print(f"🔍 Extracted keywords: {', '.join(keywords)}")
# STEP 1: GET DOCUMENTS
print("\n" + "=" * 70)
print("STEP 1: RETRIEVING RELEVANT CONTENT")
print("=" * 70)


if choice=="3":

    Entrez.email=input("📧 Enter your email (required by PubMed): ").strip()

    max_papers = int(input("📊 How many papers to search? (default 10): ").strip() or "10")

    papers=search_pubmed(question,max_results=max_papers)

    if not papers:
        print("❌ No papers found on PubMed")

    document_sentences= create_sentence_from_pubmed(papers)

    print(f"\n✅ Extracted {len(document_sentences)} sentences from {len(papers)} papers")

elif choice =="2":
    print("\n Enter PDF URLs (one per line, press enter twice when done):")
    urls=[]
    while True:
        url =input().strip()
        if not url:
            break
        urls.append(url)
    if not urls:
        print("❌ No URLs provided")
        exit()

    downloaded_files = download_multiple_pdfs(urls)

    if not downloaded_files:
        print("❌ No files downloaded successfully")
        exit()
    
    folder = "downloads_papers"
    document_sentences = load_all_pdfs_with_tracking(folder)

else:

      folder = input("\n 📂Enter folder path : ").strip('"').strip("'")
      document_sentences = load_all_pdfs_with_tracking(folder)


if not document_sentences:
    print("❌ Failed to load documents")
    exit()



# print("\n" + "=" * 70)
# print("STEP 1: RETRIEVING RELEVANT CONTENT")
# print("=" * 70)

#document_sentences=load_all_pdfs_with_tracking(folder)



print("\n🔎 Searching with TF-IDF algorithm...\n")
top_results= tfidf_search(question, document_sentences, top_n=5)

if not top_results:
    print("❌ No relevant content found")
    exit()

print("\n" + "=" * 70)
print("RETRIEVED CONTEXT:")
print("=" * 70)





for i, result in enumerate(top_results, 1):
       print(f"\n📄 Source {i}: {result['source']} (Page {result['page']})")
       print(f"   Relevance: {result['score']:.4f}")
       print(f"   Text: {result['text'][:100]}...")


print("\n" + "="*70)
print("Step 2: Generating AI-Powered Answer")
print("="*70)

answer = generate_research_answer(question, top_results, API_KEY)


print("\n" + "="*70)
print("FINAL ANSWER:")
print("="*70)
print(answer)

print("\n" + "=" * 70)
print("SOURCES USED:")
print("=" * 70)
for i, result in enumerate(top_results, 1):
    print(f"{i}. {result['source']} (Page {result['page']}) - Score: {result['score']:.4f}")

print("\n" + "=" * 70)