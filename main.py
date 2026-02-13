import os                                #importing this lib by which code can get accees to pdfs in system
from groq import Groq
from PyPDF2 import PdfReader             #this library for reading pdf(python inbuilt)
import nltk                              #natural lang toolkit for call its toll stopword lib which will be predifined set for filtering
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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






def tfidf_search(question, document_sentences, top_n=3):                    #beter scoring system which tell how much an word is important with all document
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


question=input("\n💭Ask your research question: ")


keywords = extract_keywords_for_display(question)
print(f"🔍 Extracted keywords: {', '.join(keywords)}")

folder = input("\n 📂Enter folder path : ").strip('"').strip("'")

print("\n" + "=" * 70)
print("STEP 1: RETRIEVING RELEVANT CONTENT")
print("=" * 70)

document_sentences=load_all_pdfs_with_tracking(folder)

if not document_sentences:
    print("❌ Failed to load documents")
    exit()

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
    
