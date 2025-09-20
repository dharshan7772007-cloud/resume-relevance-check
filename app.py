import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from resume
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.read().decode("utf-8", errors="ignore")
    return text

# Function to extract top keywords
def top_keywords(text, top_k=20):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform([text])
    names = vec.get_feature_names_out()
    scores = X.toarray()[0]
    kws = sorted(list(zip(names, scores)), key=lambda x: -x[1])[:top_k]
    return [k for k, s in kws if s > 0]

# Function to check relevance using TF-IDF cosine similarity
def check_relevance(jd_text, resume_text):
    # TF-IDF similarity
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([jd_text, resume_text])
    sim_score = cosine_similarity(X[0:1], X[1:2])[0][0]

    # Keyword matching
    jd_kws = top_keywords(jd_text, top_k=20)
    matched = [k for k in jd_kws if k.lower() in resume_text.lower()]
    keyword_score = len(matched) / max(1, len(jd_kws))

    combined = 0.6 * sim_score + 0.4 * keyword_score

    return {
        "tfidf_score": round(sim_score, 3),
        "keyword_score": round(keyword_score, 3),
        "combined_score": round(combined, 3),
        "matched_keywords": matched,
        "top_job_keywords": jd_kws,
    }

# ---------------- Streamlit UI -----------------
st.title("📄 Automated Resume Relevance Checker (Lightweight)")
st.write("Upload a Job Description and Resume to check relevance!")

jd = st.text_area("Enter Job Description", height=200)
resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if st.button("Check Relevance"):
    if not jd or not resume_file:
        st.error("Please provide both Job Description and Resume.")
    else:
        resume_text = extract_text(resume_file)
        if not resume_text.strip():
            st.error("Could not extract text from resume. Try another file.")
        else:
            result = check_relevance(jd, resume_text)
            st.subheader("Results ✅")
            st.write(f"**Combined Score:** {result['combined_score']}")
            st.write(f"TF-IDF Score: {result['tfidf_score']}")
            st.write(f"Keyword Score: {result['keyword_score']}")
            st.write("**Matched Keywords:**", result['matched_keywords'])
            st.write("**Top JD Keywords:**", result['top_job_keywords'])
