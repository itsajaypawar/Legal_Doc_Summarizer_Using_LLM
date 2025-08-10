import streamlit as st

from PIL import Image
import pytesseract
import pdfplumber
import docx
from summary_t5 import summarize_text
from qa_t5 import answer_question

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode('utf-8')

st.title("Legal Document Summarizer ")

uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf", "docx", "txt"])


if uploaded_file:
    file_type = uploaded_file.type
    st.write(f"**File Type:** {file_type}")

    text = ""
    if "image" in file_type:
        image = Image.open(uploaded_file)
        text = extract_text_from_image(image)
    elif file_type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type!")

    if text:
        st.subheader("📜 Extracted Text:")
        
        edited_text = st.text_area("Text Output", text, height=300)
       
    else:
        st.warning("No text found in the uploaded document.")
    if st.button("Summarize"):
    # st.subheader("📄 Summary")
        with st.spinner("Generating summary..."):

            summary = summarize_text(edited_text)
        st.write(summary)


else:
    st.write("Please Upload the file ")

st.subheader("🔎 Ask a Question")
user_question = st.text_input("Enter your question ??")

if user_question and text:
    if st.button("Get Answer"):
        with st.spinner("Finding answer..."):
            answer = answer_question(user_question, text)
        st.success("Answer:")
        st.write(answer)