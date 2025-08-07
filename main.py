import streamlit as st
from app.config import Config
from app.utils.blog_scraper import BlogScraper
from app.utils.pdf_processor import PDFProcessor
from app.utils.text_summarizer import TextSummarizer

Config.ensure_directories()  # Make sure folders exist

st.set_page_config(page_title=Config.APP_TITLE)
st.title(Config.APP_TITLE)
st.write(Config.APP_DESCRIPTION)

tab_blog, tab_pdf = st.tabs(["📝 Blog Summarizer", "📄 PDF Summarizer"])
summarizer = TextSummarizer()

with tab_blog:
    st.subheader("Summarize a Blog Post")
    blog_url = st.text_input("Enter blog post URL")
    summarize_blog = st.button("Summarize Blog")
    if summarize_blog and blog_url:
        with st.spinner("Scraping and summarizing blog..."):
            try:
                blog = BlogScraper().extract_content(blog_url)
                summary = summarizer.summarize(blog['content'], method="ollama")["summary"]
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")

with tab_pdf:
    st.subheader("Summarize a PDF Document")
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if uploaded:
        summarize_pdf = st.button("Summarize PDF")
        if summarize_pdf:
            with st.spinner("Reading and summarizing PDF..."):
                try:
                    pdf_data = PDFProcessor().extract_text(uploaded, uploaded.name)
                    summary = summarizer.summarize(pdf_data['text'], method="ollama")["summary"]
                    st.success("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
