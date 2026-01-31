import streamlit as st
from huggingface_hub import notebook_login

notebook_login()

from transformers import pipeline

# Initialize the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit app
st.title("Text File Summarizer")
st.write("Upload a text file (.txt) and I'll summarize it for you!")

uploaded_file = st.file_uploader(
    "Choose a text file", 
    type=["txt"],
    accept_multiple_files=False,
    help="Only .txt files are supported"
)

if uploaded_file is not None:
    # Verify it's a text file (additional check)
    if uploaded_file.type == "text/plain":
        st.success("Text file uploaded successfully!")
        
        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                try:
                    # Read the text file
                    text = uploaded_file.read().decode("utf-8")
                    
                    st.subheader("Original Text Preview")
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))
                    
                    # Generate summary using your function
                    summary = summarizer(
                        text, 
                        max_length=150, 
                        min_length=30, 
                        do_sample=False
                    )
                    
                    st.subheader("Key Points Summary")
                    summary_text = summary[0]['summary_text']
                    
                    # Split the summary into sentences and display as bullet points
                    sentences = [s.strip() for s in summary_text.split('. ') if s.strip()]
                    
                    for sentence in sentences:
                        # Ensure sentences end with proper punctuation
                        if not sentence.endswith('.'):
                            sentence += '.'
                        st.write(f"• {sentence}")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please upload only .txt files")