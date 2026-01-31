import streamlit as st
from transformers import pipeline

# Initialize the summarization pipeline with the lighter model
@st.cache_resource(show_spinner=False)
def load_summarizer():
    with st.spinner("Loading summarization model (one-time process)..."):
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Streamlit app
st.title("Text File Summarizer")
st.write("Upload a text file (.txt) for fast summarization")

uploaded_file = st.file_uploader(
    "Choose a text file", 
    type=["txt"],
    help="Only .txt files are supported"
)

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        st.success("✅ Text file uploaded successfully!")
        
        if st.button("Summarize", type="primary"):
            with st.spinner("Generating summary..."):
                try:
                    text = uploaded_file.read().decode("utf-8")
                    
                    # Display original text preview in expander
                    with st.expander("Original Text Preview"):
                        st.text(text[:500] + ("..." if len(text) > 500 else ""))
                    
                    # Generate summary with adjusted parameters for the smaller model
                    summary = summarizer(
                        text,
                        max_length=130,  # Slightly shorter for the smaller model
                        min_length=30,
                        do_sample=False
                    )
                    
                    st.subheader("🔑 Key Points Summary")
                    summary_text = summary[0]['summary_text']
                    
                    # Improved bullet point formatting
                    sentences = [s.strip() for s in summary_text.split('. ') if s.strip()]
                    
                    bullet_points = ""
                    for sentence in sentences:
                        if not sentence.endswith('.'):
                            sentence += '.'
                        bullet_points += f"• {sentence}\n\n"
                    
                    st.markdown(bullet_points)
                    
                    # Add download button for the summary
                    st.download_button(
                        label="Download Summary",
                        data=bullet_points,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.error("⚠️ Please upload only .txt files")