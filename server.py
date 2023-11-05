import transformers
from transformers import pipeline
from transformers import AutoTokenizer
import streamlit as st

def summarizer_text(text, len):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": len}
    pipe = pipeline("summarization", model="t5-small-sumarization",tokenizer=tokenizer)
    return pipe(text, **gen_kwargs)[0]["summary_text"]

st.title("Summarization Text")
st.write("The model can only output the maximum length of 512 words, and the maximum length of words can be adjusted by the generalization of the text itself.")
st.write("This site is intended to summarize texts in English prepared using the Transformation Hugging Face, Tensorflow and Numpy libraries.  You can find all the code on my github in the upper right corner.")

len = st.slider("Максимальная длина (max_length):", min_value=50, max_value=128)

input_text = st.text_area("Input text:", height=400)

if st.button("Summarization"):
    processed_text = summarizer_text(input_text, len)

    st.write("Output text:", processed_text)