"""Refactored script implementing two approaches over a Whisper audio transcription:

1. Traditional RAG (chunk + embed + retrieve + answer)
2. Alternative "prompt stuffing" (entire transcript in prompt)

Original procedural code split into functions for clarity and reuse.
"""
from typing import List
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


# ------------------ Transcription ------------------
def transcribe_audio(audio_path: str, model_name: str = "medium") -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


# ------------------ Chunking / Splitting ------------------
def split_transcript(transcription: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(transcription)


# ------------------ Vectorstore Build ------------------
def build_vectorstore(chunks: List[str], audio_label: str) -> FAISS:
    embeddings = OllamaEmbeddings()
    metadatas = [{"file": audio_label, "source": str(i)} for i in range(len(chunks))]
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)


# ------------------ Prompts ------------------
TRAD_PROMPT_TEMPLATE = """You answer questions about the contents of a transcribed audio file. 
Use only the provided audio file transcription as context to answer the question. 
Do not use any additional information.
If you don't know the answer, just say that you don't know. Do not use external knowledge. 
Use three sentences maximum and keep the answer concise. 
Make sure to cite references by referencing quotes of the provided context. Do not use any other knowledge.

Question: {question}
Context: {context}
Answer:"""

EVAL_TEMPLATE = """Rate the answer: " {answer} " to the question "{question}" given only the context: "{context}".
The Rating should be between 1 (lowest) and 10 (highest), followed by a one-sentence explanation.
Format:
x/10\nExplanation sentence."""

COMPARE_TEMPLATE = """Compare the following two answers given the question: {question}.
Answer "Trad": "{trad}"
Answer "Alt": "{alt}"
Say which is better and why in one concise sentence."""


# ------------------ Traditional RAG Query ------------------
def traditional_rag_query(docsearch: FAISS, question: str, llm_model: str = "llama2", k: int = 4) -> str:
    docs = docsearch.similarity_search(question, k=k)
    # Build context string
    context = "\n\n".join([d.page_content for d in docs])
    prompt = PromptTemplate.from_template(TRAD_PROMPT_TEMPLATE).format(question=question, context=context)
    llm = Ollama(model=llm_model, temperature=0)
    return llm(prompt), docs


# ------------------ Alternative Prompt Stuffing ------------------
def alternative_full_prompt(transcription: str, question: str, llm_model: str = "llama2") -> str:
    prompt = PromptTemplate.from_template(TRAD_PROMPT_TEMPLATE).format(question=question, context=transcription)
    llm = Ollama(model=llm_model, temperature=0)
    return llm(prompt)


# ------------------ Evaluation ------------------
def evaluate_answer(answer: str, question: str, context: str, llm_model: str = "llama2") -> str:
    eval_prompt = PromptTemplate.from_template(EVAL_TEMPLATE).format(answer=answer, question=question, context=context)
    llm = Ollama(model=llm_model, temperature=0)
    return llm(eval_prompt)


def compare_answers(trad_answer: str, alt_answer: str, question: str, llm_model: str = "llama2") -> str:
    prompt = PromptTemplate.from_template(COMPARE_TEMPLATE).format(trad=trad_answer, alt=alt_answer, question=question)
    llm = Ollama(model=llm_model, temperature=0)
    return llm(prompt)


# ------------------ Main Flow ------------------
def main():
    audio_file = "BryanThe_Ideal_Republic.ogg"
    question = "What is the idea of the republic?"
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    print("Transcription snippet:", transcription)

    print("Splitting transcript...")
    chunks = split_transcript(transcription)
    print(f"Created {len(chunks)} chunks")

    print("Building vectorstore...")
    docsearch = build_vectorstore(chunks, audio_file)

    print("Running Traditional RAG query...")
    trad_answer, docs = traditional_rag_query(docsearch, question)
    print("\n[Traditional Answer]\n", trad_answer)

    trad_context = "\n\n".join([d.page_content for d in docs])
    trad_eval = evaluate_answer(trad_answer, question, trad_context)
    print("\n[Traditional Evaluation]\n", trad_eval)

    print("Running Alternative full-transcript query...")
    alt_answer = alternative_full_prompt(transcription, question)
    print("\n[Alternative Answer]\n", alt_answer)

    alt_eval = evaluate_answer(alt_answer, question, transcription)
    print("\n[Alternative Evaluation]\n", alt_eval)

    comparison = compare_answers(trad_answer, alt_answer, question)
    print("\n[Comparison]\n", comparison)


if __name__ == "__main__":
    main()