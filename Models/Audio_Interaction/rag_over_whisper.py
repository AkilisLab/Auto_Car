# Uses Whisper API from OpenAI to generate audio from text
import whisper 

# Load the model from whisper (i have chosen to use the medium model, as it is more accurate than the base model)
model = whisper.load_model("medium")

audio = "BryanThe_Ideal_Republic.ogg"

# Run the transcription and save it to "result"
# Note: the original audio file is available on Wikipedia at this link: https://commons.wikimedia.org/wiki/File:A_J_Cook_Speech_from_Lansbury%27s_Labour_Weekly.ogg 
result = model.transcribe(audio, fp16=False)

# Print the transcription
print(result["text"])

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.llms import Ollama

# Define the text to split
transcription = result["text"]

# Display the text
transcription

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
 
texts = splitter.split_text(transcription)

# Let's take a look at the texts, notice how there is a bit of overlap between them
print(texts)

# So that we know how many texts we have:
print(f"\nThe length of the texts is {len(texts)}")

# define the embeddings 

embeddings = OllamaEmbeddings()

# Create the vector store using the texts and embeddings and put it in a vector database

docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"file": audio,"source": str(i)} for i in range(len(texts))])

# Define the local LLM Model we will use
llm = Ollama(model='llama2', temperature=0)

#import chatprompttemplate 
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Create the RAG prompt
rag_prompt = ChatPromptTemplate(
    input_variables=['context', 'question'], 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'], 
                template="""You answer questions about the contents of a transcribed audio file. 
                Use only the provided audio file transcription as context to answer the question. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Use three sentences maximum and keep the answer concise. 
                Make sure to reference your sources with quotes of the provided context as citations.
                \nQuestion: {question} \nContext: {context} \nAnswer:"""
                )
        )
    ]
)


from langchain.chains.question_answering import load_qa_chain

# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt)

# Define a query
query = "What is the idea of the republic?"

# Find similar documents to the search query 
docs = docsearch.similarity_search(query)

# Display the docs determined to be semantically similar to the query
docs

# Basic Stuff Rag prompt (manually created)

trad_rag_template = """You answer questions about the contents of a transcribed audio file. 
                Use only the provided audio file transcription as context to answer the question. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Use three sentences maximum and keep the answer concise. 
                Make sure to cite references by referencing quotes of the provided context. Do not use any other knowledge.

                \nQuestion: {question} \nContext: {context} \nAnswer:"""

trad_prompt = PromptTemplate.from_template(trad_rag_template)

query = "What is the idea of the republic?"

trad_rag_prompt = trad_prompt.format(context=docs, question=query)

trad_answer = llm(trad_rag_prompt)

print(trad_answer)

from langchain.prompts import PromptTemplate

evaluation_template = """
    Rate the answer: " {answer_trad} " to the question "{question} " given only the context provided by an audio file: "{context}". 
    The Rating should be between 1 (lowest score) and 10 (highest score), and contain a max-1 sentence explanation of the rating.
    The rating should be based on the quality of the answer considering that the answer was ONLY based on the context, and nothing else.
    Format the answer as starting with the rating, followed by a newline, followed by the explanation.
    "x/10 
    The question asked about xxx, and the context provided xxxx, and the answer was .... .
    In order to receive a full score, the answer should be ...." """

prompt = PromptTemplate.from_template(evaluation_template)

my_prompt = prompt.format(answer_trad=trad_answer, context=docs, question=query)

print(llm(my_prompt))

### Method 2
from langchain.docstore.document import Document

transcript_doc=Document(
                page_content=transcription,
                metadata={"source": audio}
            )

transcript_doc

# Basic Stuff Rag prompt (manually created)

alt_rag_template = """You answer questions about the contents of a transcribed audio file. 
                Use only the provided audio file transcription as context to answer the question. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Use three sentences maximum and keep the answer concise. 
                Make sure to cite references by referencing quotes of the provided context. Do not use any other knowledge.
                
                \nQuestion: {question} \nContext: {context} \nAnswer:"""

alt_prompt = PromptTemplate.from_template(alt_rag_template)

query = "What is the idea of the republic?"

alt_rag_prompt = alt_prompt.format(context=transcript_doc, question=query)

answer_alt = llm(alt_rag_prompt)

print(answer_alt)

evaluation_template = """
    Rate the answer: " {answer_alt} " to the question "{question} " given only the context provided by an audio file: "{context}". 
    The Rating should be between 1 (lowest score) and 10 (highest score), and contain a max-1 sentence explanation of the rating.
    The rating should be based on the quality of the answer considering that the answer was ONLY based on the context, and nothing else.
    Format the answer as starting with the rating, followed by a newline, followed by the explanation.
    "x/10 
    The question asked about xxx, and the context provided xxxx, and the answer was .... .
    In order to receive a full score, the answer should be ...." """

prompt = PromptTemplate.from_template(evaluation_template)

my_prompt = prompt.format(answer_alt=answer_alt, context=transcript_doc, question=query)

print(llm(my_prompt))

compare_answers_evaluation_template = """
    Compare the following two answers: 
    Answer "Trad": "{answer_trad}" 
    Answer "Alt": "{answer_alt}"
    which were provide in response to the question: {question}.
    
    Start by saying which answer you think is better, and then explain why you think so.
    Explain the reasoning of your comparison in a max-1 sentence explanation.
    """

compare_prompt = PromptTemplate.from_template(compare_answers_evaluation_template)

my_prompt = compare_prompt.format(answer_trad=trad_answer, answer_alt=answer_alt, question=query)

print(llm(my_prompt))