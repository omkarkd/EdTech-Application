from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()
llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"],temprature = 0)

### Creating Embeddings
google_palm_embeddings = GooglePalmEmbeddings(google_api_key=os.environ["GOOGLE_API_KEY"])
vectordb_filepath = "faiss_index"
def create_db():
    loader = CSVLoader(file_path="faqs.csv",source_column="\ufeffprompt",encoding="UTF-8")
    data = loader.load()
    
    vectordb = FAISS.from_documents(documents = data, embedding = google_palm_embeddings)
    vectordb.save_local(vectordb_filepath)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_filepath,google_palm_embeddings)
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))