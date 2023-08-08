from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader,DataFrameLoader
from langchain.prompts.chat import (ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate)
from chainlit.types import AskFileResponse
import os 
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath('.'))


#load it from a .env file
# load_dotenv()
# cl.user_session.get("OPENAI_API_KEY")
# OPENAI_API_KEY=os.getenv("env")



__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

BASE_DIR="/workspaces/demo"
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

messages=[
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt=ChatPromptTemplate.from_messages(messages)
chain_type_kwargs={"prompt":prompt}



def process_files(file:AskFileResponse):
    '''
    This is a function takes a file object as a parameter.
    It checks the type of the file and assigns the appropriate loader class based on the file 
    type (TextLoader for plain text files and PyPDFLoader for PDF files). It then creates a temporary 
    file and writes the content of the input file to it.
    The loader is initialized with the path to the temporary file and the document is loaded. 
    The text_splitter is used to split the document into multiple documents.
    Each document's metadata is updated with a source identifier.
    Finally, the function returns the list of documents.
    '''
    import tempfile
    Loader = TextLoader if file.type == "text/plain" else PyPDFLoader
    
    with tempfile.NamedTemporaryFile(mode='wb') as tempfile:
        tempfile.write(file.content)
        loader=Loader(tempfile.name)
        document=loader.load()
        docs= text_splitter.split_documents(document)
        for i , doc in enumerate(docs):
            doc.metadata["source"]=f"source-{i}"
        return docs
    
    
    
def get_db(file:AskFileResponse):
    '''
    a function called get_db that takes a file parameter.
    It creates an instance of the OpenAIEmbeddings class, 
    processes the files using the process_files function, 
    saves the processed data in the user session, and then creates a database using the Chroma.
    from_documents method with the processed data and embeddings. 
    Finally, it returns the created database.
    '''
    docs=process_files(file)

    embeddings=OpenAIEmbeddings()
    
    #save data in user session
    cl.user_session.set("docs",docs)
    
    #create the db
    db=Chroma.from_documents(docs,embeddings)

    return db

def load_only_pdf(file):
    '''
    a function that takes a file object as input.
    It uses the pypdf library to extract text from a PDF file.
    The function reads the content of the file, creates a PdfReader object,
    and iterates over each page of the PDF to extract the text
    The extracted text is then split using a text_splitter 
    and returned along with corresponding metadata.
    '''
    pdf_stream=BytesIO(file.content)
    pdf=PyPDF2.PdfReader(pdf_stream)
    pdf_text=""
    for page in pdf.pages:
        pdf_text+=page.extract_text()
    
    
    texts=text_splitter.split_text(pdf_text)

    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]

    return texts,metadatas

@cl.on_chat_start
async def on_chat_start():
    startup_message="""This project implements a question answering bot that can answer questions
                based on PDF documents. The bot utilizes natural language processing and
                machine learning techniques to extract relevant information from PDF files
                and generate accurate answers to user queries."""
    
    await cl.Message(content=startup_message).send()

    user_env = cl.user_session.get("env")
    os.environ["OPENAI_API_KEY"] = user_env.get("OPENAI_API_KEY")
    
    files=None
    
    while files is None:
        #asking for a file
        files=await cl.AskFileMessage(
            content="upload your file to start chatting",
            accept=["text/plain","application/pdf"],
            max_size_mb=20,
            timeout=100,
        ).send()

        file=files[0]

        msg=cl.Message(content=f"processing '{file.name}'...")

        await msg.send()


        #making an async call based on chainlit
        docsearch=await cl.make_async(get_db)(file)

        #######if only pdf load######
        
        # docsearch= await cl.make_async(Chroma.from_texts)(texts,embeddings,metadatas=metadatas)
        
        #######if only pdf load######

        chain=RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=docsearch.as_retriever(max_token_limits=4097)
        )


        #######if only pdf load######

        # cl.user_session.set("metadatas",metadatas)
        # cl.user_session.set("texts",texts)
        
        #######if only pdf load######


        msg.content=f"processing '{file.name}' done , shoot your questions now"

        await msg.update()

        cl.user_session.set("chain",chain)



@cl.on_message
async def main(message:str):
    '''
    an asynchronous function called main that is executed when a message is received. 
    It retrieves a "chain" from a user session and creates an instance of AsyncLangchainCallbackHandler with certain settings.
    It then makes a call to the chain with the received message and a callback.
    The result is stored in res and the answer and sources are extracted from it. 
    The code then retrieves metadata and sources from the user session and processes them to add source elements to the message.
    Finally, depending on whether a final streamed answer is available or not,
    it either updates the stream or sends a message with the answer and source elements.
    '''

    chain=cl.user_session.get("chain")
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,answer_prefix_tokens=["FINAL","ANSWER"]
    )
    #stores data on the cache
    cb.answer_cached=True

    #make a call
    res=await chain.acall(message,callbacks=[cb])

    answer=res["answer"]
    sources=res["sources"].strip()
    source_elements=[]

    
    # metadatas=cl.user_session.get("metadatas")
    # texts=cl.user_session.get("texts")

    docs=cl.user_session.get("docs")
    metadatas=[doc.metadata for doc in docs]
    all_sources=[meta["source"] for meta in metadatas]
    
    
    if sources:
        found_sources=[]
    
    #add the sources to the messages
    for source in sources.split(","):
        source_name=source.strip().replace(".","")
        #Get the index of the source
        try:
            index=all_sources.index(source_name)
        except ValueError:
            continue
        

        # texts=texts[index]
        texts=docs[index].page_content
        found_sources.append(source_name)
        #create text elements refrenced in the message
        source_elements.append(cl.Text(content=texts,name=source_name))
        
        if found_sources:
            answer+=f"\nSources: {','.join(found_sources)}"
        else:
            answer+="\nNo Answers Found"

    if cb.has_streamed_final_answer:
        cb.has_streamed_final_answer=source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer,elements=source_elements).send()
