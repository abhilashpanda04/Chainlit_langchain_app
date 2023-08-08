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
from chainlit import user_session

sys.path.append(os.path.abspath('.'))

cl.user_session.get("env")
# user_env.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = user_env.get("OPENAI_API_KEY")


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
embeddings=OpenAIEmbeddings()


messages=[
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt=ChatPromptTemplate.from_messages(messages)
chain_type_kwargs={"prompt":prompt}



def process_files(file:AskFileResponse):
    '''
    load the data from a text or a pdf file
    with the help of loader classes from langchain
    for creating document which will be used to create
    the document object
    '''
    import tempfile

    if file.type=="text/plain":
        Loader=TextLoader
    elif file.type=="application/pdf":
        Loader=PyPDFLoader

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader=Loader(tempfile.name)
        document=loader.load()
        docs= text_splitter.split_documents(document)
        for i , doc in enumerate(docs):
            doc.metadata["source"]=f"source-{i}"
        return docs

def get_db(file:AskFileResponse):
    '''
    get the vectorstore created.
    
    '''
    docs=process_files(file)
    
    #save data in user session
    cl.user_session.set("docs",docs)
    
    #create the db
    db=Chroma.from_documents(docs,embeddings)

    return db

def load_only_pdf(file):
    '''
    load only pdf with pypdf
    
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
    # await cl.Message(content="hello there welcome to chainlit").send()
    
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
    Executes ones the file is loaded
    '''

    # user_env = cl.user_session.get("env")
    # os.environ["OPENAI_API_KEY"] = user_env.get("OPENAI_API_KEY")

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
