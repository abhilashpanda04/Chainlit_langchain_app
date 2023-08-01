from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate)
import os 
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import sys
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

BASE_DIR="/workspaces/demo"
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


system_template=system_template = """Use the following pieces of context to answer the users question.
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


messages=[
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt=ChatPromptTemplate.from_messages(messages)
chain_type_kwargs={"prompt":prompt}


@cl.on_chat_start
async def on_chat_start():
    # await cl.Message(content="hello there welcome to chainlit").send()
    
    files=None
    
    while files is None:
        files=await cl.AskFileMessage(
            content="upload your file to start chatting",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=100,
        ).send()

        file=files[0]

        msg=cl.Message(content=f"processing '{file.name}'...")

        await msg.send()

        pdf_stream=BytesIO(file.content)
        pdf=PyPDF2.PdfReader(pdf_stream)
        pdf_text=""
        for page in pdf.pages:
            pdf_text+=page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts=text_splitter.split_text(pdf_text)

        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]

        embeddings=OpenAIEmbeddings()
        docsearch= await cl.make_async(Chroma.from_texts)(texts,embeddings,metadatas=metadatas)

        chain=RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=docsearch.as_retriever()
        )

        cl.user_session.set("metadatas",metadatas)
        cl.user_session.set("texts",texts)

        msg.content=f"processing '{file.name}' done , shoot your questions now"

        await msg.update()

        cl.user_session.set("chain",chain)

@cl.on_message
async def main(message:str):
    chain=cl.user_session.get("chain")
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.answer_cached=True
    res=await chain.acall(message,callbacks=[cb])

    answer=res["answer"]
    sources=res["sources"].strip()
    source_elements=[]


    metadatas=cl.user_session.get("metadatas")
    all_sources=[m["source"] for m in metadatas]
    texts=cl.user_session.get("texts")
    
    if sources:
        found_sources=[]
    for source in sources.split(","):
        source_name=source.strip().replace(".","")

        try:
            index=all_sources.index(source_name)
        except ValueError:
            continue

        texts=texts[index]
        found_sources.append(source_name)
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


# if __name__=="__main__":
#     main()