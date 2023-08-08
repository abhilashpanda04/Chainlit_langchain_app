from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import chainlit as cl
import os
import io

import sys

sys.path.append(os.path.abspath('.'))



llm=OpenAI()


def create_csv_agent(data: pd.DataFrame, llm):
    return create_pandas_dataframe_agent(llm,data,verbose=False)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="upload your csv or excel to start chatting").send()

    files=await cl.AskFileMessage(
            content="upload your file to start chatting",
            accept=["csv"],
            max_size_mb=20,
            timeout=100,
        ).send()

    file=files[0]

    msg=cl.Message(content=f"processing '{file.name}'...")
    
    #send user the message that the file is processing
    await msg.send()

    df=pd.read_csv(io.BytesIO(file.content),header=0)
    
    cl.user_session.set("data",df)

    msg.content=f"processing of {file.name} is done , shoot your question now!"

    #updating the same message
    await msg.update()

@cl.on_message
async def main(message: str):

    res = await cl.AskUserMessage(content="please provide you openai api key?", timeout=10).send()
    if res:
        await cl.Message(
            content=f"received your open ai api key",
        ).send()
    #get data

    # os.environ["OPENAI_API_KEY"] = res['content']

    df=cl.user_session.get("data")

    #create the agent instance
    agent=create_csv_agent(df,llm)

    #run agent
    response=agent.run(message)

    #get reponse from agent
    await cl.Message(
        content=response
    ).send()

