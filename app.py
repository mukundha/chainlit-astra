import os 
import cassio
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from astra_retreiver import APIDocRetriever
from langchain.chains import RetrievalQA
from cassio.config import check_resolve_session, check_resolve_keyspace

token = os.environ['ASTRA_DB_APPLICATION_TOKEN']
database_id = os.environ['ASTRA_DB_DATABASE_ID']
table = os.environ['ASTRA_DB_TABLE']
cassio.init(token=token, database_id=database_id)
template = """An user is going to perform a questions, The retriever will help fetch relevant context to answer questions.
              Please try to leverage them in your answer as much as possible.
              Take into consideration that the user is always asking questions relevant to the given context.
              If you provide code or YAML snippets, please explicitly state that they are examples.
              Do not provide information that is not related to provided context."""

@cl.on_chat_start
def main():
    keyspace=check_resolve_keyspace(None),
    llm = ChatOpenAI(temperature=0.4, streaming=True)
    embedding = OpenAIEmbeddings()
    retriever = APIDocRetriever(
            embedding=embedding, 
            #Replace with your retriever Query
            cql_st= f"""SELECT * FROM {keyspace}.{table} 
                        ORDER BY <REPLACE_WITH_VECTOR_COLUMN> 
                        ANN of ? LIMIT 5""",
            session=check_resolve_session(None),
            keyspace=keyspace
            )   
    agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):        
    agent = cl.user_session.get("agent")
    res = await cl.make_async(agent.run)(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()] )     
    await cl.Message(content=res).send()
