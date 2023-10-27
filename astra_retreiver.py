from langchain.schema import BaseRetriever, Document
from typing import List
from cassandra.cluster import  Session
from langchain.embeddings.base import Embeddings

class APIDocRetriever(BaseRetriever):
    session: Session 
    keyspace: str
    embedding: Embeddings        
    cql_st: str 

    class Config:
        arbitrary_types_allowed = True

    def init(self, embedding, cql_st, session, keyspace):
        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding        
        self.cql_st = cql_st 

    def get_relevant_documents(self, query: str) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        print(embedding_vector)
        q = self.session.prepare(self.cql_st)
        results = self.session.execute(q,(embedding_vector,) )
        if not results:
            return []       
        return [
            Document(page_content=row.text, metadata={"filename":row.filename})
            for row in results
        ]