# Import required packages
import chromadb
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import nest_asyncio
nest_asyncio.apply()

class RAG:
    def setup_RAG(self, file_dir, template):
        # Embed model
        self.llm = Ollama(model="llama3", request_timeout=300.0)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Set up context from the intended document
        self.documents = SimpleDirectoryReader(input_files=[file_dir]).load_data()
        self.chroma_client = chromadb.EphemeralClient()
        self.chroma_collection = self.chroma_client.create_collection("ollama")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(self.documents, 
                                                storage_context=self.storage_context, 
                                                embed_model=self.embed_model,
                                                transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=10)])

        # Set up template incorporating the context and query variables
        # template = (
        #     "Imagine you are an astrophysicist and"
        #     "you answer questions about the astrophysicist's thesis."
        #     "Here is thesis in it's entirety::\n"
        #     "-----------------------------------------\n"
        #     "{context_str}\n"
        #     "-----------------------------------------\n"
        #     "Considering the above information, "
        #     "please respond to the following question:\n\n"
        #     "Question: {query_str}\n\n"
        #     "Answer succinctly and ensure your response is "
        #     "clear to someone without an astrophysics background."
        #     "The astrophysicist's name is Vijith."
        # )
        self.qa_template = PromptTemplate(template)
        self.query_engine = self.index.as_query_engine(text_qa_template=self.qa_template,
                                                                  similarity_top_k=3)

    def query_RAG(self, query):
        # Questions
        response = self.query_engine.query(query)
        print(response.response)
        return(response.response)