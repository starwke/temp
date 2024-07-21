from configs.user_config import EMBEDDING_URL, OPENA__API_KEY, OPENAI_API_BASE
from llama_hub.wikipedia import WikipediaReader
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import KnowledgeGraphIndex, ServiceContext, load_index_from_storage
from llama_index.embeddings import TextEmbeddingsInference
from llama_index.graph_stores import NebulaGraphStore
from llama_index.llms import OpenAI, OpenAILike
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.query_engine import KnowledgeGraphQueryEngine
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from IPython.display import Markdown, display

space_name = "phillies_rag"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

llm = OpenAILike(temperature=0.1, model="qwen", api_base=OPENAI_API_BASE, api_key=OPENA__API_KEY)
embed_model = TextEmbeddingsInference(base_url=EMBEDDING_URL)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model=embed_model)

# 加载数据，建立索引

try:

    storage_context = StorageContext.from_defaults(persist_dir="./storage_graph", graph_store=graph_store)
    kg_index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        max_triplets_per_chunk=15,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        verbose=True,
    )
    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    # WikipediaReader = download_loader("WikipediaReader")
    # loader = WikipediaReader()
    # wiki_documents = loader.load_data(pages=['Philadelphia Phillies'], auto_suggest=False)
    content = ""
    with open("C:\\Users\\Jerry\\Desktop\\Philadelphia Phillies.txt", "r", encoding="utf8") as f:
        content = str(f.read())

    wiki_documents = [Document(text=content)]
    print(f"Loaded {len(wiki_documents)} documents")

    youtube_loader = YoutubeTranscriptReader()
    youtube_documents = youtube_loader.load_data(ytlinks=["https://www.youtube.com/watch?v=k-HTQ8T7oVw"])
    print(f"Loaded {len(youtube_documents)} YouTube documents")

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=wiki_documents + youtube_documents,
        storage_context=storage_context,
        max_triplets_per_chunk=15,
        service_context=service_context,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
    )
    kg_index.storage_context.persist(persist_dir="../storage_graph")


# 开始查询数据

# query_engine = KnowledgeGraphQueryEngine(
#     storage_context=storage_context,
#     service_context=service_context,
#     llm=llm,
#     verbose=True,
# )

hybrid_query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=3,
    explore_global_knowledge=True,
)

hybrid_query_engine = kg_index.as_query_engine()
response = hybrid_query_engine.query("Tell me about Bryce Harper.")
display(Markdown(f"<b>{response}</b>"))

query_engine = kg_index.as_query_engine()
response = query_engine.query("Tell me about Bryce Harper.")
display(Markdown(f"<b>{response}</b>"))

response = query_engine.query("Tell me about Bryce Harper.")
display(Markdown(f"<b>{response}</b>"))

