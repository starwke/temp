from typing import List
from llama_index.text_splitter.types import TextSplitter
class LineTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []

        return re.split(r"\n|\\n", text, flags=re.I)

    @classmethod
    def class_name(cls) -> str:
        return "LineTextSplitter"





from llama_index.node_parser import SimpleNodeParser
from llama_index.callbacks.base import CallbackManager

callback_manager = CallbackManager([])
node_parser = SimpleNodeParser.from_defaults(text_splitter=LineTextSplitter(), callback_manager=callback_manager)



from llama_index import (
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index import download_loader
from llama_index.llms import OpenAILike,OpenAI

# define LLM
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo",)

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, node_parser=node_parser, callback_manager=callback_manager)
