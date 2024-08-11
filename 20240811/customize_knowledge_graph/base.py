from typing import Any, Callable, List, Sequence, Tuple

from llama_index import (
    BasePromptTemplate,
    KnowledgeGraphIndex,
    ServiceContext,
    StorageContext,
)
from llama_index.data_structs.data_structs import KG
from llama_index.schema import BaseNode, MetadataMode
from llama_index.utils import get_tqdm_iterable

from customize_nebulagraph import CustomizeNebulaGraphStore
from table_info import Table, parse_table


class CustomizeKnowledgeGraphIndex(KnowledgeGraphIndex):
    def __init__(
        self,
        nodes: Sequence[BaseNode] | None = None,
        index_struct: KG | None = None,
        service_context: ServiceContext | None = None,
        storage_context: StorageContext | None = None,
        kg_triple_extract_template: BasePromptTemplate | None = None,
        max_triplets_per_chunk: int = 10,
        include_embeddings: bool = False,
        show_progress: bool = False,
        max_object_length: int = 128,
        kg_triplet_extract_fn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            nodes,
            index_struct,
            service_context,
            storage_context,
            kg_triple_extract_template,
            max_triplets_per_chunk,
            include_embeddings,
            show_progress,
            max_object_length,
            kg_triplet_extract_fn,
            **kwargs,
        )

    def _extract_triplets(self, text: str) -> Table:
        return parse_table(content=text)

    def upsert_triplet(self, table: Table) -> None:
        if not table or not isinstance(table, Table):
            print(f"skip due to cannot parse table")
            return

        if not isinstance(self._graph_store, CustomizeNebulaGraphStore):
            raise ValueError(f"upsert triplet failed, only class CustomizeNebulaGraphStore support this function")

        return self._graph_store.upsert_triplet_batch(table)

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> KG:
        index_struct = self.index_struct_cls()
        nodes_with_progress = get_tqdm_iterable(nodes, self._show_progress, "Processing nodes")
        for n in nodes_with_progress:
            table = self._extract_triplets(n.get_content(metadata_mode=MetadataMode.LLM))
            if not table:
                continue

            self.upsert_triplet(table)

            for column in table.columns:
                index_struct.add_node([table.name, column.name], n)

            if self.include_embeddings:
                triplet_texts = [str((table.description, column.name_cn)) for column in table.columns]

                embed_model = self._service_context.embed_model
                embed_outputs = embed_model.get_text_embedding_batch(triplet_texts, show_progress=self._show_progress)
                for rel_text, rel_embed in zip(triplet_texts, embed_outputs):
                    index_struct.add_to_embedding_dict(rel_text, rel_embed)

        return index_struct
