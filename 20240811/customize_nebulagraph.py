from typing import Any, Dict, List

from llama_index.graph_stores import NebulaGraphStore
from llama_index.graph_stores.nebulagraph import QUOTE, escape_str

from table_info import Table


class CustomizeNebulaGraphStore(NebulaGraphStore):
    def _generate_column_id(self, table_name: str, column_name: str) -> str:
        return f"{table_name}::{column_name}"

    def _extract_fields(self, name: str) -> list[str]:
        text = self._tag_prop_names_map.get(name, "")
        if not text:
            return []

        content = text.split(".", maxsplit=1)[1]
        return content[1 : len(content) - 1].split(",")

    def _upsert_triplet_table(self, table: Table) -> None:
        entity_type = self._tags[0]
        fields = self._extract_fields(entity_type)
        if not fields or len(fields) != 2:
            raise ValueError(f"cannot extract entity fields for {entity_type}")

        table_name = escape_str(table.name)
        description = escape_str(table.description)

        dml = f'INSERT VERTEX {entity_type}({", ".join(fields)}) VALUES {QUOTE}{table_name}{QUOTE}:({QUOTE}{table_name}{QUOTE}, {QUOTE}{description}{QUOTE});'
        result = self.execute(dml)
        assert result and result.is_succeeded(), f"failed to upsert table: {table.name} {table.description}, dml: {dml}"

    def _upsert_triplet_columns(self, table: Table) -> list[str]:
        if not table.columns:
            print(f"no columns need upsert for table: {table.name}")
            return

        entity_type = self._tags[1]
        fields = self._extract_fields(entity_type)
        if not fields or len(fields) != 3:
            raise ValueError(f"cannot extract entity fields for {entity_type}")

        items, column_ids, table_name = [], [], escape_str(table.name)
        for column in table.columns:
            name = escape_str(column.name)
            name_cn = escape_str(column.name_cn)
            comment = (column.comment or "").strip()
            if comment:
                comment = escape_str(comment)

            column_id = self._generate_column_id(table_name, name)
            column_ids.append(column_id)
            items.append(
                f"{QUOTE}{column_id}{QUOTE}:({QUOTE}{name_cn}{QUOTE}, {QUOTE}{name}{QUOTE}, {QUOTE}{comment}{QUOTE})"
            )

        syntax = ",".join(items)
        dml = f'INSERT VERTEX {entity_type}({", ".join(fields)}) VALUES {syntax};'
        result = self.execute(dml)
        assert result and result.is_succeeded(), f"failed to upsert columns: {table.name}, dml: {dml}"

        return column_ids

    def upsert_triplet_edges(self, table_name: str, column_ids: list[str]) -> None:
        if not table_name or not column_ids:
            print(f"table name or column ids cannot be empty")
            return

        edge_type = self._edge_types[0]
        properities = self._edge_prop_map.get(edge_type, [])
        if not properities or len(properities) != 1:
            print(f"properities for edge {edge_type} cannot be empty")
            return

        items, table_name = [], escape_str(table_name)
        for column_id in column_ids:
            items.append(f"{QUOTE}{table_name}{QUOTE} -> {QUOTE}{column_id}{QUOTE}:({QUOTE}包含{QUOTE})")

        syntax = ",".join(items)
        dml = f'INSERT EDGE {edge_type}({", ".join(properities)}) VALUES {syntax};'
        result = self.execute(dml)
        assert result and result.is_succeeded(), f"failed to upsert edges: {table_name}, dml: {dml}"

    def upsert_triplet_batch(self, table: Table) -> None:
        if not table or not isinstance(table, Table) or len(self._tags) < 2 or not self._edge_types:
            return

        # upsert table
        self._upsert_triplet_table(table)

        # upsert columns
        columns = self._upsert_triplet_columns(table)

        # upsert edges
        self.upsert_triplet_edges(table.name, columns)
