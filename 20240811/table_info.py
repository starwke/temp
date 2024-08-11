import os
from dataclasses import dataclass, field

import pandas as pd

import utils

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "table_schema")


@dataclass
class Column(object):
    # 列名
    name: str

    # 列名对应的中文名
    name_cn: str

    # 描述信息
    comment: str = None


@dataclass
class Table(object):
    # 表名
    name: str

    # 描述信息
    description: str

    # 字段
    columns: list[Column] = field(default_factory=list)


def get_delimiter(delimiter: str = None) -> str:
    return "," if not delimiter or not isinstance(delimiter, str) else delimiter


def read_file_headers(filepath: str, delimiter: str = ",") -> list[str]:
    filepath = utils.trim(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise ValueError(f"file {filepath} not exists")

    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
        return utils.trim(lines[0]).split(get_delimiter(delimiter)) if lines else []


def get_column_filepath(table_name: str) -> str:
    table_name = utils.trim(table_name)
    if not table_name:
        raise ValueError(f"table name cannot be empty")

    return os.path.join(DATA_DIR, "columns", f"{table_name}.csv")


def parse_columns(filepath: str, delimiter: str = ",") -> list[Column]:
    filepath = utils.trim(filepath)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        print(f"not found file: {filepath}")
        return []

    columns = list()
    df = pd.read_csv(filepath, sep=get_delimiter(delimiter))

    for _, row in df.iterrows():
        name = utils.trim(row["column_name"])
        name_cn = utils.trim(row["column_name_cn"])

        if not name:
            raise ValueError(f"column name is empty in file: {filepath}")

        columns.append(Column(name=name, name_cn=name_cn))

    return columns


def parse_table(content: str, headers: list[str] = None, delimiter: str = ",") -> Table:
    text = utils.trim(content)
    if not text or text.startswith("#") or text.startswith(";"):
        print(f"skip parse text=[{text}] due to it's a comment")
        return None

    words = text.split(get_delimiter(delimiter))
    if not headers or not isinstance(headers, list):
        record = {"table_name": words[0]}
        if len(words) > 1:
            record["table_comment"] = words[1]
    else:
        if len(headers) != len(words):
            raise ValueError(f"headers: {headers} missing match with content: {content}")

        record = dict(zip(headers, words))

    name = utils.trim(record.get("table_name", ""))
    if not name:
        raise ValueError(f"table name cannot be empty, content: {content}")

    description = utils.trim(record.get("table_comment", ""))
    filepath = get_column_filepath(table_name=name)
    columns = parse_columns(filepath=filepath, delimiter=delimiter)

    return Table(name=name, description=description, columns=columns)
