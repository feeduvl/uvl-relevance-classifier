from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict

Documents = List[Dict[Literal["text"], str]]


class Code(TypedDict):
    tokens: List[int]
    name: str
    tore: str
    index: int
    relationship_memberships: List[None]


class Token(TypedDict):
    index: int
    name: str
    lemma: str
    pos: str
    num_name_codes: int
    num_tore_codes: int


class Annotation(TypedDict):
    uploaded_at: datetime
    last_updated: datetime

    name: str
    dataset: str

    tores: List[str]
    show_recommendationtore: bool
    docs: List[Documents]
    tokens: List[Token]
    codes: List[Code]
    tore_relationships: List[Any]


class TruthElement(TypedDict):
    id: str
    value: str


class Dataset (TypedDict):
    uploaded_at: datetime
    name: str
    size: int
    documents: List[Documents]
    ground_truth: List[TruthElement]
