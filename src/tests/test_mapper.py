import json

import pytest  # noqa: F401

from main.structure.DataModels import Annotation, Token
from main.tooling.AnnotationMapper import AnnotationMapper

with open("src/tests/WordBasedAnno.json", 'r') as file:
    wordBasedAnnotation = json.load(file)

with open("src/tests/SentenceBasedAnno.json", 'r') as file:
    sentenceBasedAnnotation = json.load(file)

annotationMapper = AnnotationMapper(wordBasedAnnotation, sentenceBasedAnnotation)
wordBasedAnnotation = annotationMapper.mapRelevantSentences2WordBasedAnnotation()


def test_AnnotationMapperTokens() -> None:

    # test tokens in wordBasedAnnotation
    assert len(wordBasedAnnotation["tokens"]) == 135


def test_AnnotationMapperDocs() -> None:

    # test docs in wordBasedAnnotation
    assert wordBasedAnnotation["docs"][0]["begin_index"] == 0
    assert wordBasedAnnotation["docs"][0]["end_index"] == 58
    assert wordBasedAnnotation["docs"][1]["begin_index"] == 58
    assert wordBasedAnnotation["docs"][1]["end_index"] == 109
    assert wordBasedAnnotation["docs"][2]["begin_index"] == 109
    assert wordBasedAnnotation["docs"][2]["end_index"] == 135
    assert len(wordBasedAnnotation["docs"]) == 3
    assert wordBasedAnnotation["size"] == 3


def test_ds_mini() -> None:
    # Dataset ds_mini.cvs (Sentence *n=Non-Informative, *r=Informative)
    # 11n. |Document1
    # 21n. 22r. |Document2
    # 31r. 32n. |Document3
    # 41r. 42n. 43r. |Document4

    with open("src/tests/ds_mini_anno_word.json", 'r') as file:
        wordBasedAnnotation = json.load(file)

    with open("src/tests/ds_mini_anno_sentence.json", 'r') as file:
        sentenceBasedAnnotation = json.load(file)

    annotationMapper = AnnotationMapper(wordBasedAnnotation, sentenceBasedAnnotation)
    resultwordBasedAnnotation = annotationMapper.mapRelevantSentences2WordBasedAnnotation()

    # Not change the originals
    assert len(wordBasedAnnotation["docs"]) == 4
    assert len(sentenceBasedAnnotation["docs"]) == 4

    # Check Documents
    assert len(resultwordBasedAnnotation["docs"]) == 3  # Document1 contains only one sentense non-informative
    assert resultwordBasedAnnotation["docs"][0]["name"] == "Document2"
    assert resultwordBasedAnnotation["docs"][1]["name"] == "Document3"
    assert resultwordBasedAnnotation["docs"][2]["name"] == "Document4"

    # Check Tokens
    assert not findTokenByName(resultwordBasedAnnotation, "11n")
    assert not findTokenByName(resultwordBasedAnnotation, "21n")
    assert findTokenByName(resultwordBasedAnnotation, "22r")
    assert findTokenByName(resultwordBasedAnnotation, "31r")
    assert not findTokenByName(resultwordBasedAnnotation, "32n")
    assert findTokenByName(resultwordBasedAnnotation, "41r")
    assert not findTokenByName(resultwordBasedAnnotation, "42n")
    assert findTokenByName(resultwordBasedAnnotation, "43r")

    # Check Codes (every token has the same torecode  )
    token = findTokenByName(resultwordBasedAnnotation, "22r")
    idx = getCodeIdx(resultwordBasedAnnotation, token)
    assert resultwordBasedAnnotation['codes'][idx]['tore'] == "22r"

    token = findTokenByName(resultwordBasedAnnotation, "31r")
    idx = getCodeIdx(resultwordBasedAnnotation, token)
    assert resultwordBasedAnnotation['codes'][idx]['tore'] == "31r"

    token = findTokenByName(resultwordBasedAnnotation, "41r")
    idx = getCodeIdx(resultwordBasedAnnotation, token)
    assert resultwordBasedAnnotation['codes'][idx]['tore'] == "41r"

    token = findTokenByName(resultwordBasedAnnotation, "43r")
    idx = getCodeIdx(resultwordBasedAnnotation, token)
    assert resultwordBasedAnnotation['codes'][idx]['tore'] == "43r"

    # making first sentence informative
    idx = getCodeIdx(sentenceBasedAnnotation, sentenceBasedAnnotation["tokens"][0])
    sentenceBasedAnnotation["codes"][idx]["tore"] = "Informative"
    annotationMapper = AnnotationMapper(wordBasedAnnotation, sentenceBasedAnnotation)
    resultwordBasedAnnotation = annotationMapper.mapRelevantSentences2WordBasedAnnotation()
    assert len(resultwordBasedAnnotation["docs"]) == 4
    assert resultwordBasedAnnotation["docs"][0]["name"] == "Document1"
    assert findTokenByName(resultwordBasedAnnotation, "11n")


def getCodeIdx(annotation: Annotation, token: Token) -> int:
    for codeIndex, code in enumerate(annotation["codes"]):
        if token["index"] in code["tokens"]:
            codeIdx = codeIndex
            break
    return codeIdx


def findTokenByName(annotation: Annotation, name: str) -> Token:
    for token in annotation["tokens"]:
        if token["name"] == name:
            return token
