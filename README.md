
# webanno_tsv

A Library to parse TSV files as produced by the [webanno Software](https://github.com/webanno/webanno) and as described [in their documentation](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#sect_webannotsv).1

The following features are supported:

* WebAnnos UTF-16 indices for Text indices
* Webannos [escape sequences](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#_reserved_characters)
* Multiple span annotation layers with multiple fields
* Span annotations over multiple tokens and sentences
* Multiple Annotations per field (stacked annotations)
* Disambiguation IDs (here called `label_id`)

The following is not supported:

* Relations
* Chain annotations
* Sub-Token annotations (ignored on reading)


## Examples

To construct a Document with annotations you could do:

```py
from webanno_tsv import Document, Annotation
from dataclasses import replace

sentences = [
    ['First', 'sentence'],
    ['Second', 'sentence']
]
doc = Document.from_token_lists(sentences)

layer_defs = [('Layer1', ['Field1']), ('Layer2', ['Field2', 'Field3'])]
annotations = [
    Annotation(tokens=doc.tokens[1:2], layer='Layer1', field='Field1', label='ABC'),
    Annotation(tokens=doc.tokens[1:3], layer='Layer2', field='Field3', label='XYZ', label_id=1)
]
doc = replace(doc, annotations=annotations, layer_defs=layer_defs)
doc.tsv()
```

The call to `doc.tsv()` then returns a string:

```
#FORMAT=WebAnno TSV 3.3
#T_SP=Layer1|Field1
#T_SP=Layer2|Field2|Field3


#Text=First sentence
1-1	0-5	First	_	_	_
1-2	6-14	sentence	ABC	*[1]	XYZ[1]

#Text=Second sentence
2-1	15-21	Second	_	*[1]	XYZ[1]
2-2	22-30	sentence	_	_	_
```

Supposing that you have a file with the output above as input you could do:

```py
from webanno_tsv import webanno_tsv_read_file, Document

doc = webanno_tsv_read_file('/tmp/input.tsv')

for token in doc.tokens:
    if token.text == 'sentence':
        print(token.sentence_idx, token.idx)

# Prints:
# 1 2
# 2 2

for annotation in doc.match_annotations(layer='Layer2'):
    print(annotation.layer, annotation.field, annotation.label)

# Prints:
# Layer2 Field3 XYZ

for annotation in doc.match_annotations(sentence=doc.sentences[0]):
    print(annotation.layer, annotation.field, annotation.label)

# Prints:
# Layer1 Field1 ABC
# Layer2 Field3 XYZ

# Some lookup functions for convenience are on the Document instance
doc.token_sentence(token[0])
doc.sentence_tokens(doc.sentence[0])
doc.annotation_sentences(doc.annotations[0])
```

The classes in this library are read-only dataclasses ([dataclasses with `frozen=True`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass)).

This means that their fields are not settable. You can create new versions however with [`dataclasses.replace()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace).

```py
from dataclasses import replace

t1 = Token(sentence_idx=1, idx=0, start=0, end=3, text='Foo')
t2 = replace(t1, text='Bar')
```


## Development

Run the tests with:

```sh
python -m unittest test/*.py
```

PRs always welcome!
