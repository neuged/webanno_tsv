import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

NO_LABEL_ID = -1

COMMENT_RE = re.compile('^#')
SPAN_LAYER_DEF_RE = re.compile(r'^#T_SP=([^|]+)\|(.*)$')
SENTENCE_RE = re.compile('^#Text=(.*)')
FIELD_EMPTY_RE = re.compile('^[_*]')
FIELD_WITH_ID_RE = re.compile(r'(.*)\[([0-9]*)]$')
SUB_TOKEN_RE = re.compile(r'[0-9]+-[0-9]+\.[0-9]+')

HEADERS = ['#FORMAT=WebAnno TSV 3.1']

TOKEN_FIELDNAMES = ['sent_tok_idx', 'offsets', 'token']

# Strings that need to be escaped with a single backslash according to Webanno Appendix B
RESERVED_STRS = ['\\', '[', ']', '|', '_', '->', ';', '\t', '\n', '*']

# Mulitiline sentences are split on this character per Webanno Appendix B
MULTILINE_SPLIT_CHAR = '\f'

logger = logging.getLogger(__file__)


class WebannoTsvDialect(csv.Dialect):
    delimiter = '\t'
    quotechar = None  # disables escaping
    doublequote = False
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_NONE


@dataclass
class Token:
    sentence: 'Sentence'
    idx: int
    start: int
    end: int
    text: str

    @property
    def doc(self) -> 'Document':
        return self.sentence.doc

    @property
    def annotations(self) -> List['Annotation']:
        return [a for a in self.doc.annotations if self in a.tokens]


def _unsafe_token_sort(tokens: Iterable[Token]) -> Iterable[Token]:
    # this is unsafe in that it doesn't check the tokens' document is the same.
    return sorted(tokens, key=lambda t: (t.sentence.idx * len(t.doc.tokens) + t.idx))


def token_sort(tokens: Iterable[Token]) -> Iterable[Token]:
    if tokens:
        if not len(set(t.doc for t in tokens)) == 1:
            raise ValueError('Cannot sort tokens in different documents')
        return _unsafe_token_sort(tokens)
    else:
        return tokens


def token_is_at_begin_of_sentence(token: Token) -> bool:
    return token.idx == 1


def token_is_at_end_of_sentence(token: Token) -> bool:
    return token.idx == max(t.idx for t in token.sentence.tokens)


def token_is_followed_by(token: Token, other: Token) -> bool:
    return ((token.sentence == other.sentence and token.idx == other.idx - 1)
            or (sentence_is_followed_by(token.sentence, other.sentence)
                and token_is_at_begin_of_sentence(other)
                and token_is_at_end_of_sentence(token)))


class Annotation:

    def __init__(self, tokens: List[Token], layer_name: str, field_name: str, label: str, label_id: int = NO_LABEL_ID):
        self.tokens = tokens
        self.layer_name = layer_name
        self.field_name = field_name
        self.label = label
        self.label_id = label_id

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def sentences(self) -> List['Sentence']:
        return sorted(list(set(t.sentence for t in self.tokens)), key=lambda s: s.idx)

    @property
    def text(self):
        return ' '.join([t.text for t in self.tokens])

    @property
    def doc(self):
        return self.tokens[0].doc

    @property
    def token_texts(self):
        return [token.text for token in self.tokens]


def annotation_merge_other(a: Annotation, other: Annotation):
    assert (a.layer_name == other.layer_name)
    assert (a.field_name == other.field_name)
    assert (a.label == other.label)
    assert (a.label_id == other.label_id)
    assert (token_is_followed_by(a.tokens[-1], other.tokens[0]))
    a.tokens = token_sort(a.tokens + other.tokens)


def _annotation_type(layer_name, field_name):
    return '|'.join([layer_name, field_name])


class Sentence:

    def __init__(self, doc: 'Document', idx: int, text: str):
        self.doc = doc
        self.idx = idx
        self.text = text
        self.tokens: List[Token] = []

    @property
    def token_texts(self) -> List[str]:
        return [token.text for token in self.tokens]


def sentence_add_token(s: Sentence, token: Token):
    s.tokens.append(token)


def sentence_annotations_with_type(s: Sentence, layer_name: str, field_name: str) -> List[Annotation]:
    return [a for a in document_annotations_with_type(s.doc, layer_name, field_name) if s in a.sentences]


def sentence_is_following(s: Sentence, other: Sentence) -> bool:
    return s.doc == other.doc and s.idx == (other.idx + 1)


def sentence_is_followed_by(s: Sentence, other: Sentence) -> bool:
    return sentence_is_following(other, s)


class Document:

    def __init__(self, layer_names: List[Tuple[str, List[str]]] = None):
        """
        Create a new document using the given keys as annotation layer names.

        Example:
        Given a tsv file with lines like these:

            1-9	36-43	unhappy	JJ	abstract	negative

        You could invoke Document() with

            [ ('l1', ['POS']),
              ('l2', ['category', 'opinion']) ]

        allowing you to retrieve e.g. an annotation for 'negative' within:

            doc.annotations_with_type('l2', 'opinion')

        Layer names are also used to output '#T_SP=' fields for

        :param layer_names: The (span) layers to use. See example above.
        """
        if not layer_names:
            layer_names = [('l1', ['annotation'])]
        self.layer_names = layer_names
        self.sentences: List[Sentence] = list()
        self.annotation_by_type: Dict[str, List[Annotation]] = defaultdict(list)
        self.next_token_idx = 0
        self.path = ''  # used to indicate the path this was read from

    @property
    def next_sentence_idx(self) -> int:
        return len(self.sentences) + 1

    @property
    def annotations(self) -> List[Annotation]:
        return [a for values in list(self.annotation_by_type.values()) for a in values]

    @property
    def text(self) -> str:
        return "\n".join([s.text for s in self.sentences])

    @property
    def tokens(self) -> List[Token]:
        return [t for s in self.sentences for t in s.tokens]


def document_sentence_with_idx(doc: Document, idx) -> Optional[Sentence]:
    try:
        return doc.sentences[idx - 1]
    except IndexError:
        return None


def document_fix_annotation_ids(doc: Document) -> None:
    """
    Setup label ids for annotations contained in this document to be consistent.
    After this, there should be no duplicate label id and every multi-token
    annotation should have an id. Leave present label_ids unchanged if possible.
    """
    max_id = max([a.label_id for a in doc.annotations], default=1)
    ids_seen = set()
    for a in doc.annotations:
        if len(a.tokens) > 1 and (a.label_id == NO_LABEL_ID or a.label_id in ids_seen):
            a.label_id = max_id + 1
            max_id = a.label_id
        ids_seen.add(a.label_id)


def document_add_tokens_as_sentence(doc: Document, tokens: List[str]) -> Sentence:
    """
    Builds a Webanno Sentence instance for the token texts, incrementing
    sentence and token indices and calculating (utf-16) offsets for the tokens
    as per the TSV standard. The sentence is added to the document's sentences.

    :param doc: The document to add tokens to.
    :param tokens: The tokenized version of param text.
    :return: A Sentence instance.
    """
    text = " ".join(tokens)
    sentence = Sentence(doc=doc, idx=doc.next_sentence_idx, text=text)

    for token_idx, token_text in enumerate(tokens, start=1):
        token_utf16_length = int(len(token_text.encode('utf-16-le')) / 2)
        end = doc.next_token_idx + token_utf16_length
        token = Token(sentence=sentence, idx=token_idx, start=doc.next_token_idx, end=end, text=token_text)
        sentence_add_token(sentence, token)
        doc.next_token_idx = end + 1
    document_add_sentence(doc, sentence)
    return sentence


def document_tsv(self) -> str:
    return webanno_tsv_write(self)


def document_add_sentence(self, sentence: Sentence):
    sentence.doc = self
    self.sentences.append(sentence)


def document_add_annotation(doc: Document, annotation: Annotation):
    merged = False
    type_name = _annotation_type(annotation.layer_name, annotation.field_name)
    # check if we should merge with an existing annotation
    if annotation.label_id != NO_LABEL_ID:
        same_type = document_annotations_with_type(doc, annotation.layer_name, annotation.field_name)
        same_id = [a for a in same_type if a.label_id == annotation.label_id]
        assert (len(same_id)) <= 1
        if len(same_id) > 0:
            annotation_merge_other(same_id[0], annotation)
            merged = True
    if not merged:
        assert (annotation.doc == doc)
        doc.annotation_by_type[type_name].append(annotation)


def document_remove_annotation(doc: Document, annotation: Annotation):
    type_name = _annotation_type(annotation.layer_name, annotation.field_name)
    annotations = doc.annotation_by_type[type_name]
    if annotations:
        doc.annotation_by_type[type_name].remove(annotation)
    else:
        raise ValueError


def document_annotations_with_type(doc: Document, layer_name: str, field_name: str) -> List[Annotation]:
    return doc.annotation_by_type[_annotation_type(layer_name, field_name)]


def _unescape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace('\\' + s, s)
    return text


def _escape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace(s, '\\' + s)
    return text


def _read_span_layer_names(lines: List[str]):
    matches = [SPAN_LAYER_DEF_RE.match(line) for line in lines]
    return [(m.group(1), m.group(2).split('|')) for m in matches if m]


def _read_token(doc: Document, row: Dict) -> Token:
    """
    Construct a Token from the row object using the sentence from doc.
    This converts the first three columns of the TSV, e.g.:
        "2-3    13-20    example"
    becomes:
        Token(Sentence(idx=2), idx=3, start=13, end=20, text='example')
    """

    def intsplit(s: str):
        return [int(s) for s in s.split('-')]

    sent_idx, tok_idx = intsplit(row['sent_tok_idx'])
    start, end = intsplit(row['offsets'])
    text = _unescape(row['token'])
    sentence = document_sentence_with_idx(doc, sent_idx)
    token = Token(sentence, tok_idx, start, end, text)
    sentence_add_token(sentence, token)
    return token


def _read_label_and_id(field: str) -> Tuple[str, int]:
    """
    Reads a Webanno TSV field value, returning a label and an id.
    Returns an empty label for placeholder values '_', '*'
    Examples:
        "OBJ[6]" -> ("OBJ", 6)
        "OBJ"    -> ("OBJ", -1)
        "_"      -> ("", None)
        "*[6]"   -> ("", 6)
    """
    match = FIELD_WITH_ID_RE.match(field)
    if match:
        label = match.group(1)
        label_id = int(match.group(2))
    else:
        label = field
        label_id = NO_LABEL_ID

    if FIELD_EMPTY_RE.match(label):
        label = ''

    return _unescape(label), label_id


def _filter_sentences(lines: List[str]) -> List[str]:
    """
    Filter lines beginning with 'Text=', if multiple such lines are
    following each other, concatenate them.
    """
    result = []
    matches = [SENTENCE_RE.match(line) for line in lines]
    for i, match in enumerate(matches):
        if match:
            if i > 0 and matches[i - 1]:
                result[-1] += MULTILINE_SPLIT_CHAR + match.group(1)
            else:
                result.append(match.group(1))
    return result


def _tsv_read_lines(lines: List[str], overriding_layer_names: List[Tuple[str, List[str]]] = None) -> Document:
    non_comments = [line for line in lines if not COMMENT_RE.match(line)]
    token_data = [line for line in non_comments if not SUB_TOKEN_RE.match(line)]
    sentences = _filter_sentences(lines)

    if overriding_layer_names:
        doc = Document(overriding_layer_names)
    else:
        doc = Document(_read_span_layer_names(lines))

    for i, text in enumerate(sentences):
        sentence = Sentence(doc, idx=i + 1, text=text)
        document_add_sentence(doc, sentence)

    span_columns = [layer + '~' + field for layer, fields in doc.layer_names for field in fields]
    rows = csv.DictReader(token_data, dialect=WebannoTsvDialect, fieldnames=TOKEN_FIELDNAMES + span_columns)
    for row in rows:
        # The first three columns in each line make up a Token
        token = _read_token(doc, row)
        # Each column after the first three is (part of) a span annotation layer
        for layer, fields in doc.layer_names:
            for field in fields:
                col_name = layer + '~' + field
                if row[col_name] is None:
                    continue
                values = row[col_name].split('|')
                for value in values:
                    label, label_id = _read_label_and_id(value)
                    if label != '':
                        a = Annotation(
                            tokens=[token],
                            label=label,
                            layer_name=layer,
                            field_name=field,
                            label_id=label_id,
                        )
                        document_add_annotation(doc, a)
    return doc


def webanno_tsv_read_string(tsv: str, overriding_layer_names: List[Tuple[str, List[str]]] = None) -> Document:
    """
    Read the string content of a tsv file and return a Document representation

    :param tsv: The tsv input to read.
    :param overriding_layer_names: If this is given, use these names
        instead of headers defined in the string to identify layers
        and fields
    :return: A Document instance of string input
    """
    return _tsv_read_lines(tsv.splitlines(), overriding_layer_names)


def webanno_tsv_read_file(path: str, overriding_layer_names: List[Tuple[str, List[str]]] = None) -> Document:
    """
    Read the tsv file at path and return a Document representation.

    :param path: Path to read.
    :param overriding_layer_names: If this is given, use these names
        instead of headers defined in the file to identify layers
        and fields
    :return: A Document instance of the file at path.
    """
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    doc = _tsv_read_lines(lines, overriding_layer_names)
    doc.path = path
    return doc


def _write_span_layer_header(layer_name: str, layer_fields: List[str]) -> str:
    """
    Example:
        ('one', ['x', 'y', 'z']) => '#T_SP=one|x|y|z'
    """
    name = layer_name + '|' + '|'.join(layer_fields)
    return f'#T_SP={name}'


def _annotations_for_token(token: Token, layer: str, field: str) -> List[Annotation]:
    doc = token.doc
    return [a for a in document_annotations_with_type(doc, layer, field) if token in a.tokens]


def _write_annotation_label(annotation: Annotation) -> str:
    label = _escape(annotation.label)
    if annotation.label_id == NO_LABEL_ID:
        return label
    else:
        return f'{label}[{annotation.label_id}]'


def _write_annotation_layer_fields(token: Token, layer_name: str, field_names: List[str]) -> List[str]:
    all_annotations = []
    for field in field_names:
        all_annotations += _annotations_for_token(token, layer_name, field)

    # If there are no annotations for this layer '-' is returned for each column
    if not all_annotations:
        return ['_'] * len(field_names)

    all_ids = {a.label_id for a in all_annotations}
    all_ids = all_ids - {NO_LABEL_ID}
    fields = []
    for field in field_names:
        annotations = [a for a in all_annotations if a.field_name == field]
        labels = []

        # first write annotations without label_ids
        for annotation in [a for a in annotations if a.label_id == NO_LABEL_ID]:
            labels.append(_write_annotation_label(annotation))

        # next we treat id'ed annotations, that need id'ed indicators in columns where no
        # annotation for the id is present
        for lid in sorted(all_ids):
            try:
                annotation = next(a for a in annotations if a.label_id == lid)
                labels.append(_write_annotation_label(annotation))
            except StopIteration:
                labels.append('*' if lid == NO_LABEL_ID else f'*[{lid}]')

        if not labels:
            labels.append('*')

        fields.append('|'.join(labels))

    return fields


def webanno_tsv_write(doc: Document, linebreak='\n') -> str:
    lines = []
    lines += HEADERS
    for name, fields in doc.layer_names:
        lines.append(_write_span_layer_header(name, fields))
    lines.append('')

    document_fix_annotation_ids(doc)

    for sentence in doc.sentences:
        lines.append('')
        lines.append(f'#Text={_escape(sentence.text)}')

        for token in sentence.tokens:
            line = [
                f'{sentence.idx}-{token.idx}',
                f'{token.start}-{token.end}',
                _escape(token.text),
            ]
            for layer, fields in doc.layer_names:
                line += _write_annotation_layer_fields(token, layer, fields)
            lines.append('\t'.join(line))

    return linebreak.join(lines)
