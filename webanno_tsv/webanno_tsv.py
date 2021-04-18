import csv
import itertools
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NO_LABEL_ID = -1

COMMENT_RE = re.compile('^#')
SPAN_LAYER_DEF_RE = re.compile(r'^#T_SP=([^|]+)\|(.*)$')
SENTENCE_RE = re.compile('^#Text=(.*)')
FIELD_EMPTY_RE = re.compile('^[_*]')
FIELD_WITH_ID_RE = re.compile(r'(.*)\[([0-9]*)]$')
SUB_TOKEN_RE = re.compile(r'[0-9]+-[0-9]+\.[0-9]+')

HEADERS = ['#FORMAT=WebAnno TSV 3.3']

TOKEN_FIELDNAMES = ['sent_tok_idx', 'offsets', 'token']

# Strings that need to be escaped with a single backslash according to Webanno Appendix B
RESERVED_STRS = ['\\', '[', ']', '|', '_', '->', ';', '\t', '\n', '*']

# Mulitiline sentences are split on this character per Webanno Appendix B
MULTILINE_SPLIT_CHAR = '\f'


class WebannoTsvDialect(csv.Dialect):
    delimiter = '\t'
    quotechar = None  # disables escaping
    doublequote = False
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_NONE


@dataclass(frozen=True)
class Token:
    sentence_idx: int
    idx: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str


@dataclass(frozen=True, eq=False)  # Annotations are compared/hashed base on object identity
class Annotation:
    tokens: Sequence[Token]
    layer: str
    field: str
    label: str
    label_id: int = NO_LABEL_ID

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def text(self):
        return ' '.join([t.text for t in self.tokens])

    @property
    def token_texts(self):
        return [token.text for token in self.tokens]

    @property
    def has_label_id(self):
        return self.label_id != NO_LABEL_ID

    def should_merge(self, other: 'Annotation') -> bool:
        return self.has_label_id and other.has_label_id \
               and self.label_id == other.label_id \
               and self.label == other.label \
               and self.field == other.field \
               and self.layer == other.layer

    def merge(self, *other: 'Annotation') -> 'Annotation':
        return replace(self, tokens=token_sort(list(self.tokens) + [t for o in other for t in o.tokens]))


@dataclass(frozen=True, eq=False)
class Document:
    """
    Document binds together text features (Token, Sentence) with Annotations
    over the text. layer definitions is a tuple of layer and field names that
    defines the annotation.layer and annotation.field names when reading tsv.
    When writing, the layer definitions define which annotations are written
    and in what order.

    Example:
    Given a tsv file with lines like these:

        1-9	36-43	unhappy	JJ	abstract	negative

    You could invoke Document() with layer_defs=

        [ ('l1', ['POS']),
          ('l2', ['category', 'opinion']) ]

    allowing you to retrieve the annotation for 'abstract' within:

        doc.match_annotations(layer='l2', field='category')

    If you want to suppress output of the 'l2' layer when writing the
    document you could do:

        doc = dataclasses.replace(doc, layer_defs=[('l1', ['POS'])]
        doc.tsv()
    """
    layer_defs: Sequence[Tuple[str, Sequence[str]]]
    sentences: Sequence[Sentence]
    tokens: Sequence[Token]
    annotations: Sequence[Annotation]
    path: str = ''

    @property
    def text(self) -> str:
        return "\n".join([s.text for s in self.sentences])

    @classmethod
    def empty(cls, layer_defs=None):
        if layer_defs is None:
            layer_defs = []
        return cls(layer_defs, [], [], [])

    @classmethod
    def from_token_lists(cls, token_lists: Sequence[Sequence[str]], layer_defs: Sequence = None) -> 'Document':
        doc = Document.empty(layer_defs)
        for tlist in token_lists:
            doc = doc.with_added_token_strs(tlist)
        return doc

    def token_sentence(self, token: Token) -> Sentence:
        return next(s for s in self.sentences if s.idx == token.sentence_idx)

    def annotation_sentences(self, annotation: Annotation) -> List[Sentence]:
        return sorted({self.token_sentence(t) for t in annotation.tokens}, key=lambda s: s.idx)

    def sentence_tokens(self, sentence: Sentence) -> List[Token]:
        return [t for t in self.tokens if t.sentence_idx == sentence.idx]

    def match_annotations(self, sentence: Sentence = None, layer='', field='') -> Sequence[Annotation]:
        """
        Filter this document's annotations by the given criteria and return only those
        matching the given sentence, layer and field. Leave a parameter unfilled to
        include annotations with any value in that slot. For example:

            doc.match_annotations(layer='l1')

        returns annotations from layer 'l1' regardless of which sentence they are in or
        which field in that layer they have.
        """
        result = self.annotations
        if sentence:
            result = [a for a in result if sentence in self.annotation_sentences(a)]
        if layer:
            result = [a for a in result if a.layer == layer]
        if field:
            result = [a for a in result if a.field == field]
        return result

    def with_added_token_strs(self, token_strs: Sequence[str]) -> 'Document':
        """
        Build a new document that contains a sentence made up of tokens from the token
        texts. This increments sentence and token indices and calculates (utf-16)
        offsets for the tokens as per the TSV standard.

        :param token_strs: The token texts to add.
        :return: A new document with the token strings added.
        """
        sentence = Sentence(idx=len(self.sentences) + 1, text=' '.join(token_strs))

        start = self.tokens[-1].end + 1 if self.tokens else 0
        tokens = tokens_from_strs(token_strs, sent_idx=sentence.idx, token_start=start)

        return replace(self, sentences=[*self.sentences, sentence], tokens=[*self.tokens, *tokens])

    def tsv(self, linebreak='\n'):
        return webanno_tsv_write(self, linebreak)


def token_sort(tokens: Iterable[Token]) -> List[Token]:
    """
    Sort tokens by their sentence_idx first, then by the index in their sentence.
    """
    if not tokens:
        return []
    offset = max(t.idx for t in tokens) + 1
    return sorted(tokens, key=lambda t: (t.sentence_idx * offset) + t.idx)


def fix_annotation_ids(annotations: Iterable[Annotation]) -> List[Annotation]:
    """
    Setup label ids for the annotations to be consistent in the group.
    After this, there should be no duplicate label id and every multi-token
    annotation should have an id. Leaves present label_ids unchanged if possible.
    """
    with_ids = (a for a in annotations if a.label_id != NO_LABEL_ID)
    with_repeated_ids = {a for a in with_ids if a.label_id in [a2.label_id for a2 in with_ids if a2 != a]}
    without_ids = {a for a in annotations if len(a.tokens) > 1 and a.label_id == NO_LABEL_ID}
    both = without_ids.union(with_repeated_ids)
    if both:
        max_id = max((a.label_id for a in annotations), default=1)
        new_ids = itertools.count(max_id + 1)
        return [replace(a, label_id=next(new_ids)) if a in both else a for a in annotations]
    else:
        return list(annotations)


def utf_16_length(s: str) -> int:
    return int(len(s.encode('utf-16-le')) / 2)


def tokens_from_strs(token_strs: Sequence[str], sent_idx=1, token_start=0) -> [Token]:
    utf_16_lens = list(map(utf_16_length, token_strs))
    starts = [(sum(utf_16_lens[:i])) for i in range(len(utf_16_lens))]
    starts = [s + i for i, s in enumerate(starts)]  # offset the ' ' assumed between tokens
    starts = [s + token_start for s in starts]
    stops = [s + length for s, length in zip(starts, utf_16_lens)]
    return [Token(idx=i + 1, sentence_idx=sent_idx, start=s1, end=s2, text=t) for i, (s1, s2, t) in
            enumerate(zip(starts, stops, token_strs))]


def merge_into_annotations(annotations: Sequence[Annotation], annotation: Annotation) -> Sequence[Annotation]:
    candidate = next((a for a in annotations if a.should_merge(annotation)), None)
    if candidate:
        merged = candidate.merge(annotation)
        return [a if a != candidate else merged for a in annotations]
    else:
        return [*annotations, annotation]


def _annotation_type(layer_name, field_name):
    return '|'.join([layer_name, field_name])


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


def _read_token(row: Dict) -> Token:
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
    return Token(sentence_idx=sent_idx, idx=tok_idx, start=start, end=end, text=text)


def _read_annotation_field(row: Dict, layer: str, field: str) -> List[str]:
    col_name = _annotation_type(layer, field)
    return row[col_name].split('|') if row[col_name] else []


def _read_layer(token: Token, row: Dict, layer: str, fields: List[str]) -> List[Annotation]:
    fields_values = [(field, val) for field in fields for val in _read_annotation_field(row, layer, field)]
    fields_labels_ids = [(f, _read_label_and_id(val)) for f, val in fields_values]
    fields_labels_ids = [(f, label, lid) for (f, (label, lid)) in fields_labels_ids if label != '']

    return [Annotation(tokens=[token], layer=layer, field=field, label=label, label_id=lid) for
            field, label, lid in fields_labels_ids]


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

    def handle_label(s: str):
        return '' if FIELD_EMPTY_RE.match(s) else _unescape(s)

    match = FIELD_WITH_ID_RE.match(field)
    if match:
        return handle_label(match.group(1)), int(match.group(2))
    else:
        return handle_label(field), NO_LABEL_ID


def _filter_sentences(lines: List[str]) -> List[str]:
    """
    Filter lines beginning with 'Text=', if multiple such lines are
    following each other, concatenate them.
    """
    matches = [SENTENCE_RE.match(line) for line in lines]
    match_groups = [list(ms) for is_m, ms in itertools.groupby(matches, key=lambda m: m is not None) if is_m]
    text_groups = [[m.group(1) for m in group] for group in match_groups]
    return [MULTILINE_SPLIT_CHAR.join(group) for group in text_groups]


def _tsv_read_lines(lines: List[str], overriding_layer_names: List[Tuple[str, List[str]]] = None) -> Document:
    non_comments = [line for line in lines if not COMMENT_RE.match(line)]
    token_data = [line for line in non_comments if not SUB_TOKEN_RE.match(line)]
    sentence_strs = _filter_sentences(lines)
    sentences = [Sentence(idx=i + 1, text=text) for i, text in enumerate(sentence_strs)]

    if overriding_layer_names:
        layer_defs = overriding_layer_names
    else:
        layer_defs = _read_span_layer_names(lines)

    span_columns = [_annotation_type(layer, field) for layer, fields in layer_defs for field in fields]
    rows = csv.DictReader(token_data, dialect=WebannoTsvDialect, fieldnames=TOKEN_FIELDNAMES + span_columns)

    annotations = []
    tokens = []
    for row in rows:
        # consume the first three columns of each line
        token = _read_token(row)
        tokens.append(token)
        # Each column after the first three is (part of) a span annotation layer
        for layer, fields in layer_defs:
            for annotation in _read_layer(token, row, layer, fields):
                annotations = merge_into_annotations(annotations, annotation)
    return Document(layer_defs=layer_defs, sentences=sentences, tokens=tokens, annotations=annotations)


def webanno_tsv_read_string(tsv: str, overriding_layer_def: List[Tuple[str, List[str]]] = None) -> Document:
    """
    Read the string content of a tsv file and return a Document representation

    :param tsv: The tsv input to read.
    :param overriding_layer_def: If this is given, use these names
        instead of headers defined in the tsv string to name layers
        and fields. See Document for an example of layer_defs.
    :return: A Document instance of string input
    """
    return _tsv_read_lines(tsv.splitlines(), overriding_layer_def)


def webanno_tsv_read_file(path: str, overriding_layer_defs: List[Tuple[str, List[str]]] = None) -> Document:
    """
    Read the tsv file at path and return a Document representation.

    :param path: Path to read.
    :param overriding_layer_defs: If this is given, use these names
        instead of headers defined in the file to name layers
        and fields. See Document for an example of layer_defs.
    :return: A Document instance of the file at path.
    """
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    doc = _tsv_read_lines(lines, overriding_layer_defs)
    return replace(doc, path=path)


def _write_span_layer_header(layer_name: str, layer_fields: Sequence[str]) -> str:
    """
    Example:
        ('one', ['x', 'y', 'z']) => '#T_SP=one|x|y|z'
    """
    name = layer_name + '|' + '|'.join(layer_fields)
    return f'#T_SP={name}'


def _write_label(label: Optional[str]):
    return _escape(label) if label else '*'


def _write_label_id(lid: int):
    return '' if lid == NO_LABEL_ID else '[%d]' % lid


def _write_label_and_id(label: Optional[str], label_id: int) -> str:
    return _write_label(label) + _write_label_id(label_id)


def _write_annotation_field(annotations_in_layer: Iterable[Annotation], field: str) -> str:
    if not annotations_in_layer:
        return '_'

    with_field_val = {(a.label, a.label_id) for a in annotations_in_layer if a.field == field}

    all_ids = {a.label_id for a in annotations_in_layer if a.label_id != NO_LABEL_ID}
    ids_used = {label_id for _, label_id in with_field_val}
    without_field_val = {(None, label_id) for label_id in all_ids - ids_used}

    both = sorted(with_field_val.union(without_field_val), key=lambda t: t[1])
    labels = [_write_label_and_id(label, lid) for label, lid in both]
    if not labels:
        return '*'
    else:
        return '|'.join(labels)


def _write_sentence_header(text: str) -> List[str]:
    return ['', f'#Text={_escape(text)}']


def _write_token_fields(token: Token) -> Sequence[str]:
    return [
        f'{token.sentence_idx}-{token.idx}',
        f'{token.start}-{token.end}',
        _escape(token.text),
    ]


def _write_line(doc: Document, token: Token) -> str:
    token_fields = _write_token_fields(token)
    layer_fields = []
    for layer, fields in doc.layer_defs:
        annotations = [a for a in doc.annotations if a.layer == layer and token in a.tokens]
        layer_fields += [_write_annotation_field(annotations, field) for field in fields]
    return '\t'.join([*token_fields, *layer_fields])


def webanno_tsv_write(doc: Document, linebreak='\n') -> str:
    """
    Return a tsv string that represents the given Document.
    If there are repeated label_ids in the Docuemnt's Annotations, these
    will be corrected. If there are Annotations that are missing a label_id,
    it will be added.
    """
    lines = []
    lines += HEADERS
    for name, fields in doc.layer_defs:
        lines.append(_write_span_layer_header(name, fields))
    lines.append('')

    doc = replace(doc, annotations=fix_annotation_ids(doc.annotations))

    for sentence in doc.sentences:
        lines += _write_sentence_header(sentence.text)
        for token in doc.sentence_tokens(sentence):
            lines.append(_write_line(doc, token))

    return linebreak.join(lines)
