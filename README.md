
# webanno-tsv-py

A Library to parse TSV files as produced by the [webanno Software](https://github.com/webanno/webanno) and as described [in their documentation](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#sect_webannotsv).1

The following IS supported:

* Token and sentence numbering using UTF-16 indices for Text indices
* Webannos [esoteric escape sequences](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#_reserved_characters)
* Layers with multiple annotations
* Span annotations over multiple Tokens and sentences
* Multiple Annotations per field (stacked annotations)
* Disambiguation IDs
* Correctly fills empty fields in Layers with other fields set (with e.g. `*` or `*[1]` )

The following IS NOT supported:

* Relations
* Chain annotations
* Sub-Token annotations

## Development

Run the tests with:

```bash
python -m unittest test/*.py
```
