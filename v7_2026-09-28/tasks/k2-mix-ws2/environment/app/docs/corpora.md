# Training corpora (unique tokens after dedup)

| domain | unique tokens | notes |
|---|---|---|
| web    | 3.0e+12 | filtered CommonCrawl-style web text; effectively inexhaustible at our scale |
| code   | 3.0e+10 | permissively licensed source code, deduplicated at file level |
| math   | 4.0e+09 | math web pages, textbooks and worked solutions |
| papers | 2.0e+10 | scientific papers (full text) |

Eval sets (held out, never trained on): `general` (broad web/knowledge text), `code` (code completion),
`math` (math problems with solutions).  The team's headline number is the composite
`0.34 * general + 0.33 * code + 0.33 * math`.

The target run: 2.5B non-embedding parameters, 500B training tokens, single run, standard recipe
(the proxies use the same recipe at smaller size).
