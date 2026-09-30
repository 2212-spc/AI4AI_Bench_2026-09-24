# Training corpora (unique tokens after dedup)

| domain | unique tokens | notes |
|---|---|---|
| web    | 3.0e+12 | filtered CommonCrawl-style web text; effectively inexhaustible at our scale |
| code   | 1.3e+10 | permissively licensed source code, deduplicated at file level |
| math   | 1.1e+10 | math web pages, textbooks and worked solutions |
| papers | 1.1e+10 | scientific papers (full text) |

Eval sets (held out, never trained on): `general` (broad web/knowledge text), `code` (code completion),
`math` (math problems with solutions).  The team's headline number is the composite
`0.121 * general + 0.256 * code + 0.623 * math`.

The target run: 2.5B non-embedding parameters, 500B training tokens, single run, standard recipe
(the proxies use the same recipe at smaller size).
