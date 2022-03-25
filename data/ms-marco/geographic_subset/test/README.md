The subset of the original DEV split is used for TESTING

Contains:
Msmarco.mordecai.queries.dev.pkl:
Dictionary[query_id -> geographic info]
Msmarco.mordecai.passages.dev.pkl:
Dictionary[passage_id -> geographic info]

Queries:
The original 542 queries identified as geographical by Luis, processed by mordecai, concatenating it with the capitalized version and a relevant passage. The mordecai run yielded at least one entity with coordinates for 292 passages.

Passages:
For the 292 passages, ranked the top-1000 passages with BM25. From those, 249811 are unique, and were processed by mordecai.
