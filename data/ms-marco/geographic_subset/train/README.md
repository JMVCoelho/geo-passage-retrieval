
The pickles contain ALL train queries.

The tsv contain all but the ones sampled for validation.
TSV schema:

qID pID bm25_pos    d(qID, pID)

Contents:

Msmarco.mordecai.train.queries:
Dictionary[query_id -> geographic information]

Msmarco.mordecai.train.passages:
Dictionary[passages_id -> geographic information]

Queries: 
Contains a subset (33k) of the original training passages, which were originally identified as geographic by Luis. Those 33k were processed by mordecai again, but concatenated with a capitalized version and its relevant passage for context. Only toponyms within the query were returned as geo information for the query.

For this second processing, it identified geographic entities WITH coordinates in 27104 queries. From those 27104, 16833 have at least one relevant passage.

Passages:
For the 16833 queries where entities and coordinates were extracted and have at least one relevant passage, extracted the top-1000 passages using BM25. 3068614 unique passages were identified. Ran mordecai for those passages. 

