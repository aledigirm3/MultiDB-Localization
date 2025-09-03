# NLQ-Tables

Enabling Natural Language Queries over Tabular Data with Large Language Models

## Similarity Database Extraction

### paraphrase-mpnet-base-v2

|       | T     | T + A |
| ----- | ----- | ----- |
| Top 1 | 0.888 | 0.901 |
| Top 2 | 0.963 | 0.964 |
| Top 3 | 0.980 | 0.980 |

### all-MiniLM-L12-v2

|       | T     | T + A |
| ----- | ----- | ----- |
| Top 1 | 0.903 | 0.902 |
| Top 2 | 0.960 | 0.954 |
| Top 3 | 0.983 | 0.970 |

### BAAI/bge-large-en-v1.5

BIRD dev
|       | T     | T + A |      
| ----- | ----- | ----- |               
| Top 1 | 0.908 | 0.946 |         
| Top 2 | 0.971 | 0.981 |
| Top 3 | 0.984 | 0.990 |


SPIDER dev 1.0
|       | T     | T + A |
| ----- | ----- | ----- |
| Top 1 | 0.854 | 0.910 |