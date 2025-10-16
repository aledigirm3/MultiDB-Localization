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


## LLM TABLE Extraction

### BIRDdev TAB extraction evaluation

- STRICT RECALL (for table extraction only): 0.9510682288077188
- STRICT RECALL: 0.8996088657105606
- TABLE REDUCTION avg: 0.4722909477800767
- WRONG DB accuracy: 0.40963855421686746

### SPIDERdev1 TAB extraction evaluation

- STRICT RECALL (for table extraction only): 1.0
- STRICT RECALL: 0.9100580270793037
- TABLE REDUCTION avg: 0.6807023092623523
- WRONG DB accuracy: 0.3548387096774194