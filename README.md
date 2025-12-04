# Enhancing Text-to-SQL Pipelines via Semantic Database Retrieval and LLM-Based Schema Linking
This work introduces a pipeline composed of two stages preceding the text-to-SQL task: Database Retrieval and Schema Linking.  
The first stagerelies on sentence embeddings to select, given a natural language input, the semantically most relevant database (database retrieval).  
The second stage performs a progressive filtering of the tables and attributes required for the query (schema linking), leveraging controlled use of LLMs and purpose-built prompts.  
The proposed pipeline represents an effective step toward more scalable, interpretable and practically deployable text-to-SQL systems, particularly in
real-world scenarios involving complex and heterogeneous databases.  
  
**Benchmark used**: [*SPIDER dev 1.0*](https://yale-lily.github.io/spider), [*BIRD dev*](https://bird-bench.github.io/).

## Replicate the experiment
Install Python 3.11.11. Execute the following command.
```bash
git clone https://github.com/aledigirm3/NLQ-Tables.git
cd NLQ-Tables
pip install -r requirements.txt
```

Before executing the scripts, you must create a .env file in the root directory of the project. Use the structure provided in the .env.example file, replacing 'GROQ_API_KEY' with your personal key obtained from Groq.
```env
# Example .env file
GROQ_API_KEY=your_groq_api_key_here
```
#### ⚠️ Important:
To successfully run the experiment, you must have access to Groq's Developer Tier, which supports pay-per-token usage. Lower tiers or trial access may not be sufficient.

Now run these scripts (in order as shown)

```bash
  python src/DB_retrieval/DB_extractor.py
```
```bash
  python src/schema_linking/tables_extraction/tables_extractor.py
```
For the last script you can choose between two different prompt types:
- The first one oriented towards attribute **REDUCTION**
    ```bash
    python src/schema_linking/attributes_extraction/attributes_extractor.py r
    ```
- The second oriented towards **STRICT RECALL**
    ```bash
    python src/schema_linking/attributes_extraction/attributes_extractor.py sr
    ```
## Evaluation
The results of these scripts will be found inside the `results` folder
To perform the evaluation of the various pipeline steps you need to run the following scripts respectively:
```bash
  python src/DB_retrieval/evaluation.py
```
```bash
  python src/schema_linking/tables_extraction/evaluation.py
```
```bash
  python src/schema_linking/attributes_extraction/evaluation.py
```

### 📄 If you want to explore the project further, refer to the PDF available in the repository.