import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tables_extraction.evaluation import TAB_extraction_eval
from data_manipulation import (
    create_attribute_mapping,
    create_db_original_schema_dictionary,
    create_table_name_mapping,
    extract_qualified_columns,
    lowercase_dict,
)
from ansi_colors import *
import paths


def ATT_extraction_eval(dataset):

    # errors_ids = TAB_extraction_eval(dataset)

    if dataset == 'BIRDdev':
        print(f"\n{CYAN}BIRDdev ATT extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.ATT_RETRIEVAL.value + 'BIRDdev_ATT_extractor.json'
        schema_filename = '../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json'
    elif dataset == 'SPIDERdev1':
        print(f"\n{CYAN}SPIDERdev1 ATT extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.ATT_RETRIEVAL.value + 'SPIDERdev1_ATT_extractor.json'
        schema_filename = '../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    attributes_mapping_dict = lowercase_dict(create_attribute_mapping(schema_filename))
    table_name_mapping_dict = lowercase_dict(create_table_name_mapping(schema_filename))
    database_original_schemas = lowercase_dict(create_db_original_schema_dictionary(schema_filename))
    sqlglot_schemas = {
        db_id: {table: {column: "UNKNOWN" for column in columns} for table, columns in schema.items()}
        for db_id, schema in database_original_schemas.items()
    }

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    strict_recall_samples = 0
    samples = 0
    reductions = []
    wrong_db = []

    recall = []
    precision = []
    em = 0
    
    for sample in data:

        #if sample['question_id'] < 800:
        #    continue
        samples += 1
        q_id = sample['question_id']

        # if q_id in errors_ids:
        #     continue

        db = sample['db_id']
        tab_result = sample['TAB_result']
        att_result = sample['ATT_result']

        if db != sample['DB_result']:
            if len(tab_result) == 1 and tab_result[0] == 'NONE':
                wrong_db.append(1)
                continue
            if len(att_result) == 1 and att_result[0] == 'NONE':
                wrong_db.append(1)
                continue
            else:
                wrong_db.append(0)
                continue

        # Correct DB but no attributes identified by llm
        if len(att_result) == 1 and att_result[0] == 'NONE':
            precision.append(0)
            recall.append(0)
            continue

        att_needed = extract_qualified_columns(sample['SQL'], sqlglot_schemas[db.lower()])
        att_original_result = []
        for res in att_result:
            try:
                table, column = res.split('.', 1)
            except ValueError:
                #print(f"{RED}SPLIT error at:{RESET} {q_id}, {RED}Invalid format:{RESET} {res}")
                continue
            try:
                att_original_result.append((
                    table_name_mapping_dict[db.lower()][table.lower()],
                    attributes_mapping_dict[db.lower()][table.lower()][column.lower()],
                ))
            except KeyError:
                #print(f"{RED}KEY error at:{RESET} {q_id}, {RED}Result:{RESET} {res}")
                continue

        attributes_original_db = {
            (table, column)
            for table, columns in database_original_schemas[db.lower()].items()
            for column in columns
        }
        needed = set(att_needed)
        result = set(att_original_result)
        p = len(needed & result) / len(result) if result else 0.0
        precision.append(p)
        r = len(needed & result) / len(needed) if needed else 0.0
        recall.append(r)
        if p == 1 and r == 1:
            em += 1

        is_strict = needed.issubset(result)
        # if not is_strict:
        #     print(q_id)
        #     print(sorted(needed))
        #     print(sorted(result))
        #     print('-'*50)

        if is_strict:
            strict_recall_samples += 1

            # Compute reduction (1.0 means that att_original_result = att_needed)
            result_att = len(result - needed)
            total_att = len(attributes_original_db - needed)

            if result_att <= 0:
                reduction = 1
            else:
                reduction = 1 - (result_att / total_att)
            reductions.append(reduction)

    strict_recall = strict_recall_samples / samples
    att_strict_recall = strict_recall_samples / (samples - len(wrong_db))
    avg_reduction = sum(reductions) / len(reductions)
    mean_precision = sum(precision) / len(precision) if precision else 0.0
    mean_recall = sum(recall) / len(recall) if recall else 0.0

    print(f"- {GREEN}STRICT RECALL (for schema linking only):{RESET} {att_strict_recall}")
    print(f"- {GREEN}STRICT RECALL:{RESET} {strict_recall}")
    print(f"- {GREEN}REDUCTION avg:{RESET} {avg_reduction}")
    if len(wrong_db) != 0:
        accuracy_wrong_db = sum(wrong_db) / len(wrong_db)
        print(f"- {GREEN}WRONG DB accuracy:{RESET} {accuracy_wrong_db}")
    else:
        print(f"- {CYAN}no wrong DB detected!:{RESET}")

    print(f"\n-----\n")
    print(f"- {GREEN}Precision avg:{RESET} {mean_precision}")
    print(f"- {GREEN}Recall avg:{RESET} {mean_recall}")
    print(f"- {GREEN}EM:{RESET} {em/samples}\n")



if __name__ == '__main__':

    print(f"===================={BLUE}BIRDdev end-to-end pipeline EVALUATION{RESET}=====================")
    dataset = 'BIRDdev'
    ATT_extraction_eval(dataset)

    print(f"===================={BLUE}SPIDERdev1.0 end-to-end pipeline EVALUATION{RESET}====================")
    dataset = 'SPIDERdev1'
    ATT_extraction_eval(dataset)
