import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_manipulation import create_table_name_mapping, get_sql_table_names, create_db_schema_dictionary
from ansi_colors import *
import paths


def TAB_extraction_eval(dataset):

    if dataset == 'BIRDdev':
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'BIRDdev_TAB_extractor_checkpoint.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')
    elif dataset == 'SPIDERdev1':
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'SPIDERdev1_TAB_extractor.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    strict_recall_samples = 0
    samples = 0
    reductions = []
    wrong_db = []

    for sample in data:

        #if sample['question_id'] < 800:
        #    continue
        samples += 1

        db = sample['db_id']
        tab_result = sample['TAB_result']

        if db != sample['DB_result']:
            if len(tab_result) == 1 and tab_result[0] == 'NONE':
                wrong_db.append(1)
            else:
                wrong_db.append(0)
                continue

        tab_original_result = [table_name_mapping[db][name] for name in tab_result]
        tab_needed = get_sql_table_names(sample['SQL'])
        # To lower case
        tab_original_result = [s.lower() for s in tab_original_result]
        tab_needed = [s.lower() for s in tab_needed]
        is_strict = True


        for tab in tab_needed:
            if tab not in tab_original_result:
                is_strict = False
                print(sample['question_id'])
                print(tab_needed)
                print(tab_original_result)
                print("\n\n")
                break
        
        if is_strict:
            strict_recall_samples += 1

            result_att = 0
            for t in tab_result:
                result_att += len(database_schemas[db][t])
            
            db_schema = database_schemas[db]
            total_att = sum(len(columns) for columns in db_schema.values())

            reduction = 1 - (result_att / total_att)
            reductions.append(reduction)

    strict_recall = strict_recall_samples / samples
    table_strict_recall = strict_recall_samples / (samples - len(wrong_db))
    avg_reduction = sum(reductions) / len(reductions)
    print(f"\n- {GREEN}STRICT RECALL (for table extraction only):{RESET} {table_strict_recall}")
    print(f"- {GREEN}STRICT RECALL:{RESET} {strict_recall}")
    print(f"- {GREEN}REDUCTION avg:{RESET} {avg_reduction}")
    if len(wrong_db) != 0:
        accuracy_wrong_db = sum(wrong_db) / len(wrong_db)
        print(f"- {GREEN}WRONG DB accuracy:{RESET} {accuracy_wrong_db}\n")
    else:
        print(f"- {CYAN}no wrong DB detected!:{RESET}\n")



if __name__ == '__main__':

    dataset = 'SPIDERdev1'
    TAB_extraction_eval(dataset)