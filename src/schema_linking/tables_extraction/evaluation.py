import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_manipulation import create_table_name_mapping, create_db_schema_dictionary, create_db_original_schema_dictionary, extract_tables_and_columns
from ansi_colors import *
import paths


def TAB_extraction_eval(dataset):

    if dataset == 'BIRDdev':
        print(f"\n{CYAN}BIRDdev TAB extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'BIRDdev_TAB_extractor.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')
        database_original_schemas = create_db_original_schema_dictionary('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')
    elif dataset == 'SPIDERdev1':
        print(f"\n{CYAN}SPIDERdev1 TAB extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'SPIDERdev1_TAB_extractor.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
        database_original_schemas = create_db_original_schema_dictionary('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
    elif dataset == 'BEAVER':
        print(f"\n{CYAN}BEAVER TAB extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'BEAVER_TAB_extractor.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.BEAVER.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.BEAVER.value + 'dev_tables.json')
        database_original_schemas = create_db_original_schema_dictionary('../../' + paths.DATASETS.BEAVER.value + 'dev_tables.json')
    elif dataset == 'ARCHER':
        print(f"\n{CYAN}ARCHER TAB extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'ARCHER_TAB_extractor.json'
        table_name_mapping = create_table_name_mapping('../../' + paths.DATASETS.ARCHER.value + 'dev_tables.json')
        database_schemas = create_db_schema_dictionary('../../' + paths.DATASETS.ARCHER.value + 'dev_tables.json')
        database_original_schemas = create_db_original_schema_dictionary('../../' + paths.DATASETS.ARCHER.value + 'dev_tables.json')
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
            print(f"{RED}File not found: {filename}. Skipping evaluation.{RESET}\n")
            return

    strict_recall_samples = 0
    samples = 0
    reductions = []
    wrong_db = []
    errors_ids = []

    recall = []
    precision = []
    em = 0

    for sample in data:

        #if sample['question_id'] < 800:
        #    continue
        samples += 1

        db = sample['db_id']
        tab_result = sample['TAB_result']

        if db != sample['DB_result']:
            if len(tab_result) == 1 and tab_result[0] == 'NONE':
                wrong_db.append(1)
                continue
            else:
                wrong_db.append(0)
                continue

        # Correct DB but no table identified by llm
        if len(tab_result) == 1 and tab_result[0] == 'NONE':
            precision.append(0)
            recall.append(0)
            continue
        
        try:
            tab_original_result = [table_name_mapping[db][name] for name in tab_result]
        except Exception as e:
            print(f"{RED}Error on {sample['question_id']}:{RESET} {e}")
            continue

        tab_needed = extract_tables_and_columns(sample['SQL'])
        # To lower case
        tab_original_result = [s.lower() for s in tab_original_result]
        tab_needed = [s.lower() for s in tab_needed['table']]
        tables_original_db = [s.lower() for s in list(database_original_schemas[db].keys())]
        is_strict = True

        needed = set([a for a in tab_needed if a in tables_original_db])
        result = set(tab_original_result)
        p = len(needed & result) / len(result) if result else 0.0
        precision.append(p)
        r = len(needed & result) / len(needed) if needed else 0.0
        recall.append(r)
        if p == 1 and r == 1:
            em += 1

        for tab in needed:
            if tab not in tab_original_result:
                errors_ids.append(sample['question_id'])
                is_strict = False
                break

        if is_strict:
            strict_recall_samples += 1

            # Compute reduction (1.0 means that tab_original_result = needed)
            result_tab = 0
            for t in tab_original_result:
                if t not in needed:
                    result_tab += 1
            db_schema = database_schemas[db]
            total_tab = len(db_schema) - len(needed)

            if result_tab == 0 or total_tab == 0:
                reduction = 1
            else:
                reduction = 1 - (result_tab / total_tab)
            reductions.append(reduction)

    strict_recall = strict_recall_samples / samples
    table_strict_recall = strict_recall_samples / (samples - len(wrong_db))
    avg_reduction = sum(reductions) / len(reductions)
    mean_precision = sum(precision) / len(precision) if precision else 0.0
    mean_recall = sum(recall) / len(recall) if recall else 0.0
    print(f"- {GREEN}TABLE STRICT RECALL:{RESET} {strict_recall}")
    print(f"- {GREEN}TABLE STRICT RECALL (for table extraction only):{RESET} {table_strict_recall}")
    print(f"- {GREEN}TABLE REDUCTION avg:{RESET} {avg_reduction}")
    if len(wrong_db) != 0:
        accuracy_wrong_db = sum(wrong_db) / len(wrong_db)
        print(f"- {GREEN}WRONG DB accuracy:{RESET} {accuracy_wrong_db}")
    else:
        print(f"- {CYAN}no wrong DB detected!:{RESET}")
    print(f"-----")
    print(f"- {GREEN}Precision avg:{RESET} {mean_precision}")
    print(f"- {GREEN}Recall avg:{RESET} {mean_recall}")
    print(f"- {GREEN}EM:{RESET} {em/samples}\n")
    
    return errors_ids



if __name__ == '__main__':

    dataset = 'BIRDdev'
    TAB_extraction_eval(dataset)

    dataset = 'SPIDERdev1'
    TAB_extraction_eval(dataset)

    dataset = 'BEAVER'
    TAB_extraction_eval(dataset)

    dataset = 'ARCHER'
    TAB_extraction_eval(dataset)