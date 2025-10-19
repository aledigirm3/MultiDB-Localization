import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tables_extraction.evaluation import TAB_extraction_eval
from data_manipulation import extract_tables_and_columns, create_attribute_mapping, lowercase_dict
from ansi_colors import *
import paths


def ATT_extraction_eval(dataset):

    errors_ids = TAB_extraction_eval(dataset)

    if dataset == 'BIRDdev':
        print(f"\n{CYAN}BIRDdev ATT extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.ATT_RETRIEVAL.value + 'BIRDdev_ATT_extractor.json'
        attributes_mapping_dict = lowercase_dict(create_attribute_mapping('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json'))
    elif dataset == 'SPIDERdev1':
        print(f"\n{CYAN}SPIDERdev1 ATT extraction evaluation{RESET}")
        filename = '../../' + paths.RESULTS.ATT_RETRIEVAL.value + 'SPIDERdev1_ATT_extractor.json'
        attributes_mapping_dict = lowercase_dict(create_attribute_mapping('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'))
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
        q_id = sample['question_id']
        if q_id in errors_ids:
            continue

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

        # Correct DB but no table identified by llm
        if len(att_result) == 1 and att_result[0] == 'NONE':
            continue

        att_needed = extract_tables_and_columns(sample['SQL'])
        att_original_result = []
        for res in att_result:
            try:
                table, column = res.split('.', 1)
            except ValueError:
                #print(f"{RED}SPLIT error at:{RESET} {q_id}, {RED}Invalid format:{RESET} {res}")
                continue
            try:
                att_original_result.append(attributes_mapping_dict[db.lower()][table.lower()][column.lower()])
            except KeyError:
                #print(f"{RED}KEY error at:{RESET} {q_id}, {RED}Result:{RESET} {res}")
                continue


        attributes_original_db = []
        db_data = attributes_mapping_dict[db.lower()]
        for column_dict in db_data.values():
            attributes_original_db.extend(column_dict.values())
            
        # To lower case
        attributes_original_db = [s.lower() for s in attributes_original_db]
        #att_original_result = [s.lower() for s in att_original_result]
        att_needed = [s.lower() for s in att_needed['column']]
        is_strict = True

        att_needed_to_check = list(att_needed)
        att_original_result_to_check = list(att_original_result)
        for att in att_needed:
            if len(att_needed_to_check) == 0:
                break
            if att not in attributes_original_db:
                continue
            if att not in att_original_result_to_check:
                # print(q_id)
                # print(att_needed)
                # print(att_original_result)
                # print('-'*50)
                is_strict = False
                break
            else:
                att_needed_to_check.remove(att)
                att_original_result_to_check.remove(att)

        if is_strict:
            strict_recall_samples += 1

            # Compute reduction (1.0 means that att_original_result = att_needed)
            result_att = len(att_original_result) - len(att_needed)
            total_att = len(attributes_original_db) - len(att_needed)

            if result_att <= 0:
                reduction = 1
            else:
                reduction = 1 - (result_att / total_att)
            reductions.append(reduction)

    strict_recall = strict_recall_samples / samples
    table_strict_recall = strict_recall_samples / (samples - len(wrong_db))
    avg_reduction = sum(reductions) / len(reductions)
    print(f"- {GREEN}STRICT RECALL (for table extraction only):{RESET} {table_strict_recall}")
    print(f"- {GREEN}STRICT RECALL:{RESET} {strict_recall}")
    print(f"- {GREEN}REDUCTION avg:{RESET} {avg_reduction}")
    if len(wrong_db) != 0:
        accuracy_wrong_db = sum(wrong_db) / len(wrong_db)
        print(f"- {GREEN}WRONG DB accuracy:{RESET} {accuracy_wrong_db}\n")
    else:
        print(f"- {CYAN}no wrong DB detected!:{RESET}\n")



if __name__ == '__main__':

    print(f"===================={BLUE}BIRDdev end-to-end pipeline EVALUATION{RESET}=====================")
    dataset = 'BIRDdev'
    ATT_extraction_eval(dataset)

    print(f"===================={BLUE}SPIDERdev1.0 end-to-end pipeline EVALUATION{RESET}====================")
    dataset = 'SPIDERdev1'
    ATT_extraction_eval(dataset)
