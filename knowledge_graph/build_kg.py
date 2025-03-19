#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import torch
from neo4j import GraphDatabase
from model_code.tcmer_model import TCMERModel
from transformers import BertTokenizer

def load_tcmer_model(model_path, tokenizer_path, number_of_labels):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = TCMERModel(pretrained_name=tokenizer_path, num_labels=number_of_labels)
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, tokenizer

def recognize_entities(input_text, entity_model, entity_tokenizer, label_mapping=None):
    if label_mapping is None:
        label_mapping = {0: "O", 1: "B-SYM", 2: "I-SYM", 3: "B-HERB", 4: "I-HERB", 5: "B-TREAT", 6: "I-TREAT"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entity_model.to(device)
    tokens_list = list(input_text)
    encoding_output = entity_tokenizer(tokens_list, is_split_into_words=True, truncation=True, return_tensors="pt")
    ids_tensor = encoding_output["input_ids"].to(device)
    mask_tensor = encoding_output["attention_mask"].to(device)
    with torch.no_grad():
        loss_value, logits_tensor = entity_model(ids_tensor, mask_tensor, labels=None)
    logits_array = logits_tensor[0].cpu().numpy()
    predictions = logits_array.argmax(axis=-1)
    entity_list = []
    buffer_list = []
    current_type = None
    for index_value, label_id in enumerate(predictions):
        label_str = label_mapping.get(label_id, "O")
        char_str = tokens_list[index_value]
        if label_str.startswith("B-"):
            if buffer_list:
                entity_list.append({"entity": "".join(buffer_list), "type": current_type})
                buffer_list = []
            current_type = label_str[2:]
            buffer_list = [char_str]
        elif label_str.startswith("I-") and current_type == label_str[2:]:
            buffer_list.append(char_str)
        else:
            if buffer_list:
                entity_list.append({"entity": "".join(buffer_list), "type": current_type})
                buffer_list = []
            current_type = None
    if buffer_list:
        entity_list.append({"entity": "".join(buffer_list), "type": current_type})
    relation_list = []
    herb_list = [item["entity"] for item in entity_list if item["type"] == "HERB"]
    symptom_list = [item["entity"] for item in entity_list if item["type"] == "SYM"]
    for herb_item in herb_list:
        for symptom_item in symptom_list:
            relation_list.append({"head": herb_item, "tail": symptom_item, "rel_type": "TREATS"})
    return {"text": input_text, "entities": entity_list, "relations": relation_list}

def insert_relations(neo4j_uri, neo4j_user, neo4j_password, data_object):
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        for relation_item in data_object["relations"]:
            head_str = relation_item["head"]
            tail_str = relation_item["tail"]
            relation_type = relation_item["rel_type"]
            cypher_query = f"""
            MERGE (nodeA:TCM {{name: $h}})
            MERGE (nodeB:TCM {{name: $t}})
            MERGE (nodeA)-[r:{relation_type}]->(nodeB)
            """
            session.run(cypher_query, h=head_str, t=tail_str)
    driver.close()

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--llm_output_file", type=str, default="llm_results.json")
    argument_parser.add_argument("--tcmer_model_path", type=str, default="tcmer_model.bin")
    argument_parser.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")
    argument_parser.add_argument("--num_labels", type=int, default=7)
    argument_parser.add_argument("--neo4j_url", type=str, default="bolt://localhost:7687")
    argument_parser.add_argument("--neo4j_user", type=str, default="neo4j")
    argument_parser.add_argument("--neo4j_password", type=str, default="your_password")
    arguments = argument_parser.parse_args()
    loaded_model, loaded_tokenizer = load_tcmer_model(arguments.tcmer_model_path, arguments.tokenizer_name, arguments.num_labels)
    with open(arguments.llm_output_file, "r", encoding="utf-8") as input_file:
        data_array = json.load(input_file)
    for data_item in data_array:
        recognized_result = recognize_entities(data_item["text"], loaded_model, loaded_tokenizer)
        insert_relations(arguments.neo4j_url, arguments.neo4j_user, arguments.neo4j_password, recognized_result)
    print("Done.")

if __name__ == "__main__":
    main()
