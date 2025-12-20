from openai import OpenAI
import numpy as np
import pandas as pd
import os
import json
import csv
import json as pyjson
import re

OPENAI_API_KEY = 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

def extract_json_from_llm_output(text: str):
    """
    Extract the first valid JSON object from arbitrary LLM output.
    Handles: ```json ...```, ```...```, 'json {..}', plain {..}, mixed text.
    """

    cleaned = text.replace("```json", "").replace("```", "").strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")

    json_str = match.group(0)

    try:
        return pyjson.loads(json_str)
    except Exception as e:
        print("Failed JSON string:\n", json_str)
        raise e

def evaluate_similarity(question, pred_answer, true_answers):
    system_msg = (
        "You are an evaluator.You will compare a model prediction with the ground truth answers "
    )

    prompt = f"""
    Task: Given a question, evaluate how similar the model's answer is to the true answers.
    
    Rating Standard:
    - 5 = identical or essentially the same meaning
    - 4 = very similar
    - 3 = somewhat related
    - 2 = weakly related
    - 1 = unrelated or incorrect
    
    Scoring rule:
    1. If the predicted answer matches ANY of the true answers under **case-insensitive comparison**, score = **5**.
    2. If the predicted answer matches ANY of the true answers after **singular/plural normalization**  
       (e.g., "bun" == "buns", "tomato" == "tomatoes"), score = **5**.
    3. If the predicted answer matches after **simple lemmatization** (remove plural 's', normalize common forms), score = **5**.   
    4. Multi-item match:  
       If the predicted answer contains multiple items (e.g., "Mathematics and English")  and ALL items match the true answers after normalization and lemmatization, score = 5.
    5. Sub-category match:  
        If the predicted answer is a more specific type of a true answer  (e.g., "skewers" ⊂ "barbecue", "green tea" ⊂ "tea"), give a high score (4 or 5).
    6. Otherwise, score from 1–4 using semantic similarity.
    
    Question: {question}
    Predicted Answer: {pred_answer}
    True Answers: {true_answers}
    Now return your judgment **strictly in the following JSON format**:
    {{
      "score": <number between 1 and 5>,
      "reason": "<short explanation of why you chose this score>"
    }}
"""

    loop = 0
    while True:
        loop = loop + 1
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_msg,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )
        try:
            raw = response.choices[0].message.content
            result = extract_json_from_llm_output(raw)
            score = result.get("score", None)
            reason = result.get("reason", "").strip()
            score = int(score)
            if score not in [1, 2, 3, 4, 5]:
                raise ValueError(f"score out of range: {score}")
            return score, reason
        except :
            if loop == 3:
                score = 3
                reason = "parse_error"
                return score, reason
            else:
                pass

def compute_final_score(evaluation_file="evaluation_llm.csv"):
    df = pd.read_csv(evaluation_file)

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # score / 5
    df["normalized"] = df["score"] / 5

    avg_score = df["normalized"].mean()

    print(f"Final average similarity score (0~1): {avg_score:.4f}")
    return avg_score

def main(json_path, csv_path):
    with open(json_file, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    true_answer_map = {}
    for qid, item in data_json.items():
        en_answers = []
        for ann in item["annotations"]:
            en_answers.extend(ann["en_answers"])
        en_answers = list(set(en_answers))
        true_answer_map[qid] = en_answers

    df = pd.read_csv(csv_file)

    buffer = []
    BATCH_SIZE = 10
    out_path = "evaluation_llm.csv"

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "en_question", "score", "true_answer", "pred_answer",  "reason" ])
        writer.writeheader()

    for idx, row in df.iterrows():
        qid = row["id"]
        question = row["en_question"]
        pred_answer = row["answer"]
        true_answers = true_answer_map.get(qid, [])
        score, reason = evaluate_similarity(question,pred_answer, true_answers)
        row = {"id": qid, "en_question": question, "score":score, "true_answer": true_answers, "pred_answer":pred_answer,  "reason":reason}

        buffer.append(row)
        if len(buffer) >= BATCH_SIZE:
            with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "en_question", "score", "true_answer", "pred_answer",  "reason" ])
                writer.writerows(buffer)
            print(f"✔ write {len(buffer)}")
            buffer = []

    if buffer:
        with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "en_question", "score", "true_answer", "pred_answer",  "reason" ])
            writer.writerows(buffer)
        print(f"✔ last write{len(buffer)} ")




json_file = 'China_data.json'
csv_file = 'gpt-4-China_result.csv'
main(json_file,csv_file)
final_avg = compute_final_score("evaluation_llm.csv")
