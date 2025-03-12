import json
import re

# Path to your input CSV file and output JSONL file
input_file = "/Users/shrishmishra/Desktop/dataset/contributor_questions.csv"
output_file = "/Users/shrishmishra/Desktop/dataset/contributor_questions.jsonl"

# Process the file
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Extract question ID, question, and answer
        match = re.match(r'(\d+),(.*?),(.*)', line.strip())
        if match:
            question_id, question, answer = match.groups()
            
            # Create a JSON object and write to output file
            json_obj = {
                "id": int(question_id),
                "question": question.strip(),
                "answer": answer.strip()
            }
            f_out.write(json.dumps(json_obj) + '\n')