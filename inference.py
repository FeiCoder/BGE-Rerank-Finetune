import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to the fine-tuned model
model_path = './output_model'

print(f"Loading model from {model_path} ...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except OSError:
    print(f"Could not find model at {model_path}. Please ensure training has completed successfully.")
    exit(1)

model.eval()

# Example pairs from the user request (Chinese)
pairs = [
    ['熊猫是什么？', '你好'], 
    ['熊猫是什么？', '大熊猫（学名：Ailuropoda melanoleuca），也称作大猫熊，一般称为“熊猫”或“猫熊”，是中国的特有物种。']
]

print("\nCalculating scores for the following pairs:")
for i, (query, doc) in enumerate(pairs):
    print(f"Pair {i+1}:")
    print(f"  Query: {query}")
    print(f"  Doc:   {doc}")

with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    print("\nScores:")
    print(scores)

    # Sigmoid scores for easier interpretation (optional, but common for rerankers)
    sigmoid_scores = torch.sigmoid(scores)
    print("\nSigmoid Scores:")
    print(sigmoid_scores)
