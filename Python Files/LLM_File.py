## this is the code for the llm call
import sys
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#softmax because is a categorized function

import torch
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline as transformers_pipeline,
)

#LLM using Langchain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

MODEL_PATH = "./invoice_model"
INVOICE_CATEGORIES = ["Pending", "Paid", "Overdue"]

invoice_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
invoice_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
invoice_model.eval()

def get_invoice_category(text_input):
    encoded = invoice_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = invoice_model(**encoded).logits
        probabilities = softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        category = INVOICE_CATEGORIES[predicted.item()]
    return category, round(confidence.item(), 4)

# Model creation using google t5
GEN_MODEL_ID = "google/flan-t5-base"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)

text_gen_pipeline = transformers_pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer)
llm_wrapper = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Template for generating LLM
recommendation_prompt = PromptTemplate(
    input_variables=["category", "raw_text"],
    template=(
        "You are an assistant helping analyze invoice statuses.\n"
        "Invoice record: '{raw_text}'\n"
        "Category detected: '{category}'\n"
        "Provide a brief and professional recommendation based on this classification."
    )
)

generator_chain = LLMChain(llm=llm_wrapper, prompt=recommendation_prompt)

#this is for split the text
def parse_invoice_text(line: str) -> str:
    try:
        fields = line.split(",")
        return fields[2].strip() if len(fields) > 2 else line
    except Exception:
        return line

#request function to be call by RPA

def handle_request():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"error": "there is not an input"}))
            return

        request_data = json.loads(raw_input)
        invoice_text = request_data.get("text", "error, no text received").strip()

        if not invoice_text:
            print(json.dumps({"error": ""}))
            return


        content_for_classification = parse_invoice_text(invoice_text)

   
        category, confidence_score = get_invoice_category(content_for_classification)

        suggestion = generator_chain.run(category=category, raw_text=invoice_text).strip()

        result = {
            "label": category,
            "confidence": confidence_score,
            "recommendation": suggestion
        }

        print(json.dumps(result, ensure_ascii=False))

    except Exception as err:
        print(json.dumps({"error": str(err)}))


#Main function to call the handle request
if __name__ == "__main__":
    handle_request()