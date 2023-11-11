from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch

app = FastAPI()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased-distilled-squad')

class InputData(BaseModel):
    context: str
    question: str

class AnswerResponse(BaseModel):
    answer: str


def predict(input_data: InputData):

    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict)

    input = {
        'context' : input_data.context,
        'question' : input_data.question
    }

    input_dict = tokenizer(input['context'], input['question'], return_tensors='pt')

    input_ids = input_dict['input_ids']
    attention_mask = input_dict['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask)

    start_logits = outputs[0]
    end_logits = outputs[1]

    all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    answer = ' '.join(all_tokens[torch.argmax(start_logits, 1)[0] : torch.argmax(end_logits, 1)[0]+1])
    answer = answer.replace(" ##", "").strip().capitalize()

    return answer

@app.post("/answer", response_model=AnswerResponse)
async def predict_endpoint(input_data: InputData):
    answer = predict(input_data)
    return {'answer' : answer}