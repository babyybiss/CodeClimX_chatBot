from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM


inp = "틱택토는 AI로 어떻게 구현이 되나요?"

models ={
    "model_1": "Helsinki-NLP/opus-mt-ko-en",
    "model_2": "DunnBC22/opus-mt-ko-en-Korean_Parallel_Corpora"
}

for model_name, model in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_i = AutoModelForSeq2SeqLM.from_pretrained(model)

    input_ids = tokenizer(inp, return_tensors="pt").input_ids
    outputs = model_i.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
    print(f"\nModel: {model_name} Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    
    
    
# ========Facebook mbart model=========            
article_hi = "로지스틱 회귀가 뭐야?"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "ko_KR"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
