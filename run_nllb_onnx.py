from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import time

def load_model(model_id="facebook/nllb-200-distilled-600M", local_model_path=None):
    """
    Loads the NLLB model and tokenizer.
    """
    print(f"Loading model... (Source: {local_model_path if local_model_path else model_id})")
    
    if local_model_path:
        model = ORTModelForSeq2SeqLM.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        print("Note: If this is the first run, it might take some time to export the model to ONNX.")
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Model loaded.")
    return model, tokenizer

def translate_text(text, src_lang, tgt_lang, model, tokenizer):
    """
    Translates text using the loaded NLLB model.
    """
    # NLLB requires setting the source language in the tokenizer
    tokenizer.src_lang = src_lang

    print(f"Translating: '{text}' from {src_lang} to {tgt_lang}")
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    end_time = time.time()
    try:
        print(f"Translation: {result}")
    except UnicodeEncodeError:
        print(f"Translation: {result.encode('utf-8')}")
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    return result

if __name__ == "__main__":
    # Example usage
    input_text = "Hello, how are you doing today?"
    
    # Language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    source_language = "eng_Latn" # English
    target_language = "hin_Deva" # Hindi
    
    # Load model once
    model, tokenizer = load_model()
    
    # Translate
    translate_text(input_text, source_language, target_language, model, tokenizer)
