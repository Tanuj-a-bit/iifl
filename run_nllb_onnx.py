from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import time

def translate_text(text, src_lang, tgt_lang, model_id="facebook/nllb-200-distilled-600M", local_model_path=None):
    """
    Translates text using NLLB model with ONNX Runtime.
    
    Args:
        text (str): Text to translate.
        src_lang (str): Source language code (e.g., 'eng_Latn').
        tgt_lang (str): Target language code (e.g., 'fra_Latn').
        model_id (str): Hugging Face model ID to export/load if local_path is not provided.
        local_model_path (str): Path to a pre-exported ONNX model directory.
    """
    print(f"Loading model... (Source: {local_model_path if local_model_path else model_id})")
    
    if local_model_path:
        # Load pre-exported model from local path
        model = ORTModelForSeq2SeqLM.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Load from Hub and export to ONNX (if not already cached/exported)
        # Note: export=True will convert the model to ONNX. 
        # For a truly "pre-exported" workflow without conversion overhead on first run, 
        # one should run: optimum-cli export onnx --model facebook/nllb-200-distilled-600M nllb_onnx/
        print("Note: If this is the first run, it might take some time to export the model to ONNX.")
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Model loaded.")

    # NLLB requires setting the source language in the tokenizer
    tokenizer.src_lang = src_lang

    print(f"Translating: '{text}' from {src_lang} to {tgt_lang}")
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    # forced_bos_token_id is required for NLLB to specify target language
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    end_time = time.time()
    print(f"Translation: {result}")
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    return result

if __name__ == "__main__":
    # Example usage
    input_text = "Hello, how are you doing today?"
    
    # Language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    source_language = "eng_Latn" # English
    target_language = "hin_Deva" # Hindi
    
    # You can specify a local path if you have a pre-exported model, e.g., "./nllb_onnx"
    # translate_text(input_text, source_language, target_language, local_model_path="./nllb_onnx")
    
    # Or let Optimum handle the export/download
    translate_text(input_text, source_language, target_language)
