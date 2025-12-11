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
    print("--- NLLB ONNX Translator ---")
    
    # Language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    source_language = "eng_Latn" # English
    target_language = "hin_Deva" # Hindi
    
    # Load model once
    model, tokenizer = load_model()
    
    print(f"\nCurrent Configuration: Source={source_language}, Target={target_language}")
    print("Type 'quit' to exit, 'change' to change languages.")

    while True:
        try:
            user_input = input("\nEnter text to translate: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if user_input.lower() == 'change':
                source_language = input("Enter Source Language Code (e.g., eng_Latn): ").strip()
                target_language = input("Enter Target Language Code (e.g., hin_Deva): ").strip()
                print(f"Configuration updated: Source={source_language}, Target={target_language}")
                continue

            if not user_input:
                continue

            # Translate
            translate_text(user_input, source_language, target_language, model, tokenizer)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
