from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import os
import shutil

def quantize_model(model_id="facebook/nllb-200-distilled-600M", output_dir="nllb_onnx_quantized"):
    temp_onnx_dir = "temp_nllb_onnx"
    
    print(f"Exporting model to {temp_onnx_dir}...")
    # Load and export to local directory
    model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
    model.save_pretrained(temp_onnx_dir)
    # Also save tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(temp_onnx_dir)
    
    print(f"Quantizing model from {temp_onnx_dir}...")
    
    # Define quantization config
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    
    # Create quantizer from local ONNX model
    quantizer = ORTQuantizer.from_pretrained(temp_onnx_dir, feature="seq2seq-lm-with-past")
    
    # Quantize and save
    quantizer.export(
        quantization_config=qconfig,
        model_output=output_dir,
        use_external_data_format=False,
    )
    
    # Copy tokenizer to output dir so it's a self-contained model dir
    tokenizer.save_pretrained(output_dir)
    
    print(f"Quantized model saved to: {output_dir}")
    
    # Cleanup temp dir
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_onnx_dir)
    print("Done.")

if __name__ == "__main__":
    try:
        quantize_model()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
