import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import run_nllb_onnx

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLLB Translator (ONNX)")
        self.root.geometry("600x500")

        self.models = {} # Store models by name
        self.tokenizers = {} # Store tokenizers by name
        self.model_names = ["600M", "1.3B", "3.3B"]
        self.model_ids = {
            "600M": "facebook/nllb-200-distilled-600M",
            "1.3B": "facebook/nllb-200-distilled-1.3B",
            "3.3B": "facebook/nllb-200-3.3B"
        }
        self.is_loading = False

        # Language codes (subset of FLORES-200)
        # Indian Languages (NLLB-200 / FLORES-200)
        self.languages = {
            "English": "eng_Latn",

            # Indo-Aryan
            "Hindi": "hin_Deva",
            "Bengali": "ben_Beng",
            "Gujarati": "guj_Gujr",
            "Marathi": "mar_Deva",
            "Punjabi": "pan_Guru",
            "Odia": "ory_Orya",
            "Assamese": "asm_Beng",
            "Maithili": "mai_Deva",
            "Bhojpuri": "bho_Deva",
            "Awadhi": "awa_Deva",
            "Magahi": "mag_Deva",
            "Rajasthani": "raj_Deva",
            "Sindhi": "snd_Arab",
            "Urdu": "urd_Arab",
            "Kashmiri": "kas_Arab",
            "Kashmiri (Devanagari)": "kas_Deva",
            "Nepali": "npi_Deva",
            "Santali": "sat_Olck",

            # Dravidian
            "Tamil": "tam_Taml",
            "Telugu": "tel_Telu",
            "Kannada": "kan_Knda",
            "Malayalam": "mal_Mlym",

            "Manipuri (Meitei)": "mni_Beng",
            "Manipuri (Meitei Mayek)": "mni_Mtei",
            "Bodo": "brx_Deva",
            "Dogri": "doi_Deva",

            # Classical
            "Sanskrit": "san_Deva"
        }


        self.create_widgets()
        
        # Load model in a separate thread
        self.status_label.config(text="Initializing... Models will load sequentially.")
        threading.Thread(target=self.load_all_models_thread, daemon=True).start()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Language Selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)

        ttk.Label(lang_frame, text="Source:").pack(side=tk.LEFT)
        self.src_lang_var = tk.StringVar(value="English")
        self.src_combo = ttk.Combobox(lang_frame, textvariable=self.src_lang_var, values=list(self.languages.keys()), state="readonly")
        self.src_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(lang_frame, text="Target:").pack(side=tk.LEFT)
        self.tgt_lang_var = tk.StringVar(value="Hindi")
        self.tgt_combo = ttk.Combobox(lang_frame, textvariable=self.tgt_lang_var, values=list(self.languages.keys()), state="readonly")
        self.tgt_combo.pack(side=tk.LEFT, padx=5)

        # Input Area
        ttk.Label(main_frame, text="Input Text:").pack(anchor=tk.W, pady=(10, 0))
        self.input_text = scrolledtext.ScrolledText(main_frame, height=8)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Translate Button
        self.translate_btn = ttk.Button(main_frame, text="Translate", command=self.start_translation, state=tk.DISABLED)
        self.translate_btn.pack(pady=10)

        # Output Areas
        output_container = ttk.Frame(main_frame)
        output_container.pack(fill=tk.BOTH, expand=True, pady=5)

        self.outputs = {}
        self.time_labels = {}

        for i, name in enumerate(self.model_names):
            frame = ttk.LabelFrame(output_container, text=f"NLLB {name}")
            frame.grid(row=0, column=i, padx=5, sticky="nsew")
            output_container.columnconfigure(i, weight=1)

            txt = scrolledtext.ScrolledText(frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
            txt.pack(fill=tk.BOTH, expand=True)
            self.outputs[name] = txt

            time_lbl = ttk.Label(frame, text="Time: 0.00s")
            time_lbl.pack(anchor=tk.E)
            self.time_labels[name] = time_lbl

        # Status Bar
        self.status_label = ttk.Label(self.root, text="Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_all_models_thread(self):
        for name in self.model_names:
            try:
                self.status_label.config(text=f"Loading model {name}... (This may take a while for 3.3B)")
                model, tokenizer = run_nllb_onnx.load_model(model_id=self.model_ids[name])
                self.models[name] = model
                self.tokenizers[name] = tokenizer
            except Exception as e:
                self.root.after(0, lambda n=name, err=e: self.status_label.config(text=f"Error loading {n}: {str(err)}"))
                # Continue to next model even if one fails
        
        self.root.after(0, self.on_all_models_loaded)

    def on_all_models_loaded(self):
        loaded_count = len(self.models)
        self.status_label.config(text=f"Loaded {loaded_count}/{len(self.model_names)} models. Ready.")
        if loaded_count > 0:
            self.translate_btn.config(state=tk.NORMAL)

    def start_translation(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text or not self.models:
            return

        src_code = self.languages[self.src_lang_var.get()]
        tgt_code = self.languages[self.tgt_lang_var.get()]

        self.translate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Translating with all models...")

        for name in self.model_names:
            if name in self.models:
                # Clear previous output
                self.outputs[name].config(state=tk.NORMAL)
                self.outputs[name].delete("1.0", tk.END)
                self.outputs[name].config(state=tk.DISABLED)
                self.time_labels[name].config(text="Time: ...")
                
                # Start individual translation thread
                threading.Thread(target=self.translate_single_model, args=(name, text, src_code, tgt_code), daemon=True).start()

    def translate_single_model(self, name, text, src_code, tgt_code):
        try:
            import time
            start_time = time.time()
            result = run_nllb_onnx.translate_text(text, src_code, tgt_code, self.models[name], self.tokenizers[name])
            elapsed = time.time() - start_time
            self.root.after(0, lambda: self.show_result(name, result, elapsed))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(name, str(e)))

    def show_result(self, name, result, elapsed):
        self.outputs[name].config(state=tk.NORMAL)
        self.outputs[name].insert(tk.END, result)
        self.outputs[name].config(state=tk.DISABLED)
        self.time_labels[name].config(text=f"Time: {elapsed:.2f}s")
        
        # Check if all models are done
        # (Simple check: if button is disabled, we might still be waiting for some)
        # We'll just re-enable it here for now, or use a counter
        self.translate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Translation complete.")

    def show_error(self, name, error_msg):
        self.outputs[name].config(state=tk.NORMAL)
        self.outputs[name].insert(tk.END, f"Error: {error_msg}")
        self.outputs[name].config(state=tk.DISABLED)
        self.translate_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
