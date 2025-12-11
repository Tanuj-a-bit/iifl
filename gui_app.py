import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import run_nllb_onnx

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLLB Translator (ONNX)")
        self.root.geometry("600x500")

        self.model = None
        self.tokenizer = None
        self.is_loading = False

        # Language codes (subset of FLORES-200)
        self.languages = {
            "English": "eng_Latn",
            "Hindi": "hin_Deva",
            "French": "fra_Latn",
            "Spanish": "spa_Latn",
            "German": "deu_Latn",
            "Chinese (Simplified)": "zho_Hans",
            "Japanese": "jpn_Jpan",
            "Russian": "rus_Cyrl",
            "Arabic": "arb_Arab"
        }

        self.create_widgets()
        
        # Load model in a separate thread
        self.status_label.config(text="Loading model... Please wait.")
        threading.Thread(target=self.load_model_thread, daemon=True).start()

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

        # Output Area
        ttk.Label(main_frame, text="Translation:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(main_frame, height=8, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Status Bar
        self.status_label = ttk.Label(self.root, text="Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model_thread(self):
        try:
            self.model, self.tokenizer = run_nllb_onnx.load_model()
            self.root.after(0, self.on_model_loaded)
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error loading model: {str(e)}"))

    def on_model_loaded(self):
        self.status_label.config(text="Model loaded. Ready.")
        self.translate_btn.config(state=tk.NORMAL)

    def start_translation(self):
        if not self.model:
            return

        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return

        src_code = self.languages[self.src_lang_var.get()]
        tgt_code = self.languages[self.tgt_lang_var.get()]

        self.translate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Translating...")
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)

        threading.Thread(target=self.translate_thread, args=(text, src_code, tgt_code), daemon=True).start()

    def translate_thread(self, text, src_code, tgt_code):
        try:
            result = run_nllb_onnx.translate_text(text, src_code, tgt_code, self.model, self.tokenizer)
            self.root.after(0, lambda: self.show_result(result))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def show_result(self, result):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, result)
        self.output_text.config(state=tk.DISABLED)
        self.status_label.config(text="Translation complete.")
        self.translate_btn.config(state=tk.NORMAL)

    def show_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg}")
        self.translate_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
