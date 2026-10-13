import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class FolkInAssistant:
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path="../models/folkin-qwen-lora"):
        print("Memuat Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print("Memuat Base Model...")
        # Menggunakan float16 dan device_map="auto" agar ringan dan otomatis menggunakan GPU jika ada
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" 
        )
        
        print("Menggabungkan LoRA Adapter...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("Qwen + LoRA berhasil disiapkan!")

    def generate_response(self, instruction, input_text):
        # Format prompt ini harus sama persis dengan yang kamu gunakan saat training LoRA
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50, 
                temperature=0.7, 
                repetition_penalty=1.1
            )
            
        # Decode hasil dan ambil bagian setelah "Output: "
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("Output:")[-1].strip()
        
        return response

# --- Contoh Pengujian Mandiri ---
if __name__ == "__main__":
    assistant = FolkInAssistant(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        adapter_path="./models/folkin-qwen-lora" # Sesuaikan path
    )
    
    tugas_user = "Menyelesaikan laporan praktikum jaringan komputer."
    status = "HP terdeteksi sedang digenggam saat sesi belajar."
    
    pesan = assistant.generate_response(status, tugas_user)
    print("\n[Pesan FolkIn AI]:", pesan)