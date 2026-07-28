from transformers import AutoTokenizer, pipeline
# You can remove ConfigurationManager if you no longer need local artifact paths!
# from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        # 1. Point directly to your fine-tuned model on Hugging Face Hub
        # Swap 'your-username/pegasus-samsum-summarizer' with your actual HF repo ID
        self.model_id = "your-username/pegasus-samsum-summarizer"
        
        print(f"Loading model '{self.model_id}' into memory...")
        
        # 2. Initialize tokenizer and pipeline ONCE during startup
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "summarization", 
            model=self.model_id, 
            tokenizer=self.tokenizer
        )
        print("Model loaded successfully!")

    def predict(self, text):
        # Decoding rules for Pegasus summarization
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        print("--- Dialogue ---")
        print(text)

        # 3. Generate summary using the pre-loaded pipeline
        output = self.pipe(text, **gen_kwargs)[0]["summary_text"]
        
        print("\n--- Model Summary ---")
        print(output)

        return output