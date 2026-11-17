import os
import warnings

# 1. Suppress the Mac multiprocessing warning
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# 2. Stop HuggingFace and PyTorch from spawning too many threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1" 

from docling.document_converter import DocumentConverter

class DocumentLoader:
    def __init__(self):
        print("Initializing Docling models...")
        self.converter = DocumentConverter()

    def process(self, file_path: str):
        print(f"Loading and parsing: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find the file: {file_path}")

        # This is where it was crashing previously
        result = self.converter.convert(file_path)
        
        markdown_text = result.document.export_to_markdown()
        print(f"Successfully extracted {len(markdown_text)} characters.")
        return markdown_text

if __name__ == "__main__":
    loader = DocumentLoader()
    content = loader.process("2024AC05737.pdf")
    print("\n--- Document Preview ---")
    print(content[:500])