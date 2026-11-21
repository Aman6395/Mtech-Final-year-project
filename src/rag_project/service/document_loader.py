# import os
# import warnings

# # 1. Suppress the Mac multiprocessing warning
# warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# # 2. Stop HuggingFace and PyTorch from spawning too many threads
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1" 

# from docling.document_converter import DocumentConverter

# class DocumentLoader:
#     def __init__(self):
#         print("Initializing Docling models...")
#         self.converter = DocumentConverter()

#     def process(self, file_path: str):
#         print(f"Loading and parsing: {file_path}")
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Could not find the file: {file_path}")

#         # This is where it was crashing previously
#         result = self.converter.convert(file_path)
        
#         markdown_text = result.document.export_to_markdown()
#         print(f"Successfully extracted {len(markdown_text)} characters.")
#         return markdown_text

# if __name__ == "__main__":
#     loader = DocumentLoader()
#     content = loader.process("2024AC05737.pdf")
#     print("\n--- Document Preview ---")
#     print(content[:500])



import json
import time
import logging
from pathlib import Path


from docling.chunking import HybridChunker

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(
        self,
        chunk_size: int = 500,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        merge_peers: bool = True,
    ):
        pipeline_options = PdfPipelineOptions()

        pipeline_options.accelerator_options = AcceleratorOptions(
            device="cpu",
            num_threads=8,
        )

        # Keep OCR for scanned pages
        pipeline_options.do_ocr = True

        # Faster table extraction
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )



        # Docling's native HybridChunker
        self.chunker = HybridChunker(
            tokenizer=tokenizer_model,
            max_tokens=chunk_size,
            merge_peers=merge_peers,
        )

    def _convert(self, file_path: str | Path):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, received: {path}")

        result = self.converter.convert(str(path))

        return result.document

    def load(self, file_path: str | Path) -> str:
        """Parse a document and return Markdown."""

        try:
            document = self._convert(file_path)
            markdown = document.export_to_markdown()

            if not markdown.strip():
                raise ValueError("No text extracted from document.")

            logger.info("Document parsed: %s", file_path)
            return markdown

        except Exception:
            logger.exception("Document parsing failed: %s", file_path)
            raise

    def process(
        self,
        file_path: str | Path,
        output_dir: str | Path = "data/processed",
    ) -> dict:
        """Parse, chunk and save a document."""

        path = Path(file_path)

        try:
            # # 1. Convert document
            # document = self._convert(path)

            # # 2. Export Markdown
            # markdown = document.export_to_markdown()

            # if not markdown.strip():
            #     raise ValueError("No text extracted from document.")

            # # 3. Generate HybridChunks
            # doc_chunks = list(self.chunker.chunk(document))
            
            # 1. PDF conversion
            start = time.perf_counter()
            document = self._convert(path)
            conversion_time = time.perf_counter() - start

            # 2. Markdown export
            start = time.perf_counter()
            markdown = document.export_to_markdown()
            markdown_time = time.perf_counter() - start

            if not markdown.strip():
                raise ValueError("No text extracted from document.")

            # 3. Hybrid chunking
            start = time.perf_counter()
            doc_chunks = list(self.chunker.chunk(document))
            chunking_time = time.perf_counter() - start

            print("\n--- Document Processing Benchmark ---")
            print(f"PDF conversion : {conversion_time:.2f} sec")
            print(f"Markdown export: {markdown_time:.2f} sec")
            print(f"Hybrid chunking: {chunking_time:.2f} sec")
            print(f"Total time     : {conversion_time + markdown_time + chunking_time:.2f} sec")
            print("--------------------------------------\n")


            # 4. Create output directory
            document_dir = Path(output_dir) / path.stem
            document_dir.mkdir(parents=True, exist_ok=True)

            # 5. Save parsed Markdown
            markdown_path = document_dir / "parsed.md"
            markdown_path.write_text(markdown, encoding="utf-8")

            # 6. Prepare chunk data
            chunks_data = []

            for index, chunk in enumerate(doc_chunks):
                metadata = chunk.meta

                origin = getattr(metadata, "origin", None)
                page_no = getattr(origin, "page_no", None)

                page_numbers = sorted({
                        prov.page_no
                        for doc_item in (metadata.doc_items or [])
                        for prov in (getattr(doc_item, "prov", []) or [])
                        if getattr(prov, "page_no", None) is not None
                    })

                chunks_data.append({
                    "chunk_index": index,
                    "text": self.chunker.contextualize(chunk),
                    "raw_text": chunk.text,
                    "headings": metadata.headings or [],
                    "page_no": page_no,
                })

            # 7. Save chunks
            chunks_path = document_dir / "chunks.json"

            output = {
                "source": path.name,
                "total_chunks": len(chunks_data),
                "chunker": "Docling HybridChunker",
                "chunks": chunks_data,
            }

            chunks_path.write_text(
                json.dumps(output, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(
                "Processed %s: %d chunks generated",
                path.name,
                len(chunks_data),
            )

            return {
                "source": path.name,
                "total_chunks": len(chunks_data),
                "markdown_path": str(markdown_path),
                "chunks_path": str(chunks_path),
            }

        except Exception:
            logger.exception("Document processing failed: %s", path)
            raise


