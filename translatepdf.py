# code from:
# https://www.youtube.com/watch?v=YAgjtZKLVKo&t=575s
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

def create_picture_description_options() -> PictureDescriptionApiOptions:
    OLLAMA_URL = "http://localhost:11434"
    VLM_MODEL = "gemma3:4b-it-q4_K_M"
    VLM_PROMPT = "Describe this image in one short sentence. If an action is performed, focus only on that action. " \
                "Output only the description, no conversational filler."
    return PictureDescriptionApiOptions(
        url=f"{OLLAMA_URL}/v1/chat/completions",
        params=dict[str, Any](
            model=VLM_MODEL,
            think=False,
            seed=42,
            max_completion_tokens=256,
        ),
        prompt=VLM_PROMPT,
        timeout=90,
    )

def create_pdf_pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        enable_remote_services=True,
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        do_picture_description=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
        ),
        picture_description_options=create_picture_description_options(),
    )

def process_document(pdf_path:str):
    converter = DocumentConverter(
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=create_pdf_pipeline_options(),
                backend=PyPdfiumDocumentBackend
            )
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    content = doc.export_to_markdown(
        image_mode = ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        page_break_placeholder="",
        include_annotations=True,
        mark_annotations=True,
    )
    return content

# if __name__ == "__main__":
#     md_content = process_document(f"test.pdf")
#     print(md_content)
