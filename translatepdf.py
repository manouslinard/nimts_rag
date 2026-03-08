import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
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
from docling_core.types.doc import ImageRefMode, DocItemLabel

def create_picture_description_options() -> PictureDescriptionApiOptions:
    OLLAMA_URL = "http://localhost:11434"
    VLM_MODEL = "gemma3:4b-it-q4_K_M"
    VLM_PROMPT = """You are analyzing a screenshot from a software tutorial.
                Describe what is being shown or done in one concise sentence.
                Focus on: what software/window is open, what setting or action is highlighted,
                and any notable configuration visible. Do not describe visual style or layout.
                Write max 2 sentences. Be concise, no conversation filler."""
    return PictureDescriptionApiOptions(
        url=f"{OLLAMA_URL}/v1/chat/completions",
        params=dict[str, Any](
            model=VLM_MODEL,
            think=False,
            seed=42,
            max_completion_tokens=512,
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

def build_description_map(doc) -> dict[str, str]:
    """Build a map of picture reference -> LLM description from document items."""
    desc_map = {}
    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.PICTURE:
            # Get the self_ref (e.g. "#/pictures/0") as the key
            ref = item.self_ref
            if item.meta:
                for ann in item.meta:
                    if hasattr(ann, 'text') and ann.text:
                        desc_map[ref] = ann.text
                        break
    return desc_map

def process_document(pdf_path: str) -> str:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=create_pdf_pipeline_options(),
                backend=PyPdfiumDocumentBackend
            )
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document

    desc_map = build_description_map(doc)

    picture_items = []
    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.PICTURE:
            ref = item.self_ref
            desc = ""
            if item.meta:
                for ann in item.meta:
                    if hasattr(ann, 'text') and ann.text:
                        desc = ann.text
                        break
            picture_items.append(desc)

    PLACEHOLDER = "IMAGE_PLACEHOLDER_{index}"

    content = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="<<<IMAGE_HERE>>>",
        page_break_placeholder="",
        include_annotations=False,  # Disable so they don't get appended at the bottom
        mark_annotations=False,
    )

    idx = 0
    def replace_next(match):
        nonlocal idx
        if idx < len(picture_items):
            desc = picture_items[idx]
            idx += 1
            if desc:
                return f"*[Image: {desc}]*"
            return ""  # No description available
        return ""

    content = re.sub(r'<<<IMAGE_HERE>>>', replace_next, content)

    # Clean up any leftover annotation blocks that may still appear (safety net)
    content = re.sub(r'\n{3,}', '\n\n', content).strip()

    return content

# if __name__ == "__main__":
#     md_content = process_document(r"data_pdf\TUTORIALS\Medico - LIS - Δίαιτα - ΜΙΣΘΟΔΟΣΙΑ\Update Medico License.pdf")
#     print(md_content)
