import PyPDF2
import io
from fastapi import HTTPException


def extract_text(file_bytes, url_str):
    extracted_text = ""
    if url_str.endswith('.pdf'):
        # PDF extraction
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
                
    elif url_str.endswith('.docx'):
        # DOCX extraction
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        extracted_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
    elif url_str.endswith('.txt'):
        # TXT extraction - direct text
        extracted_text = file_bytes.decode('utf-8')
        
    else:
        extracted_text = None
    
    return extracted_text