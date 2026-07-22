"""Document layout analyzer using LayoutLMv3 (optional)"""
import io
import logging
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SECTION_KEYWORDS = {
    "experience": ["experience", "work experience", "employment history", "professional experience", "career history"],
    "education": ["education", "academic background", "qualifications", "academic history", "degrees"],
    "skills": ["skills", "technical skills", "core competencies", "expertise", "technologies", "proficiencies"],
    "projects": ["projects", "personal projects", "academic projects", "portfolio"],
    "summary": ["summary", "professional summary", "profile", "about me", "objective"],
    "contact": ["contact", "contact information", "personal information", "info"]
}

class DocumentLayoutAnalyzer:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentLayoutAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if DocumentLayoutAnalyzer._initialized:
            return
        
        self.processor = None
        self.model = None
        self._section_embeddings = None
        
        try:
            from transformers import LayoutLMv3Processor, LayoutLMv3Model
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading LayoutLMv3 on {device}...")
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self.model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)
            self.model.eval()
            logger.info("[OK] LayoutLMv3 loaded")
        except Exception as e:
            logger.warning(f"LayoutLMv3 not available: {e}")
            self.processor = None
            self.model = None
        
        DocumentLayoutAnalyzer._initialized = True
    
    def _get_section_embeddings(self):
        if self._section_embeddings is None:
            from app.models.embedding_model import get_embedding_model
            emb = get_embedding_model()
            self._section_embeddings = {}
            for section, keywords in SECTION_KEYWORDS.items():
                kw_embs = emb.encode_batch(keywords)
                self._section_embeddings[section] = np.mean(kw_embs, axis=0)
        return self._section_embeddings
    
    def extract_layout_and_sections(self, pdf_bytes: bytes) -> Dict:
        if self.model is None or self.processor is None:
            # Return basic text extraction without layout analysis
            layout_info = self._extract_pdfplumber_layout(pdf_bytes)
            if layout_info:
                return {"sections": {}, "text": layout_info.get("full_text", ""), "layout_embedding": None}
            return {"sections": {}, "text": "", "layout_embedding": None}
        
        layout_info = self._extract_pdfplumber_layout(pdf_bytes)
        if not layout_info:
            return {"sections": {}, "text": "", "layout_embedding": None}
        
        try:
            import pdf2image
            images = pdf2image.convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
            if not images:
                return {"sections": {}, "text": layout_info["full_text"], "layout_embedding": None}
            image = images[0]
        except Exception as e:
            logger.warning(f"pdf2image failed: {e}")
            return {"sections": {}, "text": layout_info["full_text"], "layout_embedding": None}
        
        layout_embedding = self._encode_with_layoutlmv3(image, layout_info["words"], layout_info["bboxes"])
        sections = self._segment_sections(layout_info)
        
        return {
            "sections": sections,
            "text": layout_info["full_text"],
            "layout_embedding": layout_embedding,
            "page_dims": (layout_info["width"], layout_info["height"])
        }
    
    def _extract_pdfplumber_layout(self, pdf_bytes: bytes) -> Optional[Dict]:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if not pdf.pages:
                    return None
                page = pdf.pages[0]
                width, height = page.width, page.height
                
                words = []
                bboxes = []
                for word in page.extract_words():
                    words.append(word["text"])
                    bboxes.append([
                        int(word["x0"] / width * 1000),
                        int(word["top"] / height * 1000),
                        int(word["x1"] / width * 1000),
                        int(word["bottom"] / height * 1000)
                    ])
                
                full_text = page.extract_text(layout=True) or page.extract_text() or ""
                
                chars = page.chars
                line_data = {}
                for char in chars:
                    line_key = round(char["top"])
                    if line_key not in line_data:
                        line_data[line_key] = {"sizes": [], "text": [], "fontnames": []}
                    line_data[line_key]["sizes"].append(char["size"])
                    line_data[line_key]["text"].append(char["text"])
                    line_data[line_key]["fontnames"].append(char["fontname"])
                
                structured_lines = []
                for top, data in sorted(line_data.items()):
                    text = "".join(data["text"]).strip()
                    if text:
                        avg_size = np.mean(data["sizes"]) if data["sizes"] else 12
                        is_bold = any("Bold" in f or "bold" in f.lower() for f in data["fontnames"])
                        structured_lines.append({
                            "text": text,
                            "top": top,
                            "size": avg_size,
                            "is_bold": is_bold
                        })
                
                return {
                    "words": words,
                    "bboxes": bboxes,
                    "full_text": full_text,
                    "structured_lines": structured_lines,
                    "width": width,
                    "height": height
                }
        except Exception as e:
            logger.error(f"pdfplumber layout extraction error: {e}")
            return None
    
    def _encode_with_layoutlmv3(self, image, words, bboxes):
        if not words or self.processor is None or self.model is None:
            return None
        try:
            import torch
            encoding = self.processor(
                image,
                text=words,
                boxes=bboxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            device = next(self.model.parameters()).device
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor):
                    encoding[k] = v.to(device)
            
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        except Exception as e:
            logger.error(f"LayoutLMv3 encoding error: {e}")
            return None
    
    def _segment_sections(self, layout_info: Dict) -> Dict[str, str]:
        lines = layout_info.get("structured_lines", [])
        full_text = layout_info.get("full_text", "")
        
        if not lines:
            return {"full": full_text}
        
        avg_size = np.mean([l["size"] for l in lines]) if lines else 12
        header_candidates = [(i, line) for i, line in enumerate(lines) 
                            if line["size"] > avg_size + 0.5 or line["is_bold"]]
        
        section_boundaries = {}
        from app.models.embedding_model import get_embedding_model
        emb = get_embedding_model()
        section_embs = self._get_section_embeddings()
        
        for idx, line in header_candidates:
            line_emb = emb.encode(line["text"].lower())
            best_section, best_score = None, -1
            
            for section, sec_emb in section_embs.items():
                # Reshape for cosine_similarity
                sim = float(cosine_similarity([line_emb], [sec_emb])[0][0])
                if sim > best_score:
                    best_score = sim
                    best_section = section
            
            if best_score > 0.55:
                section_boundaries[idx] = best_section
        
        sections = {}
        sorted_bounds = sorted(section_boundaries.items())
        
        for i, (start_idx, section_name) in enumerate(sorted_bounds):
            end_idx = sorted_bounds[i + 1][0] if i + 1 < len(sorted_bounds) else len(lines)
            section_lines = [lines[j]["text"] for j in range(start_idx + 1, end_idx)]
            sections[section_name] = "\n".join(section_lines).strip()
        
        if sorted_bounds:
            first_idx = sorted_bounds[0][0]
            header_lines = [lines[j]["text"] for j in range(0, first_idx)]
            if header_lines:
                sections["header"] = "\n".join(header_lines).strip()
        else:
            sections["header"] = full_text
        
        return sections

_layout_analyzer = None

def get_layout_analyzer():
    global _layout_analyzer
    if _layout_analyzer is None:
        _layout_analyzer = DocumentLayoutAnalyzer()
    return _layout_analyzer