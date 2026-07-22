from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.candidate_service import CandidateService
from app.schemas.candidate_schema import CandidateDetail, CandidateResponse
from app.utils.file_handler import FileHandler
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/candidate", tags=["Candidates"])

@router.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    candidate_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        raw_text = FileHandler.extract_text(contents, file.filename)
        
        if not raw_text or len(raw_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Resume text too short or empty")
        
        # LayoutLMv3: Analyze document layout for PDFs
        sections = None
        if file.filename.lower().endswith('.pdf'):
            try:
                from app.models.document_layout import get_layout_analyzer
                analyzer = get_layout_analyzer()
                layout_result = analyzer.extract_layout_and_sections(contents)
                sections = layout_result.get("sections")
                # Use layout-extracted text if pdfplumber layout extraction was richer
                if layout_result.get("text") and len(layout_result["text"]) > len(raw_text) * 0.5:
                    raw_text = layout_result["text"]
            except Exception as e:
                logger.warning(f"LayoutLMv3 analysis skipped: {e}")
        
        candidate, is_new = CandidateService.create_or_update_candidate(
            db=db,
            raw_text=raw_text,
            candidate_id=candidate_id,
            sections=sections
        )
        
        return {
            "success": True,
            "candidate": CandidateService.to_response(candidate),
            "is_new": is_new
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{candidate_id}", response_model=CandidateDetail)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    candidate = CandidateService.get_candidate(db, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateService.to_response(candidate)

@router.get("/", response_model=List[CandidateResponse])
def get_all_candidates(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return CandidateService.get_all_candidates(db, skip, limit)

@router.delete("/{candidate_id}")
def delete_candidate(candidate_id: str, db: Session = Depends(get_db)):
    success = CandidateService.delete_candidate(db, candidate_id)
    if not success:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"message": f"Candidate {candidate_id} deleted successfully"}