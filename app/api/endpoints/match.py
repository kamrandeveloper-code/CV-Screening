from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.match_service import MatchService
from app.schemas.match_schema import MatchRequest, MatchResponse
from typing import List

router = APIRouter(prefix="/match", tags=["Matching"])

@router.post("/", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
def create_match(
    match_request: MatchRequest,
    db: Session = Depends(get_db)
):
    """Create a match between candidate and job"""
    match = MatchService.create_match(
        db=db,
        candidate_id=match_request.candidate_id,
        job_id=match_request.job_id
    )
    
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate or Job not found"
        )
    
    return match

@router.post("/job/{job_id}/batch", status_code=status.HTTP_200_OK)
def batch_match_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Match all candidates against a job (background task)"""
    background_tasks.add_task(MatchService.batch_match_job, db, job_id)
    return {
        "message": "Batch matching started in background",
        "job_id": job_id
    }

@router.get("/{match_id}", response_model=MatchResponse)
def get_match(match_id: str, db: Session = Depends(get_db)):
    """Get match details"""
    match = MatchService.get_match(db, match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )
    return match

@router.get("/candidate/{candidate_id}", response_model=List[MatchResponse])
def get_matches_for_candidate(
    candidate_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all matches for a candidate"""
    return MatchService.get_matches_for_candidate(db, candidate_id, skip, limit)

@router.get("/job/{job_id}", response_model=List[MatchResponse])
def get_matches_for_job(
    job_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all matches for a job (sorted by score)"""
    return MatchService.get_matches_for_job(db, job_id, skip, limit)

@router.get("/job/{job_id}/top", response_model=List[MatchResponse])
def get_top_candidates(
    job_id: str,
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """Get top N candidates for a job"""
    return MatchService.get_best_matches_for_job(db, job_id, top_n)

@router.get("/job/{job_id}/ready", response_model=List[MatchResponse])
def get_ready_candidates(
    job_id: str,
    min_score: float = 60.0,
    db: Session = Depends(get_db)
):
    """Get candidates ready for interview"""
    return MatchService.get_ready_candidates_for_job(db, job_id, min_score)

@router.delete("/{match_id}", status_code=status.HTTP_200_OK)
def delete_match(match_id: str, db: Session = Depends(get_db)):
    """Delete match"""
    success = MatchService.delete_match(db, match_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )
    return {"message": "Match deleted successfully"}

@router.post("/job/{job_id}/recalculate", status_code=status.HTTP_200_OK)
def recalculate_matches(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Recalculate all matches for a job"""
    count = MatchService.recalculate_all_matches_for_job(db, job_id)
    return {
        "message": f"Recalculated matches for job {job_id}",
        "matches_created": count
    }