from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.job_service import JobService
from app.schemas.job_schema import JobCreate, JobResponse
from typing import List

router = APIRouter(prefix="/job", tags=["Jobs"])

@router.post("/create", response_model=JobResponse)
def create_job(job: JobCreate, db: Session = Depends(get_db)):
    """
    Create job posting.
    If job_id provided in request, uses it (must be unique).
    If not provided, auto-generates job_id.
    """
    try:
        new_job = JobService.create_job(
            db=db,
            title=job.title,
            description=job.description,
            company=job.company,
            skills_required=job.skills_required,
            job_id=job.job_id  # Pass custom ID (None if not provided)
        )
        return new_job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = JobService.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/", response_model=List[JobResponse])
def get_all_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return JobService.get_all_jobs(db, skip, limit)

@router.delete("/{job_id}")
def delete_job(job_id: str, db: Session = Depends(get_db)):
    success = JobService.delete_job(db, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}




@router.put("/update/{job_id}", response_model=JobResponse)
def update_job(job_id: str, job: JobCreate, db: Session = Depends(get_db)):
    print(job.dict())  # DEBUG

    try:
        updated_job = JobService.update_job(
            db=db,
            job_id=job_id,
            title=job.title,
            description=job.description,
            company=job.company,
            skills_required=job.skills_required
        )

        if not updated_job:
            raise HTTPException(status_code=404, detail="Job not found")

        return updated_job

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))