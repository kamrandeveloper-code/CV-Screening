"""Job service layer"""
import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from app.db.models import Job, Match
from app.models.skill_extractor import skill_extractor
from app.models.embedding_model import get_embedding_model
from app.utils.text_cleaner import TextCleaner
from app.utils.id_generator import generate_job_id

logger = logging.getLogger(__name__)

class JobService:
    @staticmethod
    def create_job(
        db: Session,
        title: str,
        description: str,
        company: Optional[str] = None,
        skills_required: Optional[List[str]] = None,
        job_id: Optional[str] = None
    ) -> Job:
        if job_id:
            existing = db.query(Job).filter(Job.job_id == job_id).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Job ID {job_id} already exists")
            final_id = job_id
        else:
            final_id = generate_job_id()
        
        cleaned = TextCleaner.clean_text(description)
        
        if not skills_required:
            skills = skill_extractor.extract_skills(cleaned)
        else:
            skills = skills_required
        
        embedder = get_embedding_model()
        embedding_vec = embedder.encode(cleaned)
        
        job = Job(
            job_id=final_id,
            title=title,
            description=description,
            company=company
        )
        job.set_skills(skills)
        job.set_embedding(embedding_vec)
        
        try:
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"Created job with ID: {final_id}")
            return job
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Job ID {final_id} already exists")
    
    @staticmethod
    def get_job(db: Session, job_id: str):
        return db.query(Job).filter(Job.job_id == job_id).first()
    
    @staticmethod
    def get_all_jobs(db: Session, skip: int = 0, limit: int = 100):
        return db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()
    
    # @staticmethod
    # def update_job(
    #     db: Session,
    #     job_id: str,
    #     title: Optional[str] = None,
    #     description: Optional[str] = None,
    #     company: Optional[str] = None,
    #     skills_required: Optional[List[str]] = None
    # ):
    #     job = JobService.get_job(db, job_id)
    #     if not job:
    #         return None
        
    #     if title:
    #         job.title = title
    #     if description:
    #         job.description = description
    #         if not skills_required:
    #             cleaned = TextCleaner.clean_text(description)
    #             skills = skill_extractor.extract_skills(cleaned)
    #             job.set_skills(skills)
    #             embedder = get_embedding_model()
    #             job.set_embedding(embedder.encode(cleaned))
    #     if company:
    #         job.company = company
    #     if skills_required:
    #         job.set_skills(skills_required)
        
    #     db.commit()
    #     db.refresh(job)
    #     return job
    


    @staticmethod
    def update_job(
        db: Session,
        job_id: str,
        title: str,
        description: str,
        company: str,
        skills_required: list
    ):
        job = db.query(Job).filter(Job.job_id == job_id).first()

        if not job:
            return None

        job.title = title
        job.description = description
        job.company = company

        # Save skills correctly
        job.set_skills(skills_required)

        # Update embedding
        cleaned = TextCleaner.clean_text(description)
        embedder = get_embedding_model()
        job.set_embedding(embedder.encode(cleaned))

        db.commit()
        db.refresh(job)

        return job

            
    @staticmethod
    def delete_job(db: Session, job_id: str) -> bool:
            job = JobService.get_job(db, job_id)
            if not job:
                return False
            try:
                db.query(Match).filter(Match.job_id == job_id).delete()
                db.delete(job)
                db.commit()
                return True
            except:
                db.rollback()
                return False