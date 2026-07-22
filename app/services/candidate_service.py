"""Candidate service layer"""
import json
import logging
from typing import Optional, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from app.db.models import Candidate, Match
from app.models.resume_parser import get_resume_parser
from app.models.skill_extractor import skill_extractor
from app.models.embedding_model import get_embedding_model
from app.utils.text_cleaner import TextCleaner
from app.utils.id_generator import generate_candidate_id
from app.utils.deduplicator import deduplicator

logger = logging.getLogger(__name__)

class CandidateService:
    @staticmethod
    def create_or_update_candidate(
        db: Session,
        raw_text: str,
        candidate_id: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        sections: Optional[Dict] = None
    ) -> Tuple[Candidate, bool]:
        parser = get_resume_parser()
        
        logger.info("Extracting information using pretrained models...")
        extracted = parser.extract_all(raw_text, sections)
        
        if not name:
            name = extracted.get("name")
        if not email:
            email = extracted.get("email")
        if not phone:
            phone = extracted.get("phone")
        
        if candidate_id:
            existing = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
            if existing:
                return CandidateService._update_candidate(
                    db, candidate_id, raw_text, name, email, phone, sections
                ), False
            final_id = candidate_id
        else:
            existing_id = deduplicator.find_duplicate(db, name, email, phone)
            if existing_id:
                return CandidateService._update_candidate(
                    db, existing_id, raw_text, name, email, phone, sections
                ), False
            final_id = generate_candidate_id()
        
        return CandidateService._create_new_candidate(
            db, final_id, raw_text, name, email, phone, sections
        ), True
    
    @staticmethod
    def _create_new_candidate(
        db: Session, 
        candidate_id: str, 
        raw_text: str, 
        name: Optional[str], 
        email: Optional[str], 
        phone: Optional[str],
        sections: Optional[Dict] = None
    ) -> Candidate:
        parser = get_resume_parser()
        embedder = get_embedding_model()
        
        skills_text = sections.get("skills", raw_text) if sections else raw_text
        cleaned = TextCleaner.clean_text(skills_text)
        skills = skill_extractor.extract_skills(cleaned)
        
        edu_text = sections.get("education", raw_text) if sections else raw_text
        exp_text = sections.get("experience", raw_text) if sections else raw_text
        
        education = parser.extract_education(edu_text)
        experience = parser.extract_experience(exp_text)
        
        embedding_vec = embedder.encode(TextCleaner.clean_text(raw_text))
        
        candidate = Candidate(
            candidate_id=candidate_id,
            name=name,
            email=email,
            phone=phone,
            raw_text=raw_text,
            education=json.dumps(education) if education else None,
            experience=json.dumps(experience) if experience else None
        )
        candidate.set_skills(skills)
        candidate.set_embedding(embedding_vec)
        
        try:
            db.add(candidate)
            db.commit()
            db.refresh(candidate)
            logger.info(f"Created candidate: {candidate_id} | Name: {name} | Skills: {len(skills)}")
            return candidate
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Candidate ID {candidate_id} already exists")
    
    @staticmethod
    def _update_candidate(db, cid, raw_text, name, email, phone, sections: Optional[Dict] = None):
        parser = get_resume_parser()
        embedder = get_embedding_model()
        
        candidate = db.query(Candidate).filter(Candidate.candidate_id == cid).first()
        if not candidate:
            raise ValueError("Candidate not found")
        
        candidate.raw_text = raw_text
        if name and (not candidate.name or len(name) > len(candidate.name or "")):
            candidate.name = name
        if email and not candidate.email:
            candidate.email = email
        if phone and not candidate.phone:
            candidate.phone = phone
        
        skills_text = sections.get("skills", raw_text) if sections else raw_text
        cleaned = TextCleaner.clean_text(skills_text)
        skills = skill_extractor.extract_skills(cleaned)
        candidate.set_skills(skills)
        
        embedding_vec = embedder.encode(TextCleaner.clean_text(raw_text))
        candidate.set_embedding(embedding_vec)
        
        edu_text = sections.get("education", raw_text) if sections else raw_text
        education = parser.extract_education(edu_text)
        candidate.education = json.dumps(education) if education else None
        
        db.commit()
        db.refresh(candidate)
        logger.info(f"Updated candidate: {cid}")
        return candidate
    
    @staticmethod
    def get_candidate(db: Session, candidate_id: str):
        return db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
    
    @staticmethod
    def get_all_candidates(db: Session, skip: int = 0, limit: int = 100):
        return db.query(Candidate).order_by(Candidate.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def delete_candidate(db: Session, candidate_id: str) -> bool:
        candidate = CandidateService.get_candidate(db, candidate_id)
        if not candidate:
            return False
        try:
            db.query(Match).filter(Match.candidate_id == candidate_id).delete()
            db.delete(candidate)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            db.rollback()
            return False
        
    @staticmethod
    def to_response(candidate: Candidate):
        return {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "email": candidate.email,
            "phone": candidate.phone,
            "skills": candidate.get_skills(),
            "education": json.loads(candidate.education) if candidate.education else [],
            "experience": json.loads(candidate.experience) if candidate.experience else [],
            "created_at": candidate.created_at.isoformat() if candidate.created_at else None
        }