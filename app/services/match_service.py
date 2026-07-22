"""Match service layer with hybrid scoring"""
import logging
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from app.db.models import Match, Candidate, Job
from app.models.matcher import get_matcher
from app.utils.id_generator import generate_match_id

logger = logging.getLogger(__name__)

class MatchService:
    @staticmethod
    def create_match(db: Session, candidate_id: str, job_id: str) -> Optional[Match]:
        matcher = get_matcher()
        
        candidate = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not candidate or not job:
            logger.warning(f"Match failed: Candidate {candidate_id} or Job {job_id} not found")
            return None
        
        try:
            existing = db.query(Match).filter(
                Match.candidate_id == candidate_id,
                Match.job_id == job_id
            ).first()
            
            if existing:
                return MatchService._update_match(db, existing, candidate, job, matcher)
            
            return MatchService._create_new_match(db, candidate, job, matcher)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Match creation error: {e}")
            raise
    
    @staticmethod
    def _create_new_match(db: Session, candidate: Candidate, job: Job, matcher) -> Match:
        result = matcher.compute_hybrid_score(
            candidate.get_embedding(),
            job.get_embedding(),
            candidate.get_skills(),
            job.get_skills()
        )
        
        match = Match(
            match_id=generate_match_id(),
            candidate_id=candidate.candidate_id,
            job_id=job.job_id,
            similarity_score=result["similarity_score"],
            embedding_score=result["embedding_score"],
            skill_score=result["skill_score"],
            ready_for_interview=result["ready_for_interview"],
            status=result["status"]
        )
        match.set_matched_skills(result["matched_skills"])
        match.set_missing_skills(result["missing_skills"])
        
        db.add(match)
        db.commit()
        db.refresh(match)
        logger.info(f"Created match: {match.match_id} (Score: {result['similarity_score']}, Status: {result['status']})")
        return match
    
    @staticmethod
    def _update_match(db: Session, match: Match, candidate: Candidate, job: Job, matcher) -> Match:
        result = matcher.compute_hybrid_score(
            candidate.get_embedding(),
            job.get_embedding(),
            candidate.get_skills(),
            job.get_skills()
        )
        
        match.similarity_score = result["similarity_score"]
        match.embedding_score = result["embedding_score"]
        match.skill_score = result["skill_score"]
        match.ready_for_interview = result["ready_for_interview"]
        match.status = result["status"]
        match.set_matched_skills(result["matched_skills"])
        match.set_missing_skills(result["missing_skills"])
        
        db.commit()
        db.refresh(match)
        logger.info(f"Updated match: {match.match_id} (Score: {result['similarity_score']}, Status: {result['status']})")
        return match
    
    @staticmethod
    def batch_match_job(db: Session, job_id: str) -> Dict[str, dict]:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return {}
        
        candidates = db.query(Candidate).all()
        results = {}
        
        for candidate in candidates:
            match_result = MatchService.create_match(db, candidate.candidate_id, job_id)
            if match_result:
                results[candidate.candidate_id] = {
                    "match_id": match_result.match_id,
                    "score": match_result.similarity_score,
                    "status": match_result.status,
                    "ready": match_result.ready_for_interview
                }
        
        return results
    
    @staticmethod
    def get_match(db: Session, match_id: str) -> Optional[Match]:
        return db.query(Match).filter(Match.match_id == match_id).first()
    
    @staticmethod
    def get_matches_for_candidate(db: Session, candidate_id: str, skip: int = 0, limit: int = 100):
        return db.query(Match).filter(
            Match.candidate_id == candidate_id
        ).order_by(Match.similarity_score.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_matches_for_job(db: Session, job_id: str, skip: int = 0, limit: int = 100):
        return db.query(Match).filter(
            Match.job_id == job_id
        ).order_by(Match.similarity_score.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_best_matches_for_job(db: Session, job_id: str, top_n: int = 5):
        return db.query(Match).filter(
            Match.job_id == job_id
        ).order_by(Match.similarity_score.desc()).limit(top_n).all()
    
    @staticmethod
    def get_ready_candidates_for_job(db: Session, job_id: str, min_score: float = 60.0):
        return db.query(Match).filter(
            Match.job_id == job_id,
            Match.similarity_score >= min_score
        ).order_by(Match.similarity_score.desc()).all()
    
    @staticmethod
    def delete_match(db: Session, match_id: str) -> bool:
        match = MatchService.get_match(db, match_id)
        if match:
            db.delete(match)
            db.commit()
            return True
        return False
    
    @staticmethod
    def recalculate_all_matches_for_job(db: Session, job_id: str) -> int:
        count = db.query(Match).filter(Match.job_id == job_id).delete()
        candidates = db.query(Candidate).all()
        created = 0
        
        for candidate in candidates:
            if MatchService.create_match(db, candidate.candidate_id, job_id):
                created += 1
        
        db.commit()
        logger.info(f"Recalculated {created} matches for job {job_id} (deleted {count} old)")
        return created