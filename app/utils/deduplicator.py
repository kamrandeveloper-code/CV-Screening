import re
from typing import Optional
from difflib import SequenceMatcher
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.db.models import Candidate
import logging

logger = logging.getLogger(__name__)

class CandidateDeduplicator:
    """Efficient database-level deduplication"""
    
    @staticmethod
    def normalize_phone(phone: str) -> str:
        """Remove all non-digits from phone"""
        if not phone:
            return ""
        return re.sub(r'\D', '', phone)
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name for comparison"""
        if not name:
            return ""
        return ' '.join(name.lower().split())
    
    @staticmethod
    def similarity_ratio(a: str, b: str) -> float:
        """Calculate string similarity (0-1)"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    @classmethod
    def find_duplicate(
        cls,
        db: Session,
        name: Optional[str],
        email: Optional[str],
        phone: Optional[str]
    ) -> Optional[str]:
        """
        Find if candidate already exists using database queries.
        Returns candidate_id if found, None otherwise.
        """
        try:
            # Priority 1: Email match (exact)
            if email:
                existing = db.query(Candidate).filter(
                    Candidate.email.ilike(email.strip())
                ).first()
                if existing:
                    logger.info(f"Duplicate found by email: {email}")
                    return existing.candidate_id
            
            # Priority 2: Phone match (normalized)
            if phone:
                normalized_phone = cls.normalize_phone(phone)
                if normalized_phone and len(normalized_phone) >= 10:
                    # Query all candidates with phone numbers
                    candidates = db.query(Candidate).filter(
                        Candidate.phone.isnot(None)
                    ).all()
                    
                    for candidate in candidates:
                        if candidate.phone:
                            existing_phone = cls.normalize_phone(candidate.phone)
                            if existing_phone == normalized_phone:
                                logger.info(f"Duplicate found by phone: {phone}")
                                return candidate.candidate_id
            
            # Priority 3: Name similarity (only if no email/phone)
            if name and len(name) > 3:
                normalized_name = cls.normalize_name(name)
                # Query candidates with similar names using LIKE for pre-filtering
                name_filter = f"%{normalized_name[:4]}%"  # First 4 chars
                candidates = db.query(Candidate).filter(
                    Candidate.name.ilike(name_filter)
                ).limit(10).all()  # Limit to avoid heavy computation
                
                for candidate in candidates:
                    if candidate.name:
                        existing_name = cls.normalize_name(candidate.name)
                        similarity = cls.similarity_ratio(normalized_name, existing_name)
                        if similarity >= 0.9:
                            logger.info(f"Duplicate found by name similarity: {name} ~ {candidate.name}")
                            return candidate.candidate_id
            
            return None
            
        except Exception as e:
            logger.error(f"Deduplication error: {e}")
            return None

deduplicator = CandidateDeduplicator()