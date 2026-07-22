"""Hybrid matcher: 70% embedding similarity + 30% skill overlap"""
import re
import logging
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from app.models.embedding_model import get_embedding_model
from app.core.config import settings

logger = logging.getLogger(__name__)

class SkillMatcher:
    def __init__(self):
        self.embedding_model = get_embedding_model()
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill string for better matching"""
        if not skill:
            return ""
        
        skill = skill.lower().strip()
        # Remove version numbers
        skill = re.sub(r'\d+\.\d+', '', skill)
        # Remove special characters
        skill = skill.replace(".js", "").replace(".", "")
        skill = skill.replace("(es6+)", "").replace("(es6)", "")
        skill = skill.replace("(es5)", "").replace("(es2020)", "")
        # Handle common variations
        skill = skill.replace("git/github", "git")
        skill = skill.replace("restful apis", "rest").replace("rest api", "rest")
        skill = skill.replace("restful", "rest")
        skill = skill.replace("_", " ").replace("-", " ")
        skill = skill.replace("/", " ")
        # Remove extra spaces
        skill = re.sub(r"\s+", " ", skill).strip()
        
        return skill
    
    def match_skills(self, candidate_skills: List[str], job_skills: List[str]) -> Tuple[List[str], List[str], float]:
        """Match candidate skills against job requirements using semantic similarity"""
        if not job_skills:
            return [], [], 1.0  # No job skills required = perfect match
        
        if not candidate_skills:
            return [], job_skills, 0.0  # No candidate skills = no match
        
        # Normalize skills
        candidate_normalized = {self.normalize_skill(s): s for s in candidate_skills}
        job_normalized = {self.normalize_skill(s): s for s in job_skills}
        
        candidate_keys = list(candidate_normalized.keys())
        job_keys = list(job_normalized.keys())
        job_originals = [job_normalized[k] for k in job_keys]
        
        matched = []
        missing = []
        match_scores = []
        
        # Precompute embeddings for similarity matching
        similarities = None
        if len(candidate_keys) > 0 and len(job_keys) > 0:
            try:
                job_embs = self.embedding_model.encode_batch(job_keys)
                cand_embs = self.embedding_model.encode_batch(candidate_keys)
                similarities = cosine_similarity(cand_embs, job_embs)
            except Exception as e:
                logger.error(f"Embedding similarity error: {e}")
                similarities = None
        
        # Match each job skill against candidate skills
        for i, (job_key, original_job) in enumerate(zip(job_keys, job_originals)):
            best_score = 0.0
            
            # Exact match check
            if job_key in candidate_normalized:
                best_score = 1.0
            # Semantic similarity check
            elif similarities is not None and len(candidate_keys) > 0:
                max_sim = float(np.max(similarities[:, i])) if similarities.shape[0] > 0 else 0.0
                if max_sim >= 0.55:  # Threshold for semantic match
                    best_score = max_sim
            
            match_scores.append(best_score)
            
            if best_score >= 0.55:
                matched.append(original_job)
            else:
                missing.append(original_job)
        
        # Calculate skill overlap score (0-1)
        skill_overlap = sum(match_scores) / len(match_scores) if match_scores else 0.0
        
        return matched, missing, skill_overlap
    
    def compute_hybrid_score(
        self,
        candidate_embedding: List[float],
        job_embedding: List[float],
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> Dict:
        """
        Compute hybrid matching score using:
        - 70% embedding similarity (semantic understanding)
        - 30% skill overlap (keyword matching)
        """
        # Calculate embedding similarity score
        embed_score = 0.0
        if candidate_embedding and job_embedding:
            try:
                cv_vec = np.array(candidate_embedding).reshape(1, -1)
                job_vec = np.array(job_embedding).reshape(1, -1)
                embed_score = float(cosine_similarity(cv_vec, job_vec)[0][0])
                logger.debug(f"Embedding similarity score: {embed_score:.4f}")
            except Exception as e:
                logger.error(f"Embedding similarity failed: {e}")
                embed_score = 0.0
        
        # Calculate skill matching score
        matched, missing, skill_overlap = self.match_skills(candidate_skills, job_skills)
        logger.debug(f"Skill overlap score: {skill_overlap:.4f}")
        logger.debug(f"Matched skills: {matched}")
        logger.debug(f"Missing skills: {missing}")
        
        # Hybrid score calculation
        final_score = (
            settings.embedding_weight * embed_score +
            settings.skill_weight * skill_overlap
        )
        final_score = round(min(max(final_score, 0.0), 1.0), 4)  # Clamp between 0 and 1
        
        # Determine status based on score
        if final_score >= settings.match_shortlist_threshold:
            status = "Shortlisted"
            ready_for_interview = True
        elif final_score >= settings.match_review_threshold:
            status = "Review"
            ready_for_interview = False
        else:
            status = "Rejected"
            ready_for_interview = False
        
        # Calculate interview readiness (using min_skill_match_threshold from config)
        min_score_threshold = settings.min_skill_match_threshold / 100.0
        ready_for_interview = ready_for_interview or (final_score >= min_score_threshold)
        
        result = {
            "similarity_score": final_score,
            "embedding_score": round(embed_score, 4),
            "skill_score": round(skill_overlap, 4),
            "matched_skills": matched,
            "missing_skills": missing,
            "ready_for_interview": ready_for_interview,
            "status": status
        }
        
        logger.info(f"Match result - Final: {final_score:.4f}, "
                   f"Embedding: {embed_score:.4f}, "
                   f"Skill: {skill_overlap:.4f}, "
                   f"Status: {status}")
        
        return result
    
    def compute_batch_scores(
        self,
        candidate_embeddings: List[List[float]],
        job_embedding: List[float],
        candidate_skills_list: List[List[str]],
        job_skills: List[str]
    ) -> List[Dict]:
        """Compute hybrid scores for multiple candidates efficiently"""
        results = []
        
        for i, (cand_emb, cand_skills) in enumerate(zip(candidate_embeddings, candidate_skills_list)):
            score = self.compute_hybrid_score(
                cand_emb, job_embedding, cand_skills, job_skills
            )
            results.append(score)
        
        return results

_matcher = None

def get_matcher():
    """Get singleton instance of SkillMatcher"""
    global _matcher
    if _matcher is None:
        _matcher = SkillMatcher()
    return _matcher