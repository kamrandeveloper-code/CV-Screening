from pydantic import BaseModel, field_validator
from typing import List
from datetime import datetime
import json

class MatchRequest(BaseModel):
    candidate_id: str
    job_id: str

class MatchResponse(BaseModel):
    match_id: str
    candidate_id: str
    job_id: str
    similarity_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    ready_for_interview: bool
    created_at: datetime
    
    @field_validator('matched_skills', mode='before')
    @classmethod
    def parse_matched_skills(cls, v):
        """Convert JSON string to list if needed"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        return v or []
    
    @field_validator('missing_skills', mode='before')
    @classmethod
    def parse_missing_skills(cls, v):
        """Convert JSON string to list if needed"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        return v or []
    
    class Config:
        from_attributes = True