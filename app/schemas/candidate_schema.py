from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime
import json

class CandidateBase(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class CandidateCreate(CandidateBase):
    candidate_id: Optional[str] = None  # User can provide custom ID

    @field_validator('candidate_id')
    @classmethod
    def validate_candidate_id(cls, v):
        if v is not None:
            if len(v) > 50:
                raise ValueError("ID must be less than 50 characters")
            if not v.replace('_', '').replace('-', '').isalnum():
                raise ValueError("ID must be alphanumeric (underscores and hyphens allowed)")
        return v

class CandidateResponse(CandidateBase):
    candidate_id: str
    skills: List[str]
    created_at: datetime
    
    @field_validator('skills', mode='before')
    @classmethod
    def parse_skills(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v or []
    
    class Config:
        from_attributes = True

class CandidateDetail(CandidateResponse):
    education: Optional[List[str]] = None
    experience: Optional[List[str]] = None
    
    @field_validator('education', 'experience', mode='before')
    @classmethod
    def parse_json(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return None
        return v