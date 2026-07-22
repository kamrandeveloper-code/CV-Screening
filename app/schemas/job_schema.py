from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime
import json

class JobBase(BaseModel):
    title: str
    description: str
    company: Optional[str] = None
    skills_required: Optional[List[str]] = None

class JobCreate(JobBase):
    job_id: Optional[str] = None  # User can provide custom ID

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v):
        if v is not None:
            if len(v) > 50:
                raise ValueError("ID must be less than 50 characters")
            if not v.replace('_', '').replace('-', '').isalnum():
                raise ValueError("ID must be alphanumeric (underscores and hyphens allowed)")
        return v

class JobResponse(JobBase):
    job_id: str
    skills_required: List[str]
    created_at: datetime
    
    @field_validator('skills_required', mode='before')
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