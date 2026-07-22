"""SQLAlchemy models"""
from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.sql import func
from app.db.database import Base
import json

class Candidate(Base):
    __tablename__ = "candidates"
    
    candidate_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=True, index=True)
    email = Column(String(200), nullable=True, index=True)
    phone = Column(String(50), nullable=True, index=True)
    raw_text = Column(Text)
    skills = Column(Text)
    education = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    embedding = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_candidate_email_phone", "email", "phone"),
    )
    
    def set_skills(self, skills_list):
        self.skills = json.dumps(list(set(skills_list)))
    
    def get_skills(self):
        return json.loads(self.skills) if self.skills else []
    
    def set_embedding(self, embedding_vec):
        self.embedding = json.dumps(embedding_vec.tolist() if hasattr(embedding_vec, "tolist") else list(embedding_vec))
    
    def get_embedding(self):
        return json.loads(self.embedding) if self.embedding else None

class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String(50), primary_key=True, index=True)
    title = Column(String(200), index=True)
    description = Column(Text)
    skills_required = Column(Text)
    company = Column(String(200), nullable=True, index=True)
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def set_skills(self, skills_list):
        self.skills_required = json.dumps(list(set(skills_list)))
    
    def get_skills(self):
        return json.loads(self.skills_required) if self.skills_required else []
    
    def set_embedding(self, embedding_vec):
        self.embedding = json.dumps(embedding_vec.tolist() if hasattr(embedding_vec, "tolist") else list(embedding_vec))
    
    def get_embedding(self):
        return json.loads(self.embedding) if self.embedding else None

class Match(Base):
    __tablename__ = "matches"
    
    match_id = Column(String(50), primary_key=True, index=True)
    candidate_id = Column(String(50), index=True)
    job_id = Column(String(50), index=True)
    similarity_score = Column(Float, index=True)
    skill_score = Column(Float, default=0.0)
    embedding_score = Column(Float, default=0.0)
    matched_skills = Column(Text)
    missing_skills = Column(Text)
    ready_for_interview = Column(Boolean, default=False, index=True)
    status = Column(String(20), default="Rejected")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_match_job_score", "job_id", "similarity_score"),
        Index("idx_match_candidate", "candidate_id", "created_at"),
    )
    
    def set_matched_skills(self, skills_list):
        self.matched_skills = json.dumps(skills_list)
    
    def get_matched_skills(self):
        return json.loads(self.matched_skills) if self.matched_skills else []
    
    def set_missing_skills(self, skills_list):
        self.missing_skills = json.dumps(skills_list)
    
    def get_missing_skills(self):
        return json.loads(self.missing_skills) if self.missing_skills else []