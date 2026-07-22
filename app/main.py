from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import engine, Base
from app.api.endpoints import candidate, job, match
from app.core.config import settings

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="AI Recruitment System",
    description="Intelligent candidate-to-job matching using ML embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candidate.router)
app.include_router(job.router)
app.include_router(match.router)

@app.get("/")
def root():
    """Health check"""
    return {
        "message": "AI Recruitment System is running",
        "version": "1.0.0",
        "endpoints": {
            "candidates": "/docs#/Candidates",
            "jobs": "/docs#/Jobs",
            "matching": "/docs#/Matching"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )