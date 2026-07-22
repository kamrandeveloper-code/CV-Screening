import click
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.services.candidate_service import CandidateService
from app.services.match_service import MatchService
from tabulate import tabulate

@click.group()
def cli():
    """Candidate & Job Matching CLI"""
    pass

@cli.command()
@click.option('--limit', default=10, help='Number of candidates to show')
def list_candidates(limit):
    """List all candidates in database"""
    db = SessionLocal()
    candidates = CandidateService.get_all_candidates(db, limit=limit)
    
    if not candidates:
        click.echo("❌ No candidates found")
        return
    
    data = [
        [
            c.candidate_id,
            c.name or "N/A",
            c.email or "N/A",
            c.phone or "N/A",
            len(c.get_skills()),
        ]
        for c in candidates
    ]
    
    click.echo(tabulate(
        data,
        headers=["Candidate ID", "Name", "Email", "Phone", "Skills Count"],
        tablefmt="grid"
    ))

@cli.command()
@click.option('--candidate-id', required=True, help='Candidate ID to delete')
def delete_candidate(candidate_id):
    """Delete a specific candidate"""
    db = SessionLocal()
    
    candidate = CandidateService.get_candidate(db, candidate_id)
    if not candidate:
        click.echo(f"❌ Candidate {candidate_id} not found")
        return
    
    if click.confirm(f"Delete candidate '{candidate.name}'? This will also delete related matches."):
        # Delete related matches first
        from app.db.models import Match
        db.query(Match).filter(Match.candidate_id == candidate_id).delete()
        
        CandidateService.delete_candidate(db, candidate_id)
        click.echo(f"✅ Candidate {candidate_id} deleted successfully")
    else:
        click.echo("Cancelled")

@cli.command()
@click.option('--job-id', required=True, help='Job ID to show matches for')
def show_matches(job_id):
    """Show all candidates matched to a job"""
    db = SessionLocal()
    
    matches = MatchService.get_matches_for_job(db, job_id)
    
    if not matches:
        click.echo(f"❌ No matches found for job {job_id}")
        return
    
    data = [
        [
            m.match_id,
            m.candidate_id,
            f"{m.similarity_score:.1f}%",
            "✅ Ready" if m.ready_for_interview else "❌ Not Ready",
            len(m.get_matched_skills()),
            len(m.get_missing_skills()),
        ]
        for m in matches
    ]
    
    click.echo(f"\n📊 Matches for Job {job_id}:\n")
    click.echo(tabulate(
        data,
        headers=["Match ID", "Candidate ID", "Score", "Interview Ready", "Matched", "Missing"],
        tablefmt="grid"
    ))

@cli.command()
@click.option('--job-id', required=True, help='Job ID')
def ready_candidates(job_id):
    """Show only candidates ready for interview (60%+ match)"""
    db = SessionLocal()
    
    matches = MatchService.get_ready_candidates_for_job(db, job_id)
    
    if not matches:
        click.echo(f"❌ No candidates ready for interview for job {job_id}")
        return
    
    data = [
        [
            m.match_id,
            m.candidate_id,
            f"{m.similarity_score:.1f}%",
            ", ".join(m.get_matched_skills()[:3]),  # Show first 3
        ]
        for m in matches
    ]
    
    click.echo(f"\n✅ {len(matches)} Candidates Ready for Interview (Job {job_id}):\n")
    click.echo(tabulate(
        data,
        headers=["Match ID", "Candidate ID", "Match %", "Top Skills"],
        tablefmt="grid"
    ))

@cli.command()
def cleanup():
    """Remove duplicate candidates based on email/phone"""
    db = SessionLocal()
    from app.db.models import Candidate, Match
    
    candidates = CandidateService.get_all_candidates(db, limit=1000)
    duplicates_removed = 0
    
    # Group by email
    email_map = {}
    for candidate in candidates:
        if candidate.email:
            if candidate.email in email_map:
                # Delete this duplicate
                db.query(Match).filter(Match.candidate_id == candidate.candidate_id).delete()
                db.delete(candidate)
                duplicates_removed += 1
            else:
                email_map[candidate.email] = candidate.candidate_id
    
    db.commit()
    
    if duplicates_removed > 0:
        click.echo(f"✅ Removed {duplicates_removed} duplicate candidates")
    else:
        click.echo("✅ No duplicates found")

if __name__ == '__main__':
    cli()