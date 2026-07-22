from datetime import datetime
import uuid
import base64

def generate_candidate_id() -> str:
    """Generate unique candidate ID: cand_YYYYMMDD_XXXX"""
    date_str = datetime.now().strftime("%Y%m%d")
    # Short UUID (first 8 chars of base64 encoded uuid)
    short_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)[:6].decode('ascii').replace('-', '').replace('_', '')
    return f"cand_{date_str}_{short_uuid}"

def generate_job_id() -> str:
    """Generate unique job ID: job_YYYYMMDD_XXXX"""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)[:6].decode('ascii').replace('-', '').replace('_', '')
    return f"job_{date_str}_{short_uuid}"

def generate_match_id() -> str:
    """Generate unique match ID: match_YYYYMMDD_XXXX"""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)[:6].decode('ascii').replace('-', '').replace('_', '')
    return f"match_{date_str}_{short_uuid}"