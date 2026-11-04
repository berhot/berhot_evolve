"""Future source adapters: clear training rights and provenance are mandatory.

Social accounts, user posts, image/video, audio and 3D are deliberately *disabled*
by default. A future adapter must use authorized APIs and enforce platform terms.
"""
from dataclasses import dataclass
from typing import Optional

SUPPORTED_MODALITIES = {"text", "image", "audio", "video", "3d"}
DEFAULT_ALLOWED_RIGHTS = {"CC0-1.0", "PUBLIC-DOMAIN", "SELF-GENERATED"}


@dataclass(frozen=True)
class SourceCandidate:
    source_url: str
    modality: str
    license_id: str
    rights_evidence_url: str
    permission_for_training: bool
    contains_personal_data: bool = False
    owner_consent_reference: Optional[str] = None


def can_ingest(candidate: SourceCandidate):
    if candidate.modality not in SUPPORTED_MODALITIES:
        return False, "Unsupported modality"
    if not candidate.source_url or not candidate.rights_evidence_url:
        return False, "Missing source or license evidence"
    if candidate.license_id not in DEFAULT_ALLOWED_RIGHTS:
        return False, "License not approved for default intake; review required"
    if not candidate.permission_for_training:
        return False, "Training permission not established"
    if candidate.contains_personal_data and not candidate.owner_consent_reference:
        return False, "Personal data requires a documented lawful basis and additional review"
    return True, "Eligible for further inspection; not automatically downloaded"


class FutureConnector:
    """API contract only. Does not scrape social media or other websites."""
    name: str = "unconfigured"
    enabled: bool = False

    def discover(self):
        raise NotImplementedError("Implement an authorized source-specific discovery adapter")

    def get_rights_evidence(self, item):
        raise NotImplementedError("Determine exact license, rights holder, and platform permissions")

    def fetch(self, item):
        raise NotImplementedError("Fetch through official APIs with rate limits and privacy controls")
