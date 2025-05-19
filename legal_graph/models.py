from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class LegalDocument(BaseModel):
    id: str
    type: str  # loi, décret, ordonnance, circulaire
    number: str
    title: str
    content: str
    date_published: datetime
    journal_officiel: Optional[str] = None
    embedding: Optional[List[float]] = None

class LegalRelation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str  # modifié_par, complété_par, abrogé_par, référence, remplace
    date_effective: datetime
    description: Optional[str] = None

class JournalOfficiel(BaseModel):
    number: str
    date_published: datetime
    content: str
    embedding: Optional[List[float]] = None
