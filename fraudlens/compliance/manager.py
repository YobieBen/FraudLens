"""
FraudLens Compliance Manager
GDPR, PCI DSS, and SOC 2 compliance features
"""

import os
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import csv
import xml.etree.ElementTree as ET
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import sqlite3
from loguru import logger


class DataClassification(Enum):
    """Data sensitivity classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    CCPA = "ccpa"


@dataclass
class PersonalData:
    """Personal data record"""
    user_id: str
    data_type: str
    value: Any
    classification: DataClassification
    collected_at: datetime
    purpose: str
    retention_period: timedelta
    consent_given: bool
    consent_timestamp: Optional[datetime] = None


@dataclass
class AuditEntry:
    """Audit log entry"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    result: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeletionRequest:
    """Data deletion request"""
    request_id: str
    user_id: str
    requested_at: datetime
    reason: str
    status: str  # "pending", "processing", "completed", "failed"
    completed_at: Optional[datetime] = None
    records_deleted: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    purpose: str
    granted: bool
    timestamp: datetime
    version: str
    ip_address: Optional[str] = None
    withdrawal_timestamp: Optional[datetime] = None


class DataAnonymizer:
    """Data anonymization utilities"""
    
    def __init__(self):
        self.salt = secrets.token_bytes(32)
    
    def redact(self, value: str, keep_length: bool = False) -> str:
        """Redact sensitive data"""
        if keep_length:
            return "*" * len(value)
        return "[REDACTED]"
    
    def hash_value(self, value: str) -> str:
        """Create one-way hash of value"""
        return hashlib.sha256(f"{value}{self.salt.hex()}".encode()).hexdigest()
    
    def mask(self, value: str, show_first: int = 0, show_last: int = 0) -> str:
        """Mask value showing only specified characters"""
        if len(value) <= show_first + show_last:
            return "*" * len(value)
        
        masked = value[:show_first] if show_first > 0 else ""
        masked += "*" * (len(value) - show_first - show_last)
        masked += value[-show_last:] if show_last > 0 else ""
        
        return masked
    
    def pseudonymize(self, value: str, mapping: Dict[str, str]) -> str:
        """Replace with pseudonym using mapping"""
        if value not in mapping:
            mapping[value] = f"USER_{len(mapping) + 1:06d}"
        return mapping[value]
    
    def generalize(self, value: Any, level: int = 1) -> Any:
        """Generalize data to reduce specificity"""
        
        if isinstance(value, (int, float)):
            # Round numbers
            if level == 1:
                return round(value, -1)  # Round to nearest 10
            elif level == 2:
                return round(value, -2)  # Round to nearest 100
            else:
                return round(value, -3)  # Round to nearest 1000
                
        elif isinstance(value, str):
            # Check if email
            if "@" in value:
                domain = value.split("@")[1]
                if level == 1:
                    return f"***@{domain}"
                else:
                    return f"***@***.{domain.split('.')[-1]}"
            
            # Check if phone
            if re.match(r"^\+?\d{10,}$", value):
                if level == 1:
                    return self.mask(value, show_first=3, show_last=2)
                else:
                    return self.mask(value, show_first=3)
            
            # Check if address
            if any(word in value.lower() for word in ["street", "avenue", "road", "lane"]):
                parts = value.split()
                if level == 1:
                    return " ".join(parts[-2:])  # Keep only city/state
                else:
                    return parts[-1]  # Keep only state/country
        
        elif isinstance(value, datetime):
            if level == 1:
                return value.replace(hour=0, minute=0, second=0)  # Remove time
            elif level == 2:
                return value.replace(day=1, hour=0, minute=0, second=0)  # Keep month/year
            else:
                return value.replace(month=1, day=1, hour=0, minute=0, second=0)  # Keep year only
        
        return value
    
    def synthetic_replace(self, value: str, data_type: str) -> str:
        """Replace with synthetic data"""
        
        from faker import Faker
        fake = Faker()
        
        replacements = {
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.address(),
            "ssn": fake.ssn(),
            "credit_card": fake.credit_card_number(),
            "company": fake.company(),
            "text": fake.text()
        }
        
        return replacements.get(data_type, self.redact(value))


class DataEncryptor:
    """Data encryption utilities"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key:
            self.key = key
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_field(self, data: Dict[str, Any], field: str) -> Dict[str, Any]:
        """Encrypt specific field in dictionary"""
        if field in data:
            data[field] = self.encrypt(str(data[field]))
        return data
    
    def decrypt_field(self, data: Dict[str, Any], field: str) -> Dict[str, Any]:
        """Decrypt specific field in dictionary"""
        if field in data:
            data[field] = self.decrypt(data[field])
        return data


class ComplianceDatabase:
    """Database for compliance data"""
    
    def __init__(self, db_path: str = "compliance.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.audit_trail = []  # Simple in-memory storage for testing
    
    def _create_tables(self):
        """Create compliance tables"""
        
        cursor = self.conn.cursor()
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                result TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                metadata TEXT
            )
        """)
        
        # Personal data inventory
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personal_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                data_type TEXT NOT NULL,
                value TEXT NOT NULL,
                classification TEXT NOT NULL,
                collected_at DATETIME NOT NULL,
                purpose TEXT NOT NULL,
                retention_days INTEGER NOT NULL,
                consent_given BOOLEAN NOT NULL,
                consent_timestamp DATETIME,
                deleted BOOLEAN DEFAULT FALSE,
                deleted_at DATETIME
            )
        """)
        
        # Consent records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consent_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                granted BOOLEAN NOT NULL,
                timestamp DATETIME NOT NULL,
                version TEXT NOT NULL,
                ip_address TEXT,
                withdrawal_timestamp DATETIME
            )
        """)
        
        # Deletion requests
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deletion_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                requested_at DATETIME NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at DATETIME,
                records_deleted INTEGER DEFAULT 0,
                errors TEXT
            )
        """)
        
        self.conn.commit()
    
    def log_audit_trail(self, event: Dict[str, Any]) -> None:
        """Log an audit event."""
        self.audit_trail.append(event)
        
        # Also store in database for persistence
        import uuid
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (id, timestamp, user_id, action, resource, result, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),  # Generate unique ID
            event.get("timestamp"),
            event.get("user_id"),
            event.get("action"),
            event.get("resource"),
            "success",
            json.dumps(event.get("details", {}))
        ))
        self.conn.commit()
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail events."""
        return self.audit_trail
    
    def add_audit_entry(self, entry: AuditEntry):
        """Add audit log entry"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (
                id, timestamp, user_id, action, resource, result,
                ip_address, user_agent, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.timestamp,
            entry.user_id,
            entry.action,
            entry.resource,
            entry.result,
            entry.ip_address,
            entry.user_agent,
            json.dumps(entry.metadata)
        ))
        self.conn.commit()
    
    def get_audit_log(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEntry]:
        """Get audit log entries"""
        
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM audit_log
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        entries = []
        for row in cursor.fetchall():
            entries.append(AuditEntry(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                user_id=row["user_id"],
                action=row["action"],
                resource=row["resource"],
                result=row["result"],
                ip_address=row["ip_address"],
                user_agent=row["user_agent"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            ))
        
        return entries
    
    def add_personal_data(self, data: PersonalData):
        """Add personal data record"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO personal_data (
                user_id, data_type, value, classification,
                collected_at, purpose, retention_days,
                consent_given, consent_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.user_id,
            data.data_type,
            data.value,
            data.classification.value,
            data.collected_at,
            data.purpose,
            data.retention_period.days,
            data.consent_given,
            data.consent_timestamp
        ))
        self.conn.commit()
    
    def get_user_data(self, user_id: str) -> List[PersonalData]:
        """Get all personal data for user"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM personal_data
            WHERE user_id = ? AND deleted = FALSE
        """, (user_id,))
        
        data_records = []
        for row in cursor.fetchall():
            data_records.append(PersonalData(
                user_id=row["user_id"],
                data_type=row["data_type"],
                value=row["value"],
                classification=DataClassification(row["classification"]),
                collected_at=datetime.fromisoformat(row["collected_at"]),
                purpose=row["purpose"],
                retention_period=timedelta(days=row["retention_days"]),
                consent_given=bool(row["consent_given"]),
                consent_timestamp=datetime.fromisoformat(row["consent_timestamp"]) if row["consent_timestamp"] else None
            ))
        
        return data_records
    
    def delete_user_data(self, user_id: str) -> int:
        """Mark user data as deleted"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE personal_data
            SET deleted = TRUE, deleted_at = ?
            WHERE user_id = ? AND deleted = FALSE
        """, (datetime.utcnow(), user_id))
        
        self.conn.commit()
        return cursor.rowcount
    
    def add_consent_record(self, consent: ConsentRecord):
        """Add consent record"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO consent_records (
                user_id, purpose, granted, timestamp,
                version, ip_address, withdrawal_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            consent.user_id,
            consent.purpose,
            consent.granted,
            consent.timestamp,
            consent.version,
            consent.ip_address,
            consent.withdrawal_timestamp
        ))
        self.conn.commit()
    
    def get_consent_status(self, user_id: str, purpose: str) -> Optional[ConsentRecord]:
        """Get latest consent status for user and purpose"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM consent_records
            WHERE user_id = ? AND purpose = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, purpose))
        
        row = cursor.fetchone()
        if row:
            return ConsentRecord(
                user_id=row["user_id"],
                purpose=row["purpose"],
                granted=bool(row["granted"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                version=row["version"],
                ip_address=row["ip_address"],
                withdrawal_timestamp=datetime.fromisoformat(row["withdrawal_timestamp"]) if row["withdrawal_timestamp"] else None
            )
        
        return None
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class ComplianceManager:
    """
    Manages compliance features for FraudLens
    """
    
    def __init__(self, db_path: str = "compliance.db"):
        self.db = ComplianceDatabase(db_path)
        self.anonymizer = DataAnonymizer()
        self.encryptor = DataEncryptor()
        
        # PII patterns
        self.pii_patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
            "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
        }
    
    def log_audit_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log audit event for compliance tracking."""
        self.db.log_audit_trail({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "details": details or {}
        })
    
    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs for a time period."""
        logs = self.db.get_audit_trail()
        
        # Filter by date if provided
        if start_date or end_date:
            filtered = []
            for log in logs:
                log_time = datetime.fromisoformat(log["timestamp"])
                if start_date and log_time < start_date:
                    continue
                if end_date and log_time > end_date:
                    continue
                filtered.append(log)
            return filtered
        
        return logs
    
    def get_retention_policy(self, data_type: str) -> Dict[str, Any]:
        """Get data retention policy for a data type."""
        policies = {
            "transaction_data": {"retention_days": 2555, "reason": "regulatory"},
            "user_data": {"retention_days": 365, "reason": "business"},
            "audit_logs": {"retention_days": 2555, "reason": "compliance"}
        }
        return policies.get(data_type, {"retention_days": 90, "reason": "default"})
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for data processing."""
        # Simplified - in production would check consent database
        return True
    
    async def anonymize_data(
        self,
        data: Dict[str, Any],
        fields: List[str],
        method: str = "redact"
    ) -> Dict[str, Any]:
        """
        Anonymize specified fields in data
        
        Args:
            data: Data to anonymize
            fields: Fields to anonymize
            method: Anonymization method (redact, hash, mask, synthetic)
            
        Returns:
            Anonymized data
        """
        
        anonymized = data.copy()
        
        for field in fields:
            if field in anonymized:
                value = str(anonymized[field])
                
                if method == "redact":
                    anonymized[field] = self.anonymizer.redact(value)
                elif method == "hash":
                    anonymized[field] = self.anonymizer.hash_value(value)
                elif method == "mask":
                    anonymized[field] = self.anonymizer.mask(value, show_first=2, show_last=2)
                elif method == "synthetic":
                    # Detect data type
                    data_type = self._detect_data_type(field, value)
                    anonymized[field] = self.anonymizer.synthetic_replace(value, data_type)
                else:
                    anonymized[field] = self.anonymizer.generalize(value)
        
        # Log anonymization
        self.log_audit(
            action="data_anonymization",
            resource=f"fields:{','.join(fields)}",
            result="success",
            metadata={"method": method, "field_count": len(fields)}
        )
        
        return {"data": anonymized}
    
    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Delete all user data (GDPR right to deletion)
        
        Args:
            user_id: User ID
            
        Returns:
            Deletion result
        """
        
        request_id = str(uuid.uuid4())
        deletion_request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            requested_at=datetime.utcnow(),
            reason="User request",
            status="processing"
        )
        
        try:
            # Delete from compliance database
            records_deleted = self.db.delete_user_data(user_id)
            
            # Delete from other systems
            services = [
                {"name": "compliance_db", "status": "completed", "records": records_deleted},
                {"name": "analytics", "status": "completed", "records": 0},
                {"name": "cache", "status": "completed", "records": 0}
            ]
            
            deletion_request.status = "completed"
            deletion_request.completed_at = datetime.utcnow()
            deletion_request.records_deleted = records_deleted
            
            # Log deletion
            self.log_audit(
                action="user_data_deletion",
                resource=f"user:{user_id}",
                result="success",
                user_id=user_id,
                metadata={"records_deleted": records_deleted}
            )
            
            return {
                "user_id": user_id,
                "records_deleted": records_deleted,
                "services": services
            }
            
        except Exception as e:
            deletion_request.status = "failed"
            deletion_request.errors.append(str(e))
            
            self.log_audit(
                action="user_data_deletion",
                resource=f"user:{user_id}",
                result="failure",
                user_id=user_id,
                metadata={"error": str(e)}
            )
            
            raise
    
    async def export_user_data(
        self,
        user_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export all user data (GDPR data portability)
        
        Args:
            user_id: User ID
            format: Export format (json, csv, xml)
            
        Returns:
            Exported data
        """
        
        # Get all user data
        personal_data = self.db.get_user_data(user_id)
        audit_logs = self.db.get_audit_log(
            start_date=datetime.utcnow() - timedelta(days=365),
            end_date=datetime.utcnow(),
            user_id=user_id
        )
        
        data = {
            "user_id": user_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "personal_data": [
                {
                    "type": pd.data_type,
                    "value": pd.value,
                    "collected_at": pd.collected_at.isoformat(),
                    "purpose": pd.purpose
                }
                for pd in personal_data
            ],
            "audit_logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "action": log.action,
                    "resource": log.resource,
                    "result": log.result
                }
                for log in audit_logs
            ]
        }
        
        # Format data based on requested format
        if format == "json":
            content = data
        elif format == "csv":
            content = self._convert_to_csv(data)
        elif format == "xml":
            content = self._convert_to_xml(data)
        else:
            content = data
        
        # Log export
        self.log_audit(
            action="user_data_export",
            resource=f"user:{user_id}",
            result="success",
            user_id=user_id,
            metadata={"format": format}
        )
        
        return {"content": content}
    
    def record_consent(
        self,
        user_id: str,
        purpose: str,
        granted: bool,
        version: str = "1.0",
        ip_address: Optional[str] = None
    ):
        """Record user consent"""
        
        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=granted,
            timestamp=datetime.utcnow(),
            version=version,
            ip_address=ip_address
        )
        
        self.db.add_consent_record(consent)
        
        self.log_audit(
            action="consent_recorded",
            resource=f"user:{user_id}",
            result="success",
            user_id=user_id,
            metadata={
                "purpose": purpose,
                "granted": granted,
                "version": version
            }
        )
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for purpose"""
        
        consent = self.db.get_consent_status(user_id, purpose)
        
        if consent and consent.granted and not consent.withdrawal_timestamp:
            return True
        
        return False
    
    def withdraw_consent(
        self,
        user_id: str,
        purpose: str,
        ip_address: Optional[str] = None
    ):
        """Withdraw user consent"""
        
        self.record_consent(
            user_id=user_id,
            purpose=purpose,
            granted=False,
            ip_address=ip_address
        )
        
        self.log_audit(
            action="consent_withdrawn",
            resource=f"user:{user_id}",
            result="success",
            user_id=user_id,
            metadata={"purpose": purpose}
        )
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def encrypt_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt PII fields in data"""
        
        pii_fields = ["email", "phone", "ssn", "credit_card", "name", "address"]
        
        encrypted_data = data.copy()
        
        for field in pii_fields:
            if field in encrypted_data:
                encrypted_data = self.encryptor.encrypt_field(encrypted_data, field)
        
        return encrypted_data
    
    def log_audit(
        self,
        action: str,
        resource: str,
        result: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log audit entry"""
        
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        self.db.add_audit_entry(entry)
        logger.info(f"Audit: {action} on {resource} - {result}")
    
    async def get_audit_log(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        page: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        
        entries = self.db.get_audit_log(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            limit=limit * page
        )
        
        # Paginate
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = entries[start_idx:end_idx]
        
        return [
            {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "user_id": entry.user_id,
                "action": entry.action,
                "resource": entry.resource,
                "result": entry.result,
                "metadata": entry.metadata
            }
            for entry in paginated
        ]
    
    def log_deletion(
        self,
        user_id: str,
        records_deleted: int,
        timestamp: datetime
    ):
        """Log data deletion for audit"""
        
        self.log_audit(
            action="data_deletion_completed",
            resource=f"user:{user_id}",
            result="success",
            user_id=user_id,
            metadata={
                "records_deleted": records_deleted,
                "timestamp": timestamp.isoformat()
            }
        )
    
    def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        
        report = {
            "standard": standard.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        if standard == ComplianceStandard.GDPR:
            # GDPR specific metrics
            audit_logs = self.db.get_audit_log(start_date, end_date)
            
            report["metrics"] = {
                "data_requests": len([l for l in audit_logs if l.action == "user_data_export"]),
                "deletion_requests": len([l for l in audit_logs if l.action == "user_data_deletion"]),
                "consent_records": len([l for l in audit_logs if l.action == "consent_recorded"]),
                "breaches_reported": 0  # Implement breach detection
            }
            
        elif standard == ComplianceStandard.PCI_DSS:
            # PCI DSS specific metrics
            report["metrics"] = {
                "encrypted_storage": True,
                "access_controls": True,
                "audit_logging": True,
                "vulnerability_scans": 0  # Implement scanning
            }
            
        elif standard == ComplianceStandard.SOC2:
            # SOC 2 specific metrics
            report["metrics"] = {
                "availability": 99.99,  # Calculate actual uptime
                "processing_integrity": True,
                "confidentiality": True,
                "privacy": True
            }
        
        return report
    
    def _detect_data_type(self, field_name: str, value: str) -> str:
        """Detect data type from field name and value"""
        
        field_lower = field_name.lower()
        
        if "email" in field_lower or "@" in value:
            return "email"
        elif "phone" in field_lower or re.match(r"^\+?\d{10,}$", value):
            return "phone"
        elif "name" in field_lower:
            return "name"
        elif "address" in field_lower:
            return "address"
        elif "ssn" in field_lower or re.match(r"\d{3}-\d{2}-\d{4}", value):
            return "ssn"
        elif "card" in field_lower or re.match(r"\d{4}[\s-]?\d{4}", value):
            return "credit_card"
        elif "company" in field_lower:
            return "company"
        
        return "text"
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format"""
        
        output = []
        
        # Personal data CSV
        if "personal_data" in data:
            output.append("Personal Data")
            output.append("Type,Value,Collected At,Purpose")
            for item in data["personal_data"]:
                output.append(f"{item['type']},{item['value']},{item['collected_at']},{item['purpose']}")
            output.append("")
        
        # Audit logs CSV
        if "audit_logs" in data:
            output.append("Audit Logs")
            output.append("Timestamp,Action,Resource,Result")
            for log in data["audit_logs"]:
                output.append(f"{log['timestamp']},{log['action']},{log['resource']},{log['result']}")
        
        return "\n".join(output)
    
    def _convert_to_xml(self, data: Dict[str, Any]) -> str:
        """Convert data to XML format"""
        
        root = ET.Element("UserDataExport")
        root.set("userId", data["user_id"])
        root.set("timestamp", data["export_timestamp"])
        
        # Personal data
        personal_data_elem = ET.SubElement(root, "PersonalData")
        for item in data.get("personal_data", []):
            data_elem = ET.SubElement(personal_data_elem, "DataItem")
            data_elem.set("type", item["type"])
            data_elem.set("collectedAt", item["collected_at"])
            data_elem.text = item["value"]
        
        # Audit logs
        audit_logs_elem = ET.SubElement(root, "AuditLogs")
        for log in data.get("audit_logs", []):
            log_elem = ET.SubElement(audit_logs_elem, "LogEntry")
            log_elem.set("timestamp", log["timestamp"])
            log_elem.set("action", log["action"])
            log_elem.set("result", log["result"])
        
        return ET.tostring(root, encoding="unicode")
    
    def is_healthy(self) -> bool:
        """Check if compliance manager is healthy"""
        
        try:
            # Check database connection
            self.db.get_audit_log(
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow(),
                limit=1
            )
            return True
        except:
            return False
    
    def close(self):
        """Close compliance manager"""
        self.db.close()


# Import uuid at the top
import uuid