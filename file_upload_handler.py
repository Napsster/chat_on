#!/usr/bin/env python3
"""
File Upload Handler for Recykal HR Chatbot
Handles user authentication, file uploads, and integration with Google Drive
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from password import hash_password, verify_password

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class User(Base):
    """User model for database"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    fullname = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class Upload(Base):
    """Upload history model"""
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_type = Column(String(50))

class KnowledgeDocument(Base):
    """A document that contributes to the WhatsApp bot's knowledge base"""
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    uploaded_by = Column(String(100), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True, nullable=False)
    # 'pre_join', 'post_join', or 'both' (visible to segments that aren't yet known)
    audience = Column(String(20), default='both', nullable=False)

class FileUploadManager:
    """Manages file uploads and user authentication"""

    def __init__(self,
                 upload_dir: str = "/home/chetan/apps/onboarding-agent/uploads",
                 db_path: str = "/home/chetan/apps/onboarding-agent/chatbot.db",
                 max_file_size_mb: int = 50):
        """
        Initialize file upload manager

        Args:
            upload_dir: Directory to store uploaded files
            db_path: Path to SQLite database
            max_file_size_mb: Maximum file size in MB
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.max_file_size = max_file_size_mb * 1024 * 1024

        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"FileUploadManager initialized: {upload_dir}")

    def _migrate_schema(self):
        """create_all() only creates missing tables, not missing columns on
        existing ones — handle columns added after the initial deploy here."""
        with self.engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(knowledge_documents)"))]
            if cols and 'audience' not in cols:
                conn.execute(text(
                    "ALTER TABLE knowledge_documents ADD COLUMN audience VARCHAR(20) DEFAULT 'both' NOT NULL"
                ))
                conn.commit()
                logger.info("Migrated knowledge_documents: added audience column (default 'both')")

    # User Authentication Methods
    def register_user(self, username: str, email: str, password: str, fullname: str = "") -> Tuple[bool, str]:
        """
        Register a new user

        Args:
            username: Username
            email: Email address
            password: Plain text password
            fullname: User's full name

        Returns:
            Tuple of (success, message)
        """
        session = self.Session()
        try:
            # Check if user exists
            existing = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()

            if existing:
                return False, "Username or email already exists"

            # Create new user
            password_hash = hash_password(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                fullname=fullname
            )
            session.add(new_user)
            session.commit()

            logger.info(f"User registered: {username}")
            return True, "User registered successfully"

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, str(e)
        finally:
            session.close()

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate a user

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            Tuple of (success, message)
        """
        session = self.Session()
        try:
            user = session.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()

            if not user:
                return False, "User not found"

            if not verify_password(password, user.password_hash):
                return False, "Invalid password"

            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()

            logger.info(f"User authenticated: {username}")
            return True, "Authentication successful"

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, str(e)
        finally:
            session.close()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Look up a user by username (no password hash in the result) — used by auth.get_current_user"""
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return None
            return {'id': user.id, 'username': user.username, 'email': user.email, 'fullname': user.fullname}
        except Exception as e:
            logger.error(f"Error looking up user: {e}")
            return None
        finally:
            session.close()

    # File Upload Methods
    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate uploaded file

        Args:
            filename: Name of file
            file_size: Size of file in bytes

        Returns:
            Tuple of (valid, message)
        """
        # Check file size
        if file_size > self.max_file_size:
            return False, f"File too large (max {self.max_file_size // 1024 // 1024}MB)"

        # Check file type
        allowed_extensions = ['.md', '.txt', '.pdf', '.docx']
        file_ext = Path(filename).suffix.lower()

        if file_ext not in allowed_extensions:
            return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"

        return True, "File valid"

    def save_file(self, username: str, file_content: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Save uploaded file

        Args:
            username: Username of uploader
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Create user upload directory
            user_dir = self.upload_dir / username
            user_dir.mkdir(exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = Path(filename).stem
            file_ext = Path(filename).suffix
            unique_filename = f"{file_stem}_{timestamp}{file_ext}"

            file_path = user_dir / unique_filename

            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"File saved: {file_path}")

            # Record in database
            session = self.Session()
            upload_record = Upload(
                username=username,
                filename=filename,
                file_path=str(file_path),
                file_size=len(file_content),
                file_type=Path(filename).suffix
            )
            session.add(upload_record)
            session.commit()
            session.close()

            return True, "File uploaded successfully", str(file_path)

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False, str(e), None

    def get_upload_history(self, username: str) -> List[Dict]:
        """
        Get upload history for a user

        Args:
            username: Username

        Returns:
            List of upload records
        """
        session = self.Session()
        try:
            uploads = session.query(Upload).filter(
                Upload.username == username
            ).order_by(Upload.uploaded_at.desc()).all()

            history = [
                {
                    'filename': u.filename,
                    'size': u.file_size,
                    'uploadedAt': u.uploaded_at.isoformat(),
                    'type': u.file_type
                }
                for u in uploads
            ]

            return history

        except Exception as e:
            logger.error(f"Error getting upload history: {e}")
            return []
        finally:
            session.close()

    # Knowledge base document methods
    def add_knowledge_document(self, title: str, content: str, uploaded_by: str, audience: str = 'both') -> Tuple[bool, str, Optional[int]]:
        """Add a document that contributes to the bot's knowledge base.
        audience: 'pre_join', 'post_join', or 'both' (default)."""
        if audience not in ('pre_join', 'post_join', 'both'):
            audience = 'both'
        session = self.Session()
        try:
            doc = KnowledgeDocument(
                title=title,
                content=content,
                uploaded_by=uploaded_by,
                audience=audience,
            )
            session.add(doc)
            session.commit()
            logger.info(f"Knowledge document added: {title} by {uploaded_by}")
            return True, "Knowledge document added", doc.id
        except Exception as e:
            logger.error(f"Error adding knowledge document: {e}")
            return False, str(e), None
        finally:
            session.close()

    def list_knowledge_documents(self, active_only: bool = False) -> List[Dict]:
        """List knowledge base documents, most recent first"""
        session = self.Session()
        try:
            query = session.query(KnowledgeDocument)
            if active_only:
                query = query.filter(KnowledgeDocument.active == True)
            docs = query.order_by(KnowledgeDocument.uploaded_at.desc()).all()

            return [
                {
                    'id': d.id,
                    'title': d.title,
                    'content': d.content,
                    'uploaded_by': d.uploaded_by,
                    'uploaded_at': d.uploaded_at.isoformat(),
                    'active': d.active,
                    'audience': d.audience,
                    'size': len(d.content),
                }
                for d in docs
            ]
        except Exception as e:
            logger.error(f"Error listing knowledge documents: {e}")
            return []
        finally:
            session.close()

    def set_knowledge_document_active(self, doc_id: int, active: bool) -> Tuple[bool, str]:
        """Activate or deactivate a knowledge document without deleting it"""
        session = self.Session()
        try:
            doc = session.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
            if not doc:
                return False, "Document not found"
            doc.active = active
            session.commit()
            logger.info(f"Knowledge document {doc_id} set active={active}")
            return True, "Updated"
        except Exception as e:
            logger.error(f"Error updating knowledge document: {e}")
            return False, str(e)
        finally:
            session.close()

    def set_knowledge_document_audience(self, doc_id: int, audience: str) -> Tuple[bool, str]:
        """Re-tag a knowledge document's audience ('pre_join'/'post_join'/'both')"""
        if audience not in ('pre_join', 'post_join', 'both'):
            return False, "Invalid audience"
        session = self.Session()
        try:
            doc = session.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
            if not doc:
                return False, "Document not found"
            doc.audience = audience
            session.commit()
            logger.info(f"Knowledge document {doc_id} audience set to {audience}")
            return True, "Updated"
        except Exception as e:
            logger.error(f"Error updating knowledge document audience: {e}")
            return False, str(e)
        finally:
            session.close()

    def delete_knowledge_document(self, doc_id: int) -> Tuple[bool, str]:
        """Permanently remove a knowledge document"""
        session = self.Session()
        try:
            doc = session.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
            if not doc:
                return False, "Document not found"
            session.delete(doc)
            session.commit()
            logger.info(f"Knowledge document {doc_id} deleted")
            return True, "Deleted"
        except Exception as e:
            logger.error(f"Error deleting knowledge document: {e}")
            return False, str(e)
        finally:
            session.close()

    # Google Drive Integration
    def sync_uploads_to_drive(self, username: str, google_sync) -> Tuple[bool, str]:
        """
        Sync uploaded files to Google Drive

        Args:
            username: Username
            google_sync: GoogleDriveSync instance

        Returns:
            Tuple of (success, message)
        """
        try:
            user_upload_dir = self.upload_dir / username

            if not user_upload_dir.exists():
                return False, "No uploads found for this user"

            # Get markdown files
            md_files = list(user_upload_dir.glob("*.md"))

            if not md_files:
                return False, "No markdown files to sync"

            # This is a placeholder - implement actual Google Drive upload
            # For now, just log the operation
            logger.info(f"Would sync {len(md_files)} files from {username} to Google Drive")

            return True, f"Ready to sync {len(md_files)} files"

        except Exception as e:
            logger.error(f"Error syncing to Drive: {e}")
            return False, str(e)


class UploadedFileProcessor:
    """Process uploaded files for chatbot integration"""

    def __init__(self, knowledge_dir: str = "/home/chetan/apps/onboarding-agent"):
        """
        Initialize processor

        Args:
            knowledge_dir: Directory containing knowledge base
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.base_file = self.knowledge_dir / "knowledge_base.md"
        self.output_file = self.knowledge_dir / "knowledge.md"

    def rebuild_knowledge_md(self, active_docs: List[Dict], output_file: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Regenerate a knowledge file from the curated base file plus a given
        set of active documents (already filtered by caller, e.g. by
        audience). Never touches the base file.

        Args:
            active_docs: list of dicts with 'title' and 'content' keys
            output_file: where to write; defaults to self.output_file (knowledge.md)

        Returns:
            Tuple of (success, message)
        """
        output_file = output_file or self.output_file
        try:
            base_content = self.base_file.read_text(encoding='utf-8') if self.base_file.exists() else ""
            sections = [base_content.rstrip()] if base_content.strip() else []
            for doc in active_docs:
                sections.append(f"## File: {doc['title']}\n\n{doc['content'].strip()}")
            merged = "\n\n---\n\n".join(sections)
            output_file.write_text(merged, encoding='utf-8')
            logger.info(f"{output_file.name} rebuilt: base + {len(active_docs)} document(s)")
            return True, f"{output_file.name} rebuilt with {len(active_docs)} document(s)"
        except Exception as e:
            logger.error(f"Error rebuilding knowledge.md: {e}")
            return False, str(e)

    def process_markdown(self, file_path: str, username: str) -> Tuple[bool, str]:
        """
        Process markdown file and integrate with knowledge base

        Args:
            file_path: Path to markdown file
            username: Username who uploaded it

        Returns:
            Tuple of (success, message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add metadata
            filename = Path(file_path).name
            metadata = f"\n\n<!-- Uploaded by: {username} on {datetime.now().isoformat()} -->\n"
            content += metadata

            logger.info(f"Processed markdown file: {filename}")
            return True, "Markdown processed successfully"

        except Exception as e:
            logger.error(f"Error processing markdown: {e}")
            return False, str(e)

    def process_text(self, filename: str, content: str, username: str) -> Tuple[bool, str]:
        """
        Process text content and save as markdown

        Args:
            filename: Name of content
            content: Text content
            username: Username who uploaded it

        Returns:
            Tuple of (success, message)
        """
        try:
            # Create markdown file
            md_filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            md_content = f"# {filename}\n\n{content}\n\n*Contributed by: {username}*\n"

            md_path = self.knowledge_dir / "uploads" / md_filename
            md_path.parent.mkdir(parents=True, exist_ok=True)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            logger.info(f"Created markdown from text: {md_filename}")
            return True, "Content saved successfully"

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return False, str(e)
