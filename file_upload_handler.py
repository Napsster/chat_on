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
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, text, func
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
    # 'admin' (full tool access) or 'user' (chat-only — for real employees/
    # candidates trying the bot without knowledge-base edit or outreach access)
    role = Column(String(20), default='admin', nullable=False)

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

class QuestionLog(Base):
    """Every real question asked of the bot (WhatsApp or web chat) — feeds
    the unanswered-questions and repeated-questions reports."""
    __tablename__ = "question_log"

    id = Column(Integer, primary_key=True)
    asked_at = Column(DateTime, default=datetime.utcnow)
    session_key = Column(String(100), nullable=False)  # phone number or web-<username>
    channel = Column(String(20))  # 'whatsapp' or 'web'
    question = Column(Text, nullable=False)
    unanswered = Column(Boolean, default=False, nullable=False)
    segment = Column(String(20))  # 'pre_join' / 'post_join' / None at time of asking

class UserSession(Base):
    """Per-session chat state — one row per WhatsApp phone or web-<username>
    session. Replaces the old data/users/*.json files' top-level fields
    (history itself lives in ChatMessage, keyed the same way)."""
    __tablename__ = "user_sessions"

    session_key = Column(String(200), primary_key=True)
    email = Column(String(200), nullable=True)
    segment = Column(String(20), nullable=True)
    segment_asked = Column(Boolean, default=False, nullable=False)
    last_message_at = Column(String(50), nullable=True)  # ISO string, matches datetime.fromisoformat() usage in agent.py
    last_feedback_prompted_date = Column(String(10), nullable=True)  # 'YYYY-MM-DD' — at most one 👍/👎 prompt per day
    last_event_nudge_date = Column(String(10), nullable=True)  # 'YYYY-MM-DD' (IST) — caps the today/tomorrow event nudge to once/day/person

class FeedbackLog(Base):
    """One row per feedback-button MESSAGE that's been rated at least once
    (not one row per tap) — sparse by design, buttons only go out once/day
    per phone. wamid ties this back to the exact button message (see
    FeedbackPrompt); a second tap on the same message updates this row's
    rating/rated_at in place instead of inserting a duplicate — WhatsApp
    briefly allows both buttons to be tapped before it locks the message,
    and without this a rapid down-then-up produced two rows that then had
    to be reconciled downstream (chat viewer, export) instead of never
    existing in the first place. wamid is nullable for rows logged before
    this column existed, which can't be de-duplicated this way."""
    __tablename__ = "feedback_log"

    id = Column(Integer, primary_key=True)
    rated_at = Column(DateTime, default=datetime.utcnow)
    session_key = Column(String(100), nullable=False)
    question = Column(Text)
    answer = Column(Text)
    rating = Column(String(10), nullable=False)  # 'up' or 'down'
    wamid = Column(String(100), nullable=True, unique=True, index=True)

class FeedbackPrompt(Base):
    """One row per feedback-button message actually sent — keyed by its
    WhatsApp message id (wamid). WhatsApp buttons never expire, and a tap
    arrives as a reply "context" pointing at this exact wamid — so when the
    tap comes back (possibly days later, after the person has sent more
    messages), we look up the wamid here instead of guessing "whatever's
    most recent right now," which silently mislabels a late tap against
    the wrong question. Rows are small and low-volume (same once/day cap
    as the buttons themselves) — no cleanup needed."""
    __tablename__ = "feedback_prompts"

    id = Column(Integer, primary_key=True)
    wamid = Column(String(100), nullable=False, unique=True, index=True)
    session_key = Column(String(100), nullable=False)
    question = Column(Text)
    answer = Column(Text)
    sent_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    """One row per chat turn — replaces the JSON 'history' list. Ordered by
    id for a given session_key to reconstruct the conversation."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_key = Column(String(200), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def _sqlcipher_creator(db_path: str, key: str):
    """SQLAlchemy create_engine(creator=...) factory — opens the
    SQLCipher-encrypted db file and unlocks it with the passphrase before
    SQLAlchemy issues any queries. Only imports sqlcipher3 when actually
    needed, so environments without DB_ENCRYPTION_KEY set (e.g. a plain
    local dev db) don't need the package installed at all."""
    from sqlcipher3 import dbapi2 as sqlcipher

    def creator():
        conn = sqlcipher.connect(db_path)
        conn.execute(f"PRAGMA key='{key}'")
        return conn
    return creator


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

        # Initialize database — SQLCipher-encrypted if DB_ENCRYPTION_KEY is
        # set (production), plain SQLite otherwise (local dev, no key configured).
        db_key = os.environ.get('DB_ENCRYPTION_KEY')
        if db_key:
            self.engine = create_engine('sqlite://', creator=_sqlcipher_creator(db_path, db_key))
            logger.info("Database opened with SQLCipher encryption")
        else:
            self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"FileUploadManager initialized: {upload_dir}")

    def _migrate_schema(self):
        """create_all() only creates missing tables, not missing columns on
        existing ones — handle columns added after the initial deploy here."""
        with self.engine.connect() as conn:
            kd_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(knowledge_documents)"))]
            if kd_cols and 'audience' not in kd_cols:
                conn.execute(text(
                    "ALTER TABLE knowledge_documents ADD COLUMN audience VARCHAR(20) DEFAULT 'both' NOT NULL"
                ))
                conn.commit()
                logger.info("Migrated knowledge_documents: added audience column (default 'both')")

            user_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(users)"))]
            if user_cols and 'role' not in user_cols:
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'admin' NOT NULL"
                ))
                conn.commit()
                logger.info("Migrated users: added role column (default 'admin' — existing accounts keep full access)")

            # One-time rename from the earlier staff/pilot naming — safe to run
            # every startup, becomes a no-op once nothing's left to rename.
            if user_cols:
                renamed_admin = conn.execute(text("UPDATE users SET role = 'admin' WHERE role = 'staff'")).rowcount
                renamed_user = conn.execute(text("UPDATE users SET role = 'user' WHERE role = 'pilot'")).rowcount
                if renamed_admin or renamed_user:
                    conn.commit()
                    logger.info(f"Migrated users: renamed {renamed_admin} 'staff'->'admin', {renamed_user} 'pilot'->'user' role value(s)")

            us_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(user_sessions)"))]
            if us_cols and 'last_feedback_prompted_date' not in us_cols:
                conn.execute(text(
                    "ALTER TABLE user_sessions ADD COLUMN last_feedback_prompted_date VARCHAR(10)"
                ))
                conn.commit()
                logger.info("Migrated user_sessions: added last_feedback_prompted_date column")

            fl_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(feedback_log)"))]
            if fl_cols and 'wamid' not in fl_cols:
                conn.execute(text("ALTER TABLE feedback_log ADD COLUMN wamid VARCHAR(100)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_feedback_log_wamid ON feedback_log (wamid)"))
                conn.commit()
                logger.info("Migrated feedback_log: added wamid column (re-taps on the same message now update in place)")

            if us_cols and 'last_event_nudge_date' not in us_cols:
                conn.execute(text(
                    "ALTER TABLE user_sessions ADD COLUMN last_event_nudge_date VARCHAR(10)"
                ))
                conn.commit()
                logger.info("Migrated user_sessions: added last_event_nudge_date column")

    # User Authentication Methods
    def register_user(self, username: str, email: str, password: str, fullname: str = "", role: str = 'admin') -> Tuple[bool, str]:
        """
        Register a new user

        Args:
            username: Username
            email: Email address
            password: Plain text password
            fullname: User's full name
            role: 'admin' (full tool access, default) or 'user' (chat-only)

        Returns:
            Tuple of (success, message)
        """
        if role not in ('admin', 'user'):
            role = 'admin'
        if not email.strip().lower().endswith('@recykal.com'):
            return False, "Email must be a @recykal.com address"
        session = self.Session()
        try:
            # Check if user exists — case-insensitive, so "Anjali" and "anjali"
            # collide as the same account rather than creating duplicates.
            uname_lower = username.strip().lower()
            email_lower = email.strip().lower()
            existing = session.query(User).filter(
                (func.lower(User.username) == uname_lower) | (func.lower(User.email) == email_lower)
            ).first()

            if existing:
                return False, "Username or email already exists"

            # Create new user
            password_hash = hash_password(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                fullname=fullname,
                role=role
            )
            session.add(new_user)
            session.commit()

            logger.info(f"User registered: {username} (role={role})")
            return True, "User registered successfully"

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, str(e)
        finally:
            session.close()

    def reset_password(self, username: str, new_password: str) -> Tuple[bool, str]:
        """Admin-triggered password reset — same case-insensitive
        username-or-email lookup as login, so it finds the account
        regardless of which form the admin typed."""
        session = self.Session()
        try:
            lookup_lower = username.strip().lower()
            user = session.query(User).filter(
                (func.lower(User.username) == lookup_lower) | (func.lower(User.email) == lookup_lower)
            ).first()

            if not user:
                return False, "User not found"

            user.password_hash = hash_password(new_password)
            session.commit()

            logger.info(f"Password reset for user: {user.username}")
            return True, "Password reset successfully"

        except Exception as e:
            logger.error(f"Password reset error: {e}")
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
            # Case-insensitive: "Anjali" and "anjali" resolve to the same account.
            login_lower = username.strip().lower()
            user = session.query(User).filter(
                (func.lower(User.username) == login_lower) | (func.lower(User.email) == login_lower)
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
        """Look up a user by username or email, case-insensitive (no password
        hash in the result) — used by auth.get_current_user on every
        authenticated request, so this must resolve the same way
        authenticate_user's login lookup does, or a case-mismatched or
        email-based login issues a token that 401s on the very next request."""
        session = self.Session()
        try:
            lookup_lower = username.strip().lower()
            user = session.query(User).filter(
                (func.lower(User.username) == lookup_lower) | (func.lower(User.email) == lookup_lower)
            ).first()
            if not user:
                return None
            return {'id': user.id, 'username': user.username, 'email': user.email, 'fullname': user.fullname, 'role': user.role}
        except Exception as e:
            logger.error(f"Error looking up user: {e}")
            return None
        finally:
            session.close()

    def list_users(self, role: Optional[str] = None) -> List[Dict]:
        """List users, most recently created first. Filter by role if given."""
        session = self.Session()
        try:
            query = session.query(User)
            if role:
                query = query.filter(User.role == role)
            users = query.order_by(User.created_at.desc()).all()
            return [
                {
                    'id': u.id,
                    'username': u.username,
                    'email': u.email,
                    'fullname': u.fullname,
                    'role': u.role,
                    'created_at': u.created_at.isoformat() if u.created_at else None,
                    'last_login': u.last_login.isoformat() if u.last_login else None,
                }
                for u in users
            ]
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
        finally:
            session.close()

    def delete_user_by_username(self, username: str) -> Tuple[bool, str]:
        """Permanently remove a user account (e.g. revoking a pilot participant)"""
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return False, "User not found"
            session.delete(user)
            session.commit()
            logger.info(f"User deleted: {username}")
            return True, "Deleted"
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False, str(e)
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

    # Question log (feeds the unanswered/repeated-questions reports)
    def log_question(self, session_key: str, channel: str, question: str, unanswered: bool, segment: Optional[str] = None) -> None:
        session = self.Session()
        try:
            session.add(QuestionLog(
                session_key=session_key,
                channel=channel,
                question=question,
                unanswered=unanswered,
                segment=segment,
            ))
            session.commit()
        except Exception as e:
            logger.error(f"Error logging question: {e}")
        finally:
            session.close()

    def get_questions_for_session(self, session_key: str) -> List[Dict]:
        """All logged questions for one session_key (phone number or
        web-<username>), oldest first — the read side of log_question."""
        session = self.Session()
        try:
            rows = (
                session.query(QuestionLog)
                .filter(QuestionLog.session_key == session_key)
                .order_by(QuestionLog.asked_at.asc())
                .all()
            )
            return [
                {
                    "asked_at": row.asked_at.isoformat() if row.asked_at else None,
                    "channel": row.channel,
                    "question": row.question,
                    "unanswered": row.unanswered,
                    "segment": row.segment,
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching questions for {session_key}: {e}")
            return []
        finally:
            session.close()

    def log_feedback(self, session_key: str, question: str, answer: str, rating: str, wamid: str | None = None) -> None:
        """Record a 👍/👎 tap. `rating` is 'up' or 'down'. When `wamid` is
        given (the exact button message being rated) and a row for that
        wamid already exists — a re-tap on the same message — update its
        rating/rated_at in place instead of inserting a duplicate row.
        The update only goes one direction: down -> up is honored (they
        reconsidered and the answer turned out fine), but up -> down is not
        (an already-positive rating can't be walked back by a later tap) —
        WhatsApp gives no way to disable the buttons after the first tap, so
        this is the backend's way of making a second tap not meaningfully
        flip-flop a rating once it's positive."""
        session = self.Session()
        try:
            existing = None
            if wamid:
                existing = session.query(FeedbackLog).filter(FeedbackLog.wamid == wamid).first()
            if existing and existing.rating == "up" and rating == "down":
                return
            if existing:
                existing.rating = rating
                existing.rated_at = datetime.utcnow()
            else:
                session.add(FeedbackLog(session_key=session_key, question=question, answer=answer, rating=rating, wamid=wamid))
            session.commit()
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
        finally:
            session.close()

    def record_feedback_prompt(self, wamid: str, session_key: str, question: str, answer: str) -> None:
        """Remember which question+answer a just-sent feedback-button message
        was for, keyed by its WhatsApp message id — so a tap on it (even a
        late one, after more messages have gone back and forth) can be
        attributed correctly instead of guessed. See FeedbackPrompt docstring."""
        if not wamid:
            return
        session = self.Session()
        try:
            session.add(FeedbackPrompt(wamid=wamid, session_key=session_key, question=question, answer=answer))
            session.commit()
        except Exception as e:
            logger.error(f"Error recording feedback prompt: {e}")
        finally:
            session.close()

    def get_feedback_prompt(self, wamid: str) -> Optional[Dict]:
        """Look up the question+answer a feedback-button tap belongs to, by
        the wamid WhatsApp's reply `context.id` points at. Returns None if
        this wamid was never recorded (e.g. a button sent before this
        feature existed) — caller should fall back gracefully."""
        if not wamid:
            return None
        session = self.Session()
        try:
            row = session.query(FeedbackPrompt).filter(FeedbackPrompt.wamid == wamid).first()
            if not row:
                return None
            return {"session_key": row.session_key, "question": row.question, "answer": row.answer}
        except Exception as e:
            logger.error(f"Error looking up feedback prompt: {e}")
            return None
        finally:
            session.close()

    def get_feedback_for_session(self, session_key: str) -> List[Dict]:
        """Every rating for one session, oldest first — used to attach a
        rating next to the specific answer it was for in the chat viewer
        and the export."""
        session = self.Session()
        try:
            rows = (
                session.query(FeedbackLog)
                .filter(FeedbackLog.session_key == session_key)
                .order_by(FeedbackLog.rated_at.asc())
                .all()
            )
            return [
                {
                    "question": row.question,
                    "answer": row.answer,
                    "rating": row.rating,
                    "rated_at": row.rated_at.isoformat() if row.rated_at else None,
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching feedback for {session_key}: {e}")
            return []
        finally:
            session.close()

    def get_feedback_summary(self, session_key: str) -> Dict:
        """{'up': N, 'down': M} rollup for one session — powers the compact
        badge in the WhatsApp session list (individual ratings would be too
        many to show inline there once a conversation spans weeks)."""
        session = self.Session()
        try:
            up = session.query(FeedbackLog).filter(
                FeedbackLog.session_key == session_key, FeedbackLog.rating == "up"
            ).count()
            down = session.query(FeedbackLog).filter(
                FeedbackLog.session_key == session_key, FeedbackLog.rating == "down"
            ).count()
            return {"up": up, "down": down}
        except Exception as e:
            logger.error(f"Error fetching feedback summary for {session_key}: {e}")
            return {"up": 0, "down": 0}
        finally:
            session.close()

    def get_whatsapp_sessions(self) -> List[Dict]:
        """List every WhatsApp session (phone numbers only — web-<username>
        sessions excluded), most recently active first, with a message
        count. Powers the portal's WhatsApp chat list."""
        session = self.Session()
        try:
            rows = (
                session.query(UserSession)
                .filter(~UserSession.session_key.like("web-%"))
                .order_by(UserSession.last_message_at.desc())
                .all()
            )
            out = []
            for row in rows:
                count = (
                    session.query(ChatMessage)
                    .filter(ChatMessage.session_key == row.session_key)
                    .count()
                )
                out.append({
                    "phone": row.session_key,
                    "last_message_at": row.last_message_at,
                    "message_count": count,
                })
            return out
        except Exception as e:
            logger.error(f"Error listing WhatsApp sessions: {e}")
            return []
        finally:
            session.close()

    def get_whatsapp_summary_stats(self) -> Dict:
        """Aggregate counts for the WhatsApp tab's summary bar — people,
        real questions asked, and feedback tally. Scoped to WhatsApp
        sessions only (phone-number keys), same "web-%" exclusion as
        get_whatsapp_sessions. Cheap: plain counts, no per-session history
        replay (unlike the export, which needs the actual Q&A pairs)."""
        session = self.Session()
        try:
            people = session.query(UserSession).filter(~UserSession.session_key.like("web-%")).count()
            questions = session.query(QuestionLog).filter(QuestionLog.channel == "whatsapp").count()
            positive = session.query(FeedbackLog).filter(
                ~FeedbackLog.session_key.like("web-%"), FeedbackLog.rating == "up"
            ).count()
            negative = session.query(FeedbackLog).filter(
                ~FeedbackLog.session_key.like("web-%"), FeedbackLog.rating == "down"
            ).count()
            return {"people": people, "questions": questions, "positive": positive, "negative": negative}
        except Exception as e:
            logger.error(f"Error computing WhatsApp summary stats: {e}")
            return {"people": 0, "questions": 0, "positive": 0, "negative": 0}
        finally:
            session.close()

    def get_all_whatsapp_questions(self) -> List[str]:
        """Every logged WhatsApp question's raw text — feeds the summary
        bar's "distinct topics" count (clustered + cached in agent.py, since
        embedding-based clustering is too slow to redo on every page load)."""
        session = self.Session()
        try:
            rows = session.query(QuestionLog.question).filter(QuestionLog.channel == "whatsapp").all()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Error fetching all WhatsApp questions: {e}")
            return []
        finally:
            session.close()

    def get_user_data(self, session_key: str) -> Dict:
        """Reconstruct the same dict shape the old data/users/*.json files
        held: {"phone": ..., "email": ..., "history": [...], "segment": ...,
        "segment_asked": ..., "last_message_at": ...}. Callers mutate this
        dict freely, then pass the whole thing back to save_user_data."""
        session = self.Session()
        try:
            row = session.query(UserSession).filter(UserSession.session_key == session_key).first()
            messages = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_key == session_key)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            history = [
                {
                    "role": m.role,
                    "content": m.content,
                    # ISO 8601 with an explicit UTC 'Z' — created_at is stored via
                    # datetime.utcnow() (naive UTC), and the 'Z' is what makes
                    # JS's `new Date(...)` parse it as UTC and convert to the
                    # viewer's own local time, instead of misreading it as
                    # already-local and rendering the wrong clock time.
                    "created_at": (m.created_at.isoformat() + "Z") if m.created_at else None,
                }
                for m in messages
            ]
            if row is None:
                return {"phone": session_key, "email": None, "history": history}
            data = {"phone": session_key, "email": row.email, "history": history}
            if row.segment is not None:
                data["segment"] = row.segment
            if row.segment_asked:
                data["segment_asked"] = row.segment_asked
            if row.last_message_at:
                data["last_message_at"] = row.last_message_at
            if row.last_feedback_prompted_date:
                data["last_feedback_prompted_date"] = row.last_feedback_prompted_date
            if row.last_event_nudge_date:
                data["last_event_nudge_date"] = row.last_event_nudge_date
            return data
        except Exception as e:
            logger.error(f"Error reading user session {session_key}: {e}")
            return {"phone": session_key, "email": None, "history": []}
        finally:
            session.close()

    def save_user_data(self, session_key: str, data: Dict) -> None:
        """Persist the full dict back — upserts the session row. Message
        rows are append-only: every caller only ever appends new turns to
        `data["history"]` before calling this (never edits or reorders an
        existing entry), so only the turns beyond what's already stored get
        inserted. This matters beyond efficiency — it's what lets each
        ChatMessage keep its real, original `created_at` instead of every
        row being reset to "now" on every single turn, which is what a
        wholesale delete-and-reinsert (the previous approach) did. The one
        exception is a shrink (e.g. /chat/reset clearing history to []) —
        append-only can't reconcile fewer rows than already stored, so that
        case still falls back to a full wipe."""
        session = self.Session()
        try:
            row = session.query(UserSession).filter(UserSession.session_key == session_key).first()
            if row is None:
                row = UserSession(session_key=session_key)
                session.add(row)
            row.email = data.get("email")
            row.segment = data.get("segment")
            row.segment_asked = bool(data.get("segment_asked", False))
            row.last_message_at = data.get("last_message_at")
            row.last_feedback_prompted_date = data.get("last_feedback_prompted_date")
            row.last_event_nudge_date = data.get("last_event_nudge_date")

            new_history = data.get("history", [])
            existing_count = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_key == session_key)
                .count()
            )
            if len(new_history) < existing_count:
                session.query(ChatMessage).filter(ChatMessage.session_key == session_key).delete()
                existing_count = 0
            for m in new_history[existing_count:]:
                session.add(ChatMessage(session_key=session_key, role=m["role"], content=m["content"]))
            session.commit()
        except Exception as e:
            logger.error(f"Error saving user session {session_key}: {e}")
            session.rollback()
        finally:
            session.close()

    def has_messaged(self, session_key: str) -> bool:
        session = self.Session()
        try:
            return session.query(UserSession.session_key).filter(UserSession.session_key == session_key).first() is not None
        finally:
            session.close()

    def all_messaged_session_keys(self) -> List[str]:
        session = self.Session()
        try:
            return [r[0] for r in session.query(UserSession.session_key).all()]
        finally:
            session.close()

    def get_questions_since(self, since: datetime, unanswered_only: bool = False) -> List[Dict]:
        session = self.Session()
        try:
            query = session.query(QuestionLog).filter(QuestionLog.asked_at >= since)
            if unanswered_only:
                query = query.filter(QuestionLog.unanswered == True)
            rows = query.order_by(QuestionLog.asked_at.asc()).all()
            return [
                {
                    'id': r.id,
                    'asked_at': r.asked_at.isoformat() if r.asked_at else None,
                    'session_key': r.session_key,
                    'channel': r.channel,
                    'question': r.question,
                    'unanswered': r.unanswered,
                    'segment': r.segment,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching question log: {e}")
            return []
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
