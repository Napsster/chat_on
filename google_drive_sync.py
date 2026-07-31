#!/usr/bin/env python3
"""
Google Drive Integration for Recykal HR Chatbot
Handles dynamic synchronization of markdown files from a Google Drive folder
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import io

try:
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False

logger = logging.getLogger(__name__)

class GoogleDriveSync:
    """
    Manages synchronization of markdown files from a Google Drive folder.
    """

    def __init__(self,
                 service_account_file: Optional[str] = None,
                 folder_id: str = "1aGaZa6N2i2CbZ1xY7k9AAzUO9b3hy4Ju",
                 cache_dir: str = "/home/chetan/apps/onboarding-agent",
                 cache_ttl_hours: int = 1):
        """
        Initialize Google Drive sync.

        Args:
            service_account_file: Path to Google service account JSON file
            folder_id: Google Drive folder ID containing markdown files
            cache_dir: Directory to store downloaded files and cache
            cache_ttl_hours: Cache TTL in hours (how often to sync with Drive)
        """
        self.folder_id = folder_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.service = None
        self.last_sync = None
        self.sync_cache_file = self.cache_dir / ".google_drive_sync_cache.json"

        if HAS_GOOGLE_API:
            try:
                if service_account_file and os.path.exists(service_account_file):
                    self.service = self._authenticate_service_account(service_account_file)
                    logger.info("Google Drive API authenticated with service account")
                else:
                    logger.warning("Service account file not found or not provided")
            except Exception as e:
                logger.error(f"Failed to authenticate Google Drive API: {e}")
        else:
            logger.warning("Google API client not installed. Install with: pip install google-api-python-client")

    def _authenticate_service_account(self, service_account_file: str):
        """Authenticate using service account credentials."""
        creds = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=creds)

    def _load_sync_cache(self) -> Dict:
        """Load the sync cache file."""
        if self.sync_cache_file.exists():
            try:
                with open(self.sync_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load sync cache: {e}")
        return {}

    def _save_sync_cache(self, cache: Dict) -> None:
        """Save the sync cache file."""
        try:
            with open(self.sync_cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Could not save sync cache: {e}")

    def _should_sync(self) -> bool:
        """Check if enough time has passed to sync with Google Drive."""
        cache = self._load_sync_cache()
        if 'last_sync_time' not in cache:
            return True

        last_sync = datetime.fromisoformat(cache['last_sync_time'])
        return datetime.now() - last_sync > self.cache_ttl

    def list_markdown_files(self) -> List[Dict]:
        """
        List all markdown files in the Google Drive folder.

        Returns:
            List of dicts with 'id', 'name', 'modified_time' keys
        """
        if not self.service:
            logger.error("Google Drive API not authenticated")
            return []

        try:
            query = f"'{self.folder_id}' in parents and mimeType='text/plain' and (name contains '.md' or name ends with 'md')"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, modifiedTime, size)',
                pageSize=100
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} markdown files in Google Drive folder")
            return files
        except Exception as e:
            logger.error(f"Error listing files from Google Drive: {e}")
            return []

    def download_file(self, file_id: str, file_name: str) -> Optional[str]:
        """
        Download a single file from Google Drive.

        Args:
            file_id: Google Drive file ID
            file_name: Name to save the file as

        Returns:
            Content of the file as string, or None if failed
        """
        if not self.service:
            logger.error("Google Drive API not authenticated")
            return None

        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='replace')
            logger.info(f"Downloaded file: {file_name}")
            return content
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None

    def sync_knowledge_base(self) -> Tuple[bool, str]:
        """
        Synchronize all markdown files from Google Drive folder into a single knowledge base.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.service:
            return False, "Google Drive API not authenticated"

        if not self._should_sync():
            logger.info("Skipping sync - cache is still valid")
            return True, "Using cached knowledge base"

        logger.info("Starting knowledge base sync from Google Drive...")

        try:
            # List all markdown files
            files = self.list_markdown_files()
            if not files:
                return False, "No markdown files found in Google Drive folder"

            # Download all files
            merged_content = self._create_merged_knowledge_header()
            file_count = 0

            for file_info in sorted(files, key=lambda x: x['name']):
                file_id = file_info['id']
                file_name = file_info['name']

                content = self.download_file(file_id, file_name)
                if content:
                    merged_content += f"\n## Source: {file_name}\n\n"
                    merged_content += content
                    merged_content += "\n\n---\n\n"
                    file_count += 1

            if file_count == 0:
                return False, "Could not download any markdown files"

            # Save merged knowledge
            knowledge_file = self.cache_dir / "knowledge.md"
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                f.write(merged_content)

            file_size = knowledge_file.stat().st_size
            logger.info(f"Merged knowledge base: {file_count} files, {file_size} bytes")

            # Update sync cache
            cache = self._load_sync_cache()
            cache['last_sync_time'] = datetime.now().isoformat()
            cache['file_count'] = file_count
            cache['file_size'] = file_size
            cache['files'] = [{'id': f['id'], 'name': f['name']} for f in files]
            self._save_sync_cache(cache)

            return True, f"Successfully synced {file_count} markdown files from Google Drive"

        except Exception as e:
            logger.error(f"Error during sync: {e}")
            return False, f"Sync failed: {str(e)}"

    def _create_merged_knowledge_header(self) -> str:
        """Create the header for merged knowledge base."""
        return f"""# Recykal Employee Onboarding Knowledge Base

**Auto-synced from Google Drive folder** on {datetime.now().isoformat()}

This knowledge base is automatically updated with the latest markdown files from the company Google Drive folder.

---

"""

    def get_sync_status(self) -> Dict:
        """Get current sync status and cache info."""
        cache = self._load_sync_cache()
        return {
            'api_available': HAS_GOOGLE_API and self.service is not None,
            'last_sync_time': cache.get('last_sync_time'),
            'file_count': cache.get('file_count', 0),
            'file_size': cache.get('file_size', 0),
            'should_sync': self._should_sync(),
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600
        }


def setup_google_drive_credentials(credentials_json_path: str) -> bool:
    """
    Helper function to set up Google Drive credentials.

    Args:
        credentials_json_path: Path to the service account JSON file

    Returns:
        True if credentials are valid and can authenticate
    """
    if not os.path.exists(credentials_json_path):
        logger.error(f"Credentials file not found: {credentials_json_path}")
        return False

    if not HAS_GOOGLE_API:
        logger.error("Google API libraries not installed")
        return False

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_json_path,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        # Test the connection
        service.about().get(fields='storageQuota').execute()
        logger.info("Google Drive credentials validated successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to validate credentials: {e}")
        return False
