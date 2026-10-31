from django.conf import settings
from django.core.management.base import BaseCommand

from ai_api.assistant import purge_expired_conversations


class Command(BaseCommand):
    help = "Delete chat conversation memory unused for CHAT_MEMORY_RETENTION_DAYS days."

    def handle(self, *args, **options):
        deleted = purge_expired_conversations()
        self.stdout.write(f"Deleted {deleted} conversation(s) older than {settings.CHAT_MEMORY_RETENTION_DAYS} days.")
