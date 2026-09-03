import imaplib
import email
from email.header import decode_header
import asyncio
from config.settings import settings

def _fetch_emails_sync(count: int = 3) -> list:
    if not settings.EMAIL_ADDRESS or not settings.EMAIL_APP_PASSWORD:
        return []

    imap = imaplib.IMAP4_SSL('imap.gmail.com')
    imap.login(settings.EMAIL_ADDRESS, settings.EMAIL_APP_PASSWORD)
    imap.select('INBOX')
    status, messages = imap.search(None, 'UNSEEN')
    mail_ids = messages[0].split()
    mail_ids = mail_ids[-count:] if len(mail_ids) >= count else mail_ids
    results = []
    for mail_id in reversed(mail_ids):
        _, data = imap.fetch(mail_id, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        subject, enc = decode_header(msg['Subject'])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(enc or 'utf-8')
        sender = msg.get('From', 'Unknown')
        results.append({'subject': subject, 'from': sender})
    imap.close()
    imap.logout()
    return results

async def handle(text: str) -> str:
    if not settings.EMAIL_ADDRESS or not settings.EMAIL_APP_PASSWORD:
        return "Email credentials are not set in .env. Configure EMAIL_ADDRESS and EMAIL_APP_PASSWORD to check your Gmail inbox."

    try:
        loop = asyncio.get_running_loop()
        emails = await loop.run_in_executor(None, _fetch_emails_sync, 3)
    except Exception as e:
        return f'Could not fetch emails: {str(e)}'

    if not emails:
        return 'You have no unread emails right now.'

    lines = [f'{i+1}. From {e["from"]}: {e["subject"]}' for i, e in enumerate(emails)]
    return f'You have {len(emails)} unread email(s): ' + ' | '.join(lines)
