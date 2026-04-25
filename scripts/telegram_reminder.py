#!/usr/bin/env python3
"""
Telegram automatic task reminder.
Reads pending tasks from plan and memory, sends to JARVIS every 3 hours.
Runs via Windows Task Scheduler.

Usage:
  python telegram_reminder.py

Reads:
  - ~/.claude/channels/telegram/.env (TELEGRAM_BOT_TOKEN)
  - ~/.claude/channels/telegram/access.json (allowlist chat_id)
  - ~/.claude/plans/*.md (pending tasks)
  - ~/.claude/projects/D--pipeline-SVM-article1/memory/ (project status)
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
import requests

# Paths
TELEGRAM_ENV = Path.home() / ".claude" / "channels" / "telegram" / ".env"
TELEGRAM_ACCESS = Path.home() / ".claude" / "channels" / "telegram" / "access.json"
PLAN_DIR = Path.home() / ".claude" / "plans"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "D--pipeline-SVM-article1" / "memory"
DASHBOARD = Path.home() / "Obsidian" / "OscarVault" / "00-Dashboard" / "Dashboard.md"

def read_env_token():
    """Read TELEGRAM_BOT_TOKEN from .env file."""
    if not TELEGRAM_ENV.exists():
        return None
    with open(TELEGRAM_ENV, 'r') as f:
        for line in f:
            if line.startswith('TELEGRAM_BOT_TOKEN='):
                return line.split('=', 1)[1].strip()
    return None

def read_access_config():
    """Read chat_id from access.json."""
    if not TELEGRAM_ACCESS.exists():
        return None
    with open(TELEGRAM_ACCESS, 'r') as f:
        config = json.load(f)
    if config.get('allowFrom'):
        return config['allowFrom'][0]  # First (only) allowed user
    return None

def extract_tasks_from_plan():
    """Read latest plan file and extract task checklist."""
    plan_files = sorted(PLAN_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plan_files:
        return []

    plan_path = plan_files[0]
    with open(plan_path, 'r', encoding='utf-8') as f:
        content = f.read()

    tasks = []
    # Extract bullet points with [ ] (unchecked tasks)
    for line in content.split('\n'):
        if '- [ ]' in line:
            task_text = line.replace('- [ ]', '').strip()
            if task_text:
                tasks.append(task_text)

    return tasks[:5]  # Return top 5 pending tasks

def extract_project_status():
    """Read project status from memory files."""
    status_file = MEMORY_DIR / "project_thesis_status.md"
    if not status_file.exists():
        return "Status: unknown"

    with open(status_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract first few lines of status
    lines = content.split('\n')[10:15]  # Skip frontmatter
    return ' | '.join(l.strip() for l in lines if l.strip())[:150]

def read_deadline():
    """Calculate days remaining to Art.2 deadline."""
    deadline = datetime(2026, 4, 15)
    today = datetime.now()
    remaining = (deadline - today).days
    return remaining

def format_message(tasks, status):
    """Format task list for Telegram."""
    deadline_days = read_deadline()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = f"📋 *Tareas Pendientes* — {timestamp}\n"
    msg += f"⏰ Deadline Art.2: {deadline_days} días\n\n"

    if tasks:
        msg += "✓ TOP 5 PENDIENTES:\n"
        for i, task in enumerate(tasks, 1):
            # Truncate task if too long
            task = task[:80]
            msg += f"{i}. {task}\n"
    else:
        msg += "✓ Todas las tareas completadas 🎉\n"

    return msg

def send_telegram(bot_token, chat_id, message):
    """Send message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending message: {e}")
        return False

def main():
    """Main entry point."""
    # Read configuration
    bot_token = read_env_token()
    chat_id = read_access_config()

    if not bot_token or not chat_id:
        print("[✗] Telegram not configured. Skipping reminder.")
        return

    # Extract tasks and status
    tasks = extract_tasks_from_plan()
    status = extract_project_status()

    # Format and send message
    message = format_message(tasks, status)

    if send_telegram(bot_token, chat_id, message):
        print(f"[✓] Reminder sent at {datetime.now().isoformat()}")
    else:
        print(f"[✗] Failed to send reminder")

if __name__ == "__main__":
    main()
