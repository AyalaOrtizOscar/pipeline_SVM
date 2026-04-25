#!/usr/bin/env python3
"""
send_definition.py
Envio diario de una definicion de la campana desde `definitions.json`
usando Gmail SMTP con contrasena de aplicacion.

Uso:
    # Envia la definicion correspondiente a la fecha actual (America/Bogota)
    python send_definition.py

    # Simula sin enviar (imprime lo que habria enviado)
    python send_definition.py --dry-run

    # Fuerza el envio de una fecha especifica (util para pruebas / recuperar dias perdidos)
    python send_definition.py --date 2026-04-21

    # Envia un dia por indice (1..25)
    python send_definition.py --day 1

Variables de entorno requeridas (produccion):
    GMAIL_SENDER       email remitente (ej. ayalaortizoscarivan@gmail.com)
    GMAIL_APP_PASSWORD contrasena de aplicacion Gmail (16 chars, sin espacios)
    GMAIL_RECIPIENT    email destinatario (default: igual a GMAIL_SENDER)

Registra resultados en `send_log.csv` (append only).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import smtplib
import ssl
import sys
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DEFINITIONS = HERE / "definitions.json"
LOG = HERE / "send_log.csv"

BOGOTA = ZoneInfo("America/Bogota")


def load_entries() -> list[dict]:
    with DEFINITIONS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def pick_entry(entries: list[dict], target: date, by_day: int | None) -> dict | None:
    if by_day is not None:
        for e in entries:
            if int(e["day"]) == by_day:
                return e
        return None
    target_iso = target.isoformat()
    for e in entries:
        if e["date"] == target_iso:
            return e
    return None


def build_html(entry: dict) -> str:
    safe_body = (
        entry["body"]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 680px; margin: 0 auto; padding: 24px; color: #1f2937;">
  <div style="border-left: 4px solid #2563eb; padding-left: 16px; margin-bottom: 20px;">
    <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em;">
      Dia {entry['day']} / 25 &middot; Tema: {entry['topic']}
    </div>
    <div style="font-size: 20px; font-weight: 600; margin-top: 4px;">{entry['subject']}</div>
  </div>
  <div style="font-size: 15px; line-height: 1.55;">{safe_body}</div>
  <hr style="margin-top: 32px; border: 0; border-top: 1px solid #e5e7eb;">
  <div style="font-size: 12px; color: #9ca3af;">
    Campana automatica de definiciones para sustentacion &middot; generada desde definitions.json
  </div>
</body>
</html>"""


def send_email(entry: dict, sender: str, password: str, recipient: str) -> None:
    msg = EmailMessage()
    msg["From"] = f"Copiloto de Sustentacion <{sender}>"
    msg["To"] = recipient
    msg["Subject"] = entry["subject"]
    msg.set_content(entry["body"])
    msg.add_alternative(build_html(entry), subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)


def log_result(entry: dict | None, target: date, status: str, note: str = "") -> None:
    exists = LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "target_date", "day", "topic", "status", "note"])
        w.writerow([
            datetime.now(BOGOTA).isoformat(timespec="seconds"),
            target.isoformat(),
            entry["day"] if entry else "",
            entry["topic"] if entry else "",
            status,
            note,
        ])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="YYYY-MM-DD (default: hoy America/Bogota)")
    p.add_argument("--day", type=int, help="Indice de dia (1..25) — alternativo a --date")
    p.add_argument("--dry-run", action="store_true", help="Imprime lo que enviaria, no manda nada")
    args = p.parse_args()

    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = datetime.now(BOGOTA).date()

    entries = load_entries()
    entry = pick_entry(entries, target, args.day)

    if entry is None:
        msg = f"No hay definicion programada para {args.day or target}."
        print(msg)
        log_result(None, target, "skipped", msg)
        return 0

    if args.dry_run:
        print("=" * 70)
        print(f"[DRY RUN] date={target} day={entry['day']}/25 topic={entry['topic']}")
        print(f"Subject: {entry['subject']}")
        print("-" * 70)
        print(entry["body"])
        print("=" * 70)
        log_result(entry, target, "dry-run")
        return 0

    sender = os.environ.get("GMAIL_SENDER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    recipient = os.environ.get("GMAIL_RECIPIENT", sender).strip()

    if not sender or not password:
        err = "Faltan variables GMAIL_SENDER / GMAIL_APP_PASSWORD en el entorno."
        print(f"ERROR: {err}", file=sys.stderr)
        log_result(entry, target, "error", err)
        return 2

    try:
        send_email(entry, sender, password, recipient)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        print(f"ERROR al enviar: {err}", file=sys.stderr)
        log_result(entry, target, "error", err)
        return 3

    print(f"OK enviado dia {entry['day']}/25 -> {recipient}: {entry['subject']}")
    log_result(entry, target, "sent", recipient)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
