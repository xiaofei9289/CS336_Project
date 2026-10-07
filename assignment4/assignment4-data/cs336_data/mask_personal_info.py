"""Mask emails, phone numbers, and IPv4 addresses."""

import re

_EMAIL = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._%+-]*@"
    r"[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}"
)
# 10-digit numbers, with optional separators: 2831823829, 283-182-3829,
# (283) 182 3829, (283)-182-3829.
_PHONE = re.compile(
    r"(?<!\d)"
    r"(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-]?)"
    r"\d{3}[\s\-]?\d{4}"
    r"(?!\d)"
)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP = re.compile(rf"(?<!\d)(?:{_OCTET}\.){{3}}{_OCTET}(?!\d)")


def _replace(pattern: re.Pattern[str], text: str, token: str) -> tuple[str, int]:
    masked, count = pattern.subn(token, text)
    return masked, count


def mask_emails(text: str) -> tuple[str, int]:
    return _replace(_EMAIL, text, "|||EMAIL_ADDRESS|||")


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _replace(_PHONE, text, "|||PHONE_NUMBER|||")


def mask_ips(text: str) -> tuple[str, int]:
    return _replace(_IP, text, "|||IP_ADDRESS|||")
