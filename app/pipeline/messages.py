"""User-facing texts for paths that do not produce a model answer."""
from __future__ import annotations

from app.pipeline.prompts import REFUSAL  # noqa: F401  (re-exported)


def exhausted_message(request_id: str, probe_in_s: float) -> str:
    when = (
        f"RelayFlow re-checks the provider about every {max(1, round(probe_in_s / 60))} minute(s) and answers "
        "normally again as soon as it accepts requests."
        if probe_in_s > 0
        else "RelayFlow is re-checking the provider now."
    )
    return (
        "RelayFlow cannot generate a new answer right now. The external model provider's daily usage quota "
        "is exhausted, and I will not guess an answer without the model or the cited evidence. "
        "Your question was NOT silently discarded: it was not processed and it was not queued, so please "
        f"submit it again later (reference {request_id}). The provider has not told us when the quota "
        "returns, so I cannot give a reliable time. " + when
    )


def unavailable_message(request_id: str) -> str:
    return (
        "The external model provider is failing repeatedly, so RelayFlow is pausing calls to it for a few "
        "seconds instead of retrying into the failure. Your question was not answered and was not queued; "
        f"please resubmit shortly (reference {request_id})."
    )


def clarify_message(reason: str) -> str:
    return (
        f"I need a little more detail before I can search the research library. {reason} "
        "Please say which paper or concept you mean (Attention Is All You Need, Moshi, or H-Net dynamic chunking)."
    )


def rejected_queue_full_message(request_id: str) -> str:
    return (
        "RelayFlow is at capacity and its request queue is full, so this question was rejected before "
        f"processing (reference {request_id}). It was not queued. Please resubmit in a moment."
    )


def timeout_message(request_id: str) -> str:
    return (
        "This question could not be completed within its time limit, so RelayFlow gave up rather than "
        f"return an unverified answer (reference {request_id}). Please resubmit."
    )
