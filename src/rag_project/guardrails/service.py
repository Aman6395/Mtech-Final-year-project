# import os
# from pathlib import Path

# os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

# from nemoguardrails import LLMRails, RailsConfig


# class GuardrailsService:
#     REFUSAL_MESSAGE = (
#         "I'm sorry, I cannot assist with this topic. "
#         "Please ask a question related to the supported scope of this RAG system."
#     )

#     def __init__(self):
#         config_path = Path(__file__).parent / "config"
#         config = RailsConfig.from_path(str(config_path))
#         self.rails = LLMRails(config)

#     async def validate_query(self, query: str) -> dict:
#         if not query or not query.strip():
#             return {
#                 "allowed": False,
#                 "query": None,
#                 "reason": "Empty query",
#             }

#         response = await self.rails.generate_async(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": query.strip(),
#                 }
#             ],
#             options={"rails": ["input"]},
#         )

#         # response_text = response.get("content", "").strip()

#         # if response_text == self.REFUSAL_MESSAGE:
#         #     return {
#         #         "allowed": False,
#         #         "query": None,
#         #         "reason": "Restricted topic or unsafe input",
#         #     }

#         # return {
#         #     "allowed": True,
#         #     "query": response_text,
#         #     "reason": None,
#         # }
#         return response

############## v2


# import os

# # Needed for engine: google_genai on newer NeMo Guardrails versions.
# # Must be set BEFORE nemoguardrails is imported.
# os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

# import logging
# from functools import lru_cache
# from pathlib import Path

# from nemoguardrails import LLMRails, RailsConfig

# logger = logging.getLogger(__name__)


# class GuardrailsService:
#     # Must match the text of `define bot refuse to respond` in your config/*.co
#     REFUSAL_MESSAGE = (
#         "I'm sorry, I cannot assist with this topic. "
#         "Please ask a question related to the supported scope of this RAG system."
#     )
#     MAX_QUERY_CHARS = 2000

#     def __init__(self):
#         config_path = Path(__file__).parent / "config"
#         config = RailsConfig.from_path(str(config_path))
#         self.rails = LLMRails(config)

#     @staticmethod
#     def _result(allowed, query=None, reason=None, blocked_by=None):
#         return {
#             "allowed": allowed,
#             "query": query,
#             "reason": reason,
#             "blocked_by": blocked_by or [],
#         }

#     @staticmethod
#     def _content(response) -> str:
#         # With `options`, generate_async returns a GenerationResponse (not a dict);
#         # the messages are in response.response.
#         messages = getattr(response, "response", None) or []
#         if messages and isinstance(messages[0], dict):
#             return (messages[0].get("content") or "").strip()
#         return ""

#     @staticmethod
#     def _blocked_rails(response) -> list:
#         log = getattr(response, "log", None)
#         rails = getattr(log, "activated_rails", None) or []
#         return [r.name for r in rails if getattr(r, "stop", False)]

#     async def validate_query(self, query: str) -> dict:
#         """Run INPUT rails only (prompt injection / off-topic) on the user query."""
#         query = (query or "").strip()
#         if not query:
#             return self._result(False, reason="Empty query")
#         if len(query) > self.MAX_QUERY_CHARS:
#             return self._result(False, reason="Query too long")

#         try:
#             response = await self.rails.generate_async(
#                 messages=[{"role": "user", "content": query}],
#                 options={"rails": ["input"], "log": {"activated_rails": True}},
#             )
#         except Exception:
#             logger.exception("Guardrails input check failed")
#             # Fail closed: don't let unchecked queries reach the RAG pipeline.
#             return self._result(False, reason="Guardrails check failed")

#         blocked_by = self._blocked_rails(response)
#         content = self._content(response)

#         if blocked_by or content == self.REFUSAL_MESSAGE:
#             return self._result(
#                 False,
#                 reason="Restricted topic or unsafe input",
#                 blocked_by=blocked_by,
#             )

#         # With input-only rails the returned content is the (possibly rewritten,
#         # e.g. PII-masked) user message, not an LLM answer.
#         return self._result(True, query=content or query)

#     async def validate_response(self, query: str, answer: str) -> dict:
#         """Optional: run OUTPUT rails on the final RAG answer (catches leaked
#         system prompt or injected instructions coming from retrieved documents)."""
#         try:
#             response = await self.rails.generate_async(
#                 messages=[
#                     {"role": "user", "content": query},
#                     {"role": "assistant", "content": answer},
#                 ],
#                 options={"rails": ["output"], "log": {"activated_rails": True}},
#             )
#         except Exception:
#             logger.exception("Guardrails output check failed")
#             return self._result(False, reason="Guardrails check failed")

#         blocked_by = self._blocked_rails(response)
#         content = self._content(response)

#         if blocked_by or content == self.REFUSAL_MESSAGE:
#             return self._result(False, reason="Unsafe response", blocked_by=blocked_by)
#         return self._result(True, query=content or answer)


# @lru_cache(maxsize=1)
# def get_guardrails_service() -> GuardrailsService:
#     """Create once (loading rails is slow) and reuse, e.g. as a FastAPI dependency."""
#     return GuardrailsService()

############## v3


import os

# Set before importing NeMo Guardrails.
os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

import logging
from functools import lru_cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)


class GuardrailsService:
    MAX_QUERY_CHARS = 2000

    def __init__(self):
        config_path = Path(__file__).parent / "config"
        config = RailsConfig.from_path(str(config_path))
        self.rails = LLMRails(config)

    @staticmethod
    def _result(
        allowed,
        query=None,
        answer=None,
        reason=None,
        blocked_by=None,
    ):
        return {
            "allowed": allowed,
            "query": query,
            "answer": answer,
            "reason": reason,
            "blocked_by": blocked_by or [],
        }

    @staticmethod
    def _messages(response):
        return getattr(response, "response", None) or []

    @classmethod
    def _content(cls, response):
        messages = cls._messages(response)

        if messages and isinstance(messages[0], dict):
            return (messages[0].get("content") or "").strip()

        return ""

    @staticmethod
    def _blocked_rails(response):
        log = getattr(response, "log", None)
        rails = getattr(log, "activated_rails", None) or []

        return [
            rail.name
            for rail in rails
            if getattr(rail, "stop", False)
        ]

    async def validate_query(self, query: str) -> dict:
        """Validate and sanitize the incoming RAG query."""

        query = (query or "").strip()

        if not query:
            return self._result(
                False,
                reason="Empty query",
            )

        if len(query) > self.MAX_QUERY_CHARS:
            return self._result(
                False,
                reason="Query too long",
            )

        try:
            response = await self.rails.generate_async(
                messages=[
                    {"role": "user", "content": query}
                ],
                options={
                    "rails": ["input"],
                    "log": {"activated_rails": True},
                },
            )

            blocked_by = self._blocked_rails(response)
            sanitized_query = self._content(response)

            if blocked_by:
                return self._result(
                    False,
                    reason="Restricted topic or unsafe input",
                    blocked_by=blocked_by,
                )

            if not sanitized_query:
                return self._result(
                    False,
                    reason="Empty response from input rails",
                )

            return self._result(
                True,
                query=sanitized_query,
            )

        # except Exception:
            # logger.exception("Guardrails input validation failed")

            # return self._result(
            #     False,
            #     reason="Guardrails check failed",
            # )
        except Exception as exc:
            logger.exception("Guardrails input validation failed")

            print(f"\nGUARDRAILS ERROR: {type(exc).__name__}")
            print(f"ERROR DETAILS: {exc}\n")

            raise

    async def validate_response(
        self,
        query: str,
        answer: str,
    ) -> dict:
        """Validate the final RAG-generated answer."""

        if not answer or not answer.strip():
            return self._result(
                False,
                reason="Empty answer",
            )

        try:
            response = await self.rails.generate_async(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ],
                options={
                    "rails": ["output"],
                    "log": {"activated_rails": True},
                },
            )

            blocked_by = self._blocked_rails(response)
            validated_answer = self._content(response)

            if blocked_by or not validated_answer:
                return self._result(
                    False,
                    reason="Unsafe response",
                    blocked_by=blocked_by,
                )

            return self._result(
                True,
                answer=validated_answer,
            )

        except Exception:
            logger.exception("Guardrails output validation failed")

            return self._result(
                False,
                reason="Guardrails check failed",
            )


@lru_cache(maxsize=1)
def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService()
