"""
Judge Agent — LLM-as-a-Judge for the Multi-Agent Debate (MAD) System

The Judge Agent is a neutral meta-evaluator that:
  1. Receives the full debate transcript from both Advocate agents
  2. Weighs both arguments independently (without bias toward either side)
  3. Issues a final verdict — and can OVERRIDE the advocates' consensus
  4. Provides a structured, explainable decision with confidence and reasoning

This implements the "LLM-as-a-Judge" paradigm described in:
  - Chen et al. (2026)   G-DMAD: multi-agent debate improves reasoning
  - Li et al. (2026)     PhishDebate: multi-agent in cybersecurity

LLM backend: OpenAI GPT-4o (configurable)
Orchestration: LangChain
"""

import os
import logging
import json
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage

from agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Judge Verdict dataclass
# ────────────────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    """Structured output from the Judge Agent."""
    final_decision: str          # "LEGITIMATE" / "ADVERSARIAL" / "SUSPICIOUS"
    risk_level: str              # "LOW" / "MEDIUM" / "HIGH"
    reasoning: str               # Judge's full explanation
    confidence: str              # "high" / "medium" / "low"
    override: bool               # True if judge overrode advocate consensus
    override_reason: str         # Why override happened (empty if no override)
    raw_response: str = field(default="", repr=False)   # LLM raw text (debug)

    def to_dict(self) -> dict:
        return {
            "final_decision": self.final_decision,
            "risk_level": self.risk_level,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "override": self.override,
            "override_reason": self.override_reason,
        }


# ────────────────────────────────────────────────────────────────────
#  System prompt for the Judge
# ────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are the Judge in an adversarial biometric security system.

Your role is that of a NEUTRAL, independent evaluator — NOT an advocate.
You have just received a debate transcript containing arguments from two agents:
  • The Proponent Advocate — argued the image is LEGITIMATE
  • The Opponent Advocate — argued the image is ADVERSARIAL

You must also consider the raw technical output from the Dynamic Threshold Engine.

Your job:
1. Weigh BOTH sides of the debate fairly and critically
2. Cross-check the arguments against the technical evidence (CNN scores, threshold)
3. Issue a single final verdict: LEGITIMATE, SUSPICIOUS, or ADVERSARIAL
4. Explicitly state whether you are OVERRIDING the advocate majority recommendation
   (override = True if your decision differs from the stronger advocate's conclusion)
5. Provide clear, explainable reasoning — your response will be shown to a human reviewer

Guidelines:
- Prioritize technical evidence (anomaly scores, threshold values) over rhetorical arguments
- If evidence is ambiguous, prefer SUSPICIOUS over LEGITIMATE (security-first principle)
- Never fabricate or infer evidence not present in the provided data
- Be concise and technically grounded
- A low confidence score from the CNN (<0.55) should raise suspicion even without other signals
- A computed threshold ≥ 0.50 should strongly bias toward SUSPICIOUS or ADVERSARIAL

Format your response EXACTLY as follows (each field on its own line):
FINAL_DECISION: <LEGITIMATE / SUSPICIOUS / ADVERSARIAL>
RISK_LEVEL: <LOW / MEDIUM / HIGH>
CONFIDENCE: <high / medium / low>
OVERRIDE: <True / False>
OVERRIDE_REASON: <brief reason if override=True, else "N/A">
REASONING: <your full multi-sentence explanation>
"""

JUDGE_HUMAN_TEMPLATE = """
Please evaluate the following debate and issue your final verdict.

{debate_transcript}

{threshold_info}

Provide your structured verdict now.
"""


# ────────────────────────────────────────────────────────────────────
#  JudgeAgent class
# ────────────────────────────────────────────────────────────────────

class JudgeAgent:
    """
    The Judge Agent — neutral meta-evaluator for the Multi-Agent Debate.

    Args:
        model_name  : OpenAI model (default: "gpt-4o" — more capable than 3.5)
        temperature : Low temperature for more deterministic verdicts (0.1)
        api_key     : OpenAI API key (reads OPENAI_API_KEY env var if None)
    """

    def __init__(self,
                 model_name: str = None,
                 temperature: float = 0.1,
                 api_key: str = None):
        self.llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)
        logger.info(f"[JUDGE] Initialized via LLM factory")

    def verdict(self,
                debate_transcript: str,
                threshold_info: str) -> JudgeVerdict:
        """
        Issue a final verdict after reviewing the debate.

        Args:
            debate_transcript : formatted debate from format_debate_for_judge()
            threshold_info    : formatted string from DynamicThresholdEngine.format_for_agent()

        Returns:
            JudgeVerdict dataclass
        """
        messages = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=JUDGE_HUMAN_TEMPLATE.format(
                debate_transcript=debate_transcript,
                threshold_info=threshold_info,
            )),
        ]

        logger.info("[JUDGE] Deliberating on debate transcript...")
        response = self.llm.invoke(messages)
        raw = response.content.strip()
        logger.debug(f"[JUDGE RAW]:\n{raw}")

        return self._parse_verdict(raw)

    # ── Private helpers ──

    def _parse_verdict(self, raw: str) -> JudgeVerdict:
        """Parse the structured LLM verdict into a JudgeVerdict dataclass."""
        lines = raw.splitlines()

        final_decision   = "SUSPICIOUS"    # safe default
        risk_level       = "MEDIUM"
        confidence       = "medium"
        override         = False
        override_reason  = "N/A"
        reasoning_lines  = []
        in_reasoning     = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_reasoning:
                    reasoning_lines.append("")
                continue

            if stripped.startswith("FINAL_DECISION:"):
                in_reasoning = False
                val = stripped[len("FINAL_DECISION:"):].strip().upper()
                if val in ("LEGITIMATE", "ADVERSARIAL", "SUSPICIOUS"):
                    final_decision = val

            elif stripped.startswith("RISK_LEVEL:"):
                in_reasoning = False
                val = stripped[len("RISK_LEVEL:"):].strip().upper()
                if val in ("LOW", "MEDIUM", "HIGH"):
                    risk_level = val

            elif stripped.startswith("CONFIDENCE:"):
                in_reasoning = False
                confidence = stripped[len("CONFIDENCE:"):].strip().lower()

            elif stripped.startswith("OVERRIDE:"):
                in_reasoning = False
                val = stripped[len("OVERRIDE:"):].strip()
                override = val.lower() in ("true", "yes", "1")

            elif stripped.startswith("OVERRIDE_REASON:"):
                in_reasoning = False
                override_reason = stripped[len("OVERRIDE_REASON:"):].strip()

            elif stripped.startswith("REASONING:"):
                in_reasoning = True
                rest = stripped[len("REASONING:"):].strip()
                if rest:
                    reasoning_lines.append(rest)

            elif in_reasoning:
                reasoning_lines.append(stripped)

        reasoning = " ".join(r for r in reasoning_lines if r).strip() or raw

        # Override reason should be empty string (not "N/A") when no override
        if not override:
            override_reason = ""

        judge_verdict = JudgeVerdict(
            final_decision=final_decision,
            risk_level=risk_level,
            reasoning=reasoning,
            confidence=confidence,
            override=override,
            override_reason=override_reason,
            raw_response=raw,
        )
        logger.info(
            f"[JUDGE] Verdict: {final_decision} | Risk: {risk_level} | "
            f"Override: {override} | Confidence: {confidence}"
        )
        return judge_verdict
