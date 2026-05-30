"""
Advocate Agents — Multi-Agent Debate (MAD) System

Two advocate agents present opposing arguments about an input image:
  - Advocate A (Proponent)  : argues the image is LEGITIMATE / clean
  - Advocate B (Opponent)   : argues the image is ADVERSARIAL / malicious

Both agents receive the same structured CNN + threshold output and produce
natural-language arguments. These arguments are forwarded to the Judge agent.

LLM backend: OpenAI GPT-4 / Claude (configurable via .env)
Orchestration: LangChain
"""

import os
import logging
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage

from agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Advocate output dataclass
# ────────────────────────────────────────────────────────────────────

@dataclass
class AdvocateOutput:
    role: str           # "proponent" or "opponent"
    stance: str         # "legitimate" or "adversarial"
    argument: str       # full natural-language argument
    risk_assessment: str
    confidence: str     # agent's self-reported confidence: high / medium / low
    key_evidence: list[str]


# ────────────────────────────────────────────────────────────────────
#  System prompts
# ────────────────────────────────────────────────────────────────────

PROPONENT_SYSTEM = """You are the Proponent Advocate in an adversarial biometric security system.

Your role is to argue that the analyzed facial image is LEGITIMATE and NOT adversarially manipulated.

You will be given structured outputs from a CNN-based adversarial detection model and a dynamic threshold engine.
Analyze ALL provided evidence and construct a strong, well-reasoned argument defending the legitimacy of the input.

Guidelines:
- Focus on evidence that supports authenticity (e.g., high confidence scores, low anomaly, low drift)
- Acknowledge weaknesses in your argument honestly — do not fabricate evidence
- Be concise but technically grounded
- Conclude with your risk assessment: LOW / MEDIUM / HIGH
- State your confidence in your argument: high / medium / low

Format your response as:
STANCE: LEGITIMATE
ARGUMENT: <your detailed argument>
KEY_EVIDENCE: <bullet points of supporting evidence>
RISK_ASSESSMENT: <LOW / MEDIUM / HIGH>
CONFIDENCE: <high / medium / low>
"""

OPPONENT_SYSTEM = """You are the Opponent Advocate in an adversarial biometric security system.

Your role is to argue that the analyzed facial image IS adversarially manipulated and poses a security threat.

You will be given structured outputs from a CNN-based adversarial detection model and a dynamic threshold engine.
Analyze ALL provided evidence and construct a strong, well-reasoned argument that the input is adversarial.

Guidelines:
- Focus on evidence that suggests manipulation (e.g., high anomaly score, low confidence, embedding drift)
- Acknowledge any evidence that contradicts your argument — do not ignore it
- Be concise but technically grounded
- Conclude with your risk assessment: LOW / MEDIUM / HIGH
- State your confidence in your argument: high / medium / low

Format your response as:
STANCE: ADVERSARIAL
ARGUMENT: <your detailed argument>
KEY_EVIDENCE: <bullet points of adversarial indicators>
RISK_ASSESSMENT: <LOW / MEDIUM / HIGH>
CONFIDENCE: <high / medium / low>
"""

ADVOCATE_HUMAN_TEMPLATE = """
Please analyze the following evidence and provide your argument:

{cnn_output}

{threshold_output}

Provide your structured argument now.
"""


# ────────────────────────────────────────────────────────────────────
#  AdvocateAgent class
# ────────────────────────────────────────────────────────────────────

class AdvocateAgent:
    """
    A single advocate agent backed by an LLM.

    Args:
        role        : "proponent" or "opponent"
        model_name  : OpenAI model name (e.g. "gpt-4o", "gpt-3.5-turbo")
        temperature : LLM temperature (lower = more deterministic)
        api_key     : OpenAI API key (reads from OPENAI_API_KEY env if None)
    """

    def __init__(self,
                 role: str = "proponent",
                 model_name: str = None,
                 temperature: float = 0.3,
                 api_key: str = None):
        assert role in ("proponent", "opponent"), "role must be 'proponent' or 'opponent'"
        self.role = role
        self.stance = "legitimate" if role == "proponent" else "adversarial"
        self.system_prompt = PROPONENT_SYSTEM if role == "proponent" else OPPONENT_SYSTEM

        self.llm = get_llm(model_name=model_name, temperature=temperature, api_key=api_key)

    def argue(self, cnn_output: str, threshold_output: str) -> AdvocateOutput:
        """
        Generate an argument given the CNN and threshold evidence.

        Args:
            cnn_output       : formatted string from format_features_for_agent()
            threshold_output : formatted string from DynamicThresholdEngine.format_for_agent()

        Returns:
            AdvocateOutput dataclass
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=ADVOCATE_HUMAN_TEMPLATE.format(
                cnn_output=cnn_output,
                threshold_output=threshold_output,
            )),
        ]

        logger.info(f"[{self.role.upper()} ADVOCATE] Generating argument...")
        response = self.llm.invoke(messages)
        raw = response.content.strip()
        logger.debug(f"[{self.role.upper()} RAW]:\n{raw}")

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> AdvocateOutput:
        """Parse the structured LLM response into an AdvocateOutput."""
        lines = raw.splitlines()
        argument = ""
        key_evidence = []
        risk_assessment = "MEDIUM"
        confidence = "medium"
        in_argument = False
        in_evidence = False

        used_defaults = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_argument:
                    argument += " "
                continue

            # Use partition to handle "KEY : value" and "KEY:value" equally
            key, sep, val = stripped.partition(":")
            key_upper = key.strip().upper()
            val = val.strip()

            if key_upper == "ARGUMENT":
                in_argument = True
                in_evidence = False
                argument = val
            elif key_upper == "KEY_EVIDENCE":
                in_argument = False
                in_evidence = True
            elif key_upper == "RISK_ASSESSMENT":
                in_argument = False
                in_evidence = False
                if val.upper() in ("LOW", "MEDIUM", "HIGH"):
                    risk_assessment = val.upper()
            elif key_upper == "CONFIDENCE":
                in_argument = False
                in_evidence = False
                if val.lower() in ("high", "medium", "low"):
                    confidence = val.lower()
            elif key_upper == "STANCE":
                in_argument = False
                in_evidence = False
            elif in_argument and stripped:
                argument += " " + stripped
            elif in_evidence and (stripped.startswith("-") or stripped.startswith("•")):
                key_evidence.append(stripped.lstrip("-•").strip())

        if not argument:
            used_defaults.append("argument")
        if used_defaults:
            logger.warning(f"[{self.role.upper()}] Parser used defaults for: {used_defaults}")

        return AdvocateOutput(
            role=self.role,
            stance=self.stance,
            argument=argument.strip() or raw,
            risk_assessment=risk_assessment,
            confidence=confidence,
            key_evidence=key_evidence,
        )


# ────────────────────────────────────────────────────────────────────
#  Convenience function: run both advocates in sequence
# ────────────────────────────────────────────────────────────────────

def run_advocates(cnn_output: str,
                  threshold_output: str,
                  model_name: str = None,
                  temperature: float = 0.3,
                  api_key: str = None) -> tuple[AdvocateOutput, AdvocateOutput]:
    """
    Run both advocate agents and return their outputs.

    Args:
        cnn_output       : formatted CNN feature string
        threshold_output : formatted threshold engine string
        model_name       : LLM model name
        temperature      : LLM temperature
        api_key          : OpenAI API key (reads env if None)

    Returns:
        (proponent_output, opponent_output) tuple of AdvocateOutput
    """
    proponent = AdvocateAgent("proponent", model_name, temperature, api_key)
    opponent  = AdvocateAgent("opponent",  model_name, temperature, api_key)

    pro_out = proponent.argue(cnn_output, threshold_output)
    opp_out = opponent.argue(cnn_output, threshold_output)

    return pro_out, opp_out


# ────────────────────────────────────────────────────────────────────
#  Format advocate outputs for Judge input
# ────────────────────────────────────────────────────────────────────

def format_debate_for_judge(pro: AdvocateOutput, opp: AdvocateOutput) -> str:
    """Combine both advocate arguments into a debate transcript for the Judge."""
    evidence_pro = "\n".join(f"  • {e}" for e in pro.key_evidence) or "  (none listed)"
    evidence_opp = "\n".join(f"  • {e}" for e in opp.key_evidence) or "  (none listed)"

    return (
        f"[DEBATE TRANSCRIPT]\n\n"
        f"═══ PROPONENT ADVOCATE (argues: LEGITIMATE) ═══\n"
        f"{pro.argument}\n\n"
        f"Key Evidence:\n{evidence_pro}\n"
        f"Risk Assessment: {pro.risk_assessment}  |  Confidence: {pro.confidence}\n\n"
        f"═══ OPPONENT ADVOCATE (argues: ADVERSARIAL) ═══\n"
        f"{opp.argument}\n\n"
        f"Key Evidence:\n{evidence_opp}\n"
        f"Risk Assessment: {opp.risk_assessment}  |  Confidence: {opp.confidence}\n"
    )
