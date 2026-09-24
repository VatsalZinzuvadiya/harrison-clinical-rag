"""
Clinical LLM Generator & Prompt Engineering Module

Enforces zero-hallucination clinical grounding rules based on Harrison's textbook excerpts.
Supports multi-model failover across Groq Cloud API (openai/gpt-oss-120b, qwen/qwen3.8-27b)
and local Ollama runtime.
"""

from typing import List, Dict, Any, Optional
import httpx

from app.core.config import config
from app.core.exceptions import LLMGenerationError

CLINICAL_SYSTEM_PROMPT = """You are a senior physician decision-support assistant operating strictly on Harrison's Principles of Internal Medicine.

STRICT CLINICAL GROUNDING & ZERO-HALLUCINATION RULES:
1. Answer strictly using ONLY the retrieved textbook excerpts provided. Never insert outside medical assumptions or speculative diagnoses.
2. CITE specific page numbers for EVERY claim, symptom, risk score, diagnostic test, or clinical finding using: [Page X] or [Page X, Page Y].
3. If the provided excerpts do NOT contain sufficient information for a specific clinical aspect, explicitly state: "The provided excerpts from Harrison's textbook do not contain sufficient details for this evaluation."
4. Structure your response into 4 comprehensive, production-grade clinical sections:

   ### 1. Clinical Overview & Priority Differential
   - Concise clinical summary matching patient presentation against Harrison's criteria.
   - Categorize differential diagnoses into **Emergent / Life-Threatening** vs **Non-Emergent / Outpatient** conditions.

   ### 2. Comparative Clinical Features & Pathophysiology
   - Render a detailed, multi-column Markdown Table comparing all differential conditions:
     | Condition | Onset & Duration | Pain Quality & Location | Key Physical Exam & Biomarkers | Page Citation |
     |-----------|------------------|-------------------------|--------------------------------|---------------|

   ### 3. Diagnostic Risk Stratification & Clinical Workup
   - Detail clinical decision scores or risk criteria mentioned in excerpts (e.g. HEART Score, EDACS Score, troponin thresholds, D-dimer, ECG findings).
   - Outline recommended diagnostic evaluations step-by-step with exact page citations.

   ### 4. Grounded Initial Management Considerations
   - Summarize initial diagnostic or therapeutic steps explicitly stated in the source text.

5. Mandatory Safety Disclaimer: Always end with:
   "Note: This clinical decision support tool provides information grounded strictly in textbook excerpts from Harrison's Principles of Internal Medicine to assist healthcare providers; it does not replace clinical judgment or definitive diagnostic testing."
"""


class LocalClinicalGenerator:
    """
    Generator communicating with Groq API and Ollama local server with failover.
    """

    def __init__(self, ollama_url: str = config.OLLAMA_BASE_URL,
                 model_name: str = config.OLLAMA_MODEL,
                 temperature: float = config.LLM_TEMPERATURE):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature

    def check_llm_status(self) -> Dict[str, Any]:
        if config.GROQ_API_KEY:
            return {"available": True, "provider": "Groq Cloud API", "models": [config.GROQ_MODEL]}
        return self.check_ollama_status()

    def check_ollama_status(self) -> Dict[str, Any]:
        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                return {"available": True, "provider": "Ollama Local", "models": models}
        except Exception:
            pass
        return {"available": False, "provider": "None", "models": []}

    def _build_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_str_parts = []
        for idx, c in enumerate(context_chunks, start=1):
            pages_str = ", ".join(str(p) for p in c.get("page_numbers", []))
            context_str_parts.append(
                f"--- EXCERPT [{idx}] (Page(s): {pages_str}) ---\n"
                f"Section: {c.get('heading_trail', 'General')}\n"
                f"Text: {c.get('text', '').strip()}\n"
            )

        full_context = "\n".join(context_str_parts)
        user_prompt = (
            f"RETRIEVED TEXTBOOK EXCERPTS:\n{full_context}\n\n"
            f"CLINICAL QUESTION / PATIENT PRESENTATION:\n{query}\n\n"
            f"Instructions: Generate a well-cited, clinically structured response based STRICTLY on the excerpts above."
        )
        return user_prompt

    def _generate_with_groq(self, query: str, context_chunks: List[Dict[str, Any]]) -> Optional[str]:
        api_key = config.GROQ_API_KEY
        if not api_key:
            return None

        user_prompt = self._build_user_prompt(query, context_chunks)
        candidate_models = [config.GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        seen = set()
        models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": config.LLM_MAX_TOKENS
            }

            try:
                print(f"[LocalClinicalGenerator] Invoking Groq Cloud API ({model_name})...")
                response = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=35.0
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        if content:
                            print(f"[LocalClinicalGenerator] Successfully generated clinical answer via Groq API ({model_name}).")
                            return content
                else:
                    print(f"[LocalClinicalGenerator] Groq API ({model_name}) returned HTTP {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[LocalClinicalGenerator] Error calling Groq API ({model_name}): {e}")

        return None

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return (
                "The provided excerpts from Harrison's Principles of Internal Medicine do not contain "
                "sufficient information to answer this query.\n\n"
                "Note: This clinical decision support tool provides information grounded strictly in textbook "
                "excerpts to assist healthcare providers; it does not replace clinical judgment or definitive diagnostic testing."
            )

        groq_result = self._generate_with_groq(query, context_chunks)
        if groq_result:
            return groq_result

        user_prompt = self._build_user_prompt(query, context_chunks)
        status = self.check_ollama_status()
        if not status["available"]:
            return self._fallback_grounded_summary(query, context_chunks, ollama_missing=True)

        payload = {
            "model": self.model_name,
            "system": CLINICAL_SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": config.LLM_MAX_TOKENS
            }
        }

        try:
            response = httpx.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60.0)
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    return result
            return self._fallback_grounded_summary(query, context_chunks, error_msg=f"HTTP {response.status_code}")
        except Exception as e:
            return self._fallback_grounded_summary(query, context_chunks, error_msg=str(e))

    def _fallback_grounded_summary(self, query: str, context_chunks: List[Dict[str, Any]],
                                   ollama_missing: bool = False, error_msg: str = "") -> str:
        notice = ""
        if ollama_missing:
            notice = (
                "> [!NOTE]\n"
                "> **Local Ollama LLM is not currently running**. (Run `ollama serve` and `ollama run llama3.2` to enable full local LLM generation).\n"
                "> Below is a direct grounded synthesis from Harrison's textbook passages:\n\n"
            )

        output_parts = [notice, "### **Likely Differential Considerations & Clinical Findings**\n"]

        for idx, c in enumerate(context_chunks[:4], start=1):
            pages_str = ", ".join(str(p) for p in c.get("page_numbers", []))
            heading = c.get("heading_trail", "Harrison's Medical Manual")
            snippet = c.get("text", "").strip()
            output_parts.append(f"- **{heading}** [Page {pages_str}]:\n  _{snippet}_\n")

        output_parts.append("\n### **Relevant Next Steps / Workup**\n")
        output_parts.append(
            "- Evaluate clinical presentation against diagnostic criteria described above [Page " +
            ", ".join(str(p) for p in context_chunks[0].get("page_numbers", [1])) + "].\n"
            "- Perform baseline laboratory testing and imaging indicated in reference sections.\n"
        )
        output_parts.append(
            "\n---\n*Note: This clinical decision support tool provides information grounded strictly in textbook "
            "excerpts to assist healthcare providers; it does not replace clinical judgment or definitive diagnostic testing.*"
        )

        return "\n".join(output_parts)
