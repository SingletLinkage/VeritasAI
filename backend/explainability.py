"""
Explainability & Accessibility Module - "Explain Like I'm 60"
Provides simple, respectful explanations for users with limited digital literacy
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


# === Models ===

class SimpleExplanation(BaseModel):
    """Simple, accessible explanation of the verdict"""
    greeting: str = Field(description="Respectful greeting (e.g., 'Dear Uncle/Aunty')")
    simple_verdict: str = Field(description="One-line verdict in simple words")
    explanation: str = Field(description="Easy explanation in 2-3 short sentences, grade-5 reading level")
    what_to_do: str = Field(description="Clear action steps in simple language")
    why_matters: str = Field(description="Why this matters in everyday terms")
    language: str = Field(default="en", description="Language code for the explanation")


# === Explainability Agent ===

class ExplainabilityAgent:
    """
    Converts technical verdicts into simple, accessible explanations
    for older users with limited digital literacy
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3  # Slightly creative for natural language
        )
    
    def generate_simple_explanation(
        self,
        verdict: str,
        confidence: float,
        reasoning: str,
        red_flags: List[str],
        recommendation: str,
        claim: str,
        target_language: str = "en"
    ) -> SimpleExplanation:
        """
        Generate simple explanation for older users
        
        Args:
            verdict: Technical verdict (LIKELY_TRUE, LIKELY_FALSE, etc.)
            confidence: Confidence score (0-1)
            reasoning: Technical reasoning
            red_flags: List of red flags
            recommendation: Technical recommendation
            claim: The original claim
            target_language: Target language for explanation
            
        Returns:
            SimpleExplanation with accessible content
        """
        
        # Determine appropriate greeting based on language
        greeting_map = {
            "en": "Dear Uncle/Aunty",
            "hi": "प्रिय अंकल/आंटी",
            "es": "Estimado Tío/Tía",
            "fr": "Cher Oncle/Tante",
            "de": "Lieber Onkel/Liebe Tante",
            "pt": "Querido Tio/Tia",
            "ar": "عزيزي العم/العمة",
            "zh": "亲爱的叔叔/阿姨",
            "ja": "おじさん/おばさんへ",
            "ko": "삼촌/이모님께"
        }
        
        # Create prompt for simple explanation
        prompt = f"""
You are helping an older person (60+) understand if a message they received is true or false.
They may have limited digital literacy and English may not be their first language.

CLAIM THEY SAW: "{claim}"

TECHNICAL VERDICT: {verdict} (Confidence: {confidence:.0%})
REASONING: {reasoning}
RED FLAGS: {', '.join(red_flags) if red_flags else 'None'}
RECOMMENDATION: {recommendation}

YOUR TASK:
Generate a simple, respectful explanation following these rules:

1. **GREETING**: Use "{greeting_map.get(target_language, 'Dear Uncle/Aunty')}"

2. **SIMPLE VERDICT**: 
   - One short sentence
   - Use words like: "This is FALSE", "This is TRUE", "We're not sure", "This is misleading"
   - No technical terms

3. **EXPLANATION** (2-3 sentences):
   - Grade-5 reading level
   - Short sentences (max 15 words each)
   - No jargon or technical words
   - Use everyday language
   - Be respectful and warm

4. **WHAT TO DO** (Clear action):
   - Start with "Please..."
   - Simple, specific steps
   - What to do, what NOT to do

5. **WHY IT MATTERS**:
   - Why should they care?
   - Real-world impact
   - Relatable examples

TONE:
- Respectful and warm (like talking to your elder)
- Patient and kind
- Not condescending
- Clear and direct

TARGET LANGUAGE: {target_language}

If target language is not English:
- Generate explanation in that language
- Use culturally appropriate terms
- Keep the warm, respectful tone

DO NOT:
- Use technical terms like "misinformation", "fact-check", "verification"
- Use complex sentences
- Sound robotic or formal
- Be dismissive or condescending

Example style:
"Dear Aunty, this message is FALSE. The government did not announce this scheme. Someone made it up to trick people. Please do not share this message with your family. If you see messages asking for money or bank details, always check with your children first. This helps keep your money safe."
"""

        # Generate structured explanation
        result = self.llm.with_structured_output(SimpleExplanation).invoke(prompt)
        
        return result
    
    def translate_explanation(self, explanation: SimpleExplanation, target_language: str) -> SimpleExplanation:
        """
        Translate an existing explanation to another language
        
        Args:
            explanation: Existing SimpleExplanation
            target_language: Target language code
            
        Returns:
            Translated SimpleExplanation
        """
        
        if explanation.language == target_language:
            return explanation
        
        prompt = f"""
Translate this simple explanation to {target_language}.

KEEP THE SAME:
- Respectful, warm tone (like talking to an elder)
- Simple language (grade-5 level)
- Short sentences
- Cultural respect

ORIGINAL:
Greeting: {explanation.greeting}
Verdict: {explanation.simple_verdict}
Explanation: {explanation.explanation}
What to do: {explanation.what_to_do}
Why it matters: {explanation.why_matters}

Translate to {target_language} while maintaining the respectful, simple tone.
Use culturally appropriate greetings for older people.
"""
        
        result = self.llm.with_structured_output(SimpleExplanation).invoke(prompt)
        result.language = target_language
        
        return result


# === Convenience Functions ===

def explain_simply(
    verdict: str,
    confidence: float,
    reasoning: str,
    red_flags: List[str],
    recommendation: str,
    claim: str,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Quick function to generate simple explanation
    
    Args:
        verdict: Technical verdict
        confidence: Confidence score
        reasoning: Technical reasoning
        red_flags: List of red flags
        recommendation: Technical recommendation
        claim: Original claim
        language: Target language
        
    Returns:
        Dictionary with simple explanation
    """
    
    agent = ExplainabilityAgent()
    
    explanation = agent.generate_simple_explanation(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        red_flags=red_flags,
        recommendation=recommendation,
        claim=claim,
        target_language=language
    )
    
    return {
        "greeting": explanation.greeting,
        "simple_verdict": explanation.simple_verdict,
        "explanation": explanation.explanation,
        "what_to_do": explanation.what_to_do,
        "why_matters": explanation.why_matters,
        "language": explanation.language
    }


def format_simple_explanation(explanation: Dict[str, Any]) -> str:
    """
    Format simple explanation as readable text
    
    Args:
        explanation: Simple explanation dictionary
        
    Returns:
        Formatted text
    """
    
    return f"""
{explanation['greeting']},

{explanation['simple_verdict']}

{explanation['explanation']}

{explanation['what_to_do']}

{explanation['why_matters']}
""".strip()


# === Testing ===

if __name__ == "__main__":
    # Test the explainability agent
    
    print("=" * 70)
    print("EXPLAINABILITY AGENT TEST")
    print("=" * 70)
    
    # Test case: False claim
    test_explanation = explain_simply(
        verdict="LIKELY_FALSE",
        confidence=0.92,
        reasoning="Multiple credible sources including WHO and FactCheck.org contradict this claim. NASA has not made any official statement about hot water curing COVID-19. This appears to be a fabricated claim using NASA's name for credibility.",
        red_flags=[
            "Misattribution to NASA",
            "No scientific evidence supporting hot water cure",
            "Contradicted by WHO guidelines"
        ],
        recommendation="Verify with official NASA and WHO sources. Do not share unverified medical claims.",
        claim="NASA confirmed that drinking hot water cures COVID-19 in 5 days",
        language="en"
    )
    
    print("\n" + "=" * 70)
    print("ENGLISH EXPLANATION")
    print("=" * 70)
    print(format_simple_explanation(test_explanation))
    
    # Test Hindi translation
    print("\n" + "=" * 70)
    print("TESTING HINDI TRANSLATION")
    print("=" * 70)
    
    agent = ExplainabilityAgent()
    english_exp = SimpleExplanation(**test_explanation)
    hindi_exp = agent.translate_explanation(english_exp, "hi")
    
    print(f"Greeting: {hindi_exp.greeting}")
    print(f"Verdict: {hindi_exp.simple_verdict}")
    print(f"Explanation: {hindi_exp.explanation}")
    print(f"What to do: {hindi_exp.what_to_do}")
    print(f"Why matters: {hindi_exp.why_matters}")
