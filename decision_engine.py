import json
import asyncio
import random
import logging
from preprocessing import TextPreprocessor
from faiss_indexer import VectorKnowledgeIndexer
from intent_classifier import ScikitIntentClassifier

# Pointing to the local Rasa server's NLU parsing API
RASA_URL = "http://localhost:5005/model/parse"

class DecisionEngine:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
        self.faiss_db = VectorKnowledgeIndexer()
        
        self.intent_classifier = ScikitIntentClassifier()
        self.intent_classifier.load_model()
        
        # Hardcoding the domain responses into memory for 0ms latency retrieval
        # In a massive system, load this from domain.yml dynamically.
        self.responses = {
            "greet": [
                "Hi! I'm the Nivetix AI Assistant. I can help you with our services, pricing, portfolio, or connect you with our strategy team. How can I help?",
                "Hello! Welcome to Nivetix. Are you looking to build software, run an AI automation, or inquire about our costs?"
            ],
            "ask_price": [
                "Our pricing is heavily customized based on scale, but generally:\n\n- Web Platforms: Starting ~$399\n- AI Chatbot Deployments: Starting ~$329\n- Custom SaaS: Starting ~$899\n\nWould you like a quote?",
                "We deal in premium enterprise systems. A simple business site begins around ₹29,999 ($399), while large custom platforms scale upwards. Want an exact breakdown?"
            ],
            "ask_services": [
                "We are a full-stack digital agency! We primarily build:\n\n1. AI Chatbots & Workflows\n2. Next.js SaaS Platforms\n3. Enterprise Dashboards\n4. E-commerce\n\nWhat are you looking to build?",
                "We engineer revenue-driven digital systems! This includes UI/UX mapping, Full-stack Web Development, AI Business Automation, and high-end brand graphics."
            ],
            "ask_contact": [
                "You can reach us directly via:\n- Phone / WhatsApp: +91 8586053408\n- Email: contact@nivetix.software",
                "Need to talk to a human? \nEmail: contact@nivetix.software \nCall/WhatsApp: +91-8586053408 (Mon-Fri, 9am-6pm IST)."
            ],
            "ask_demo": [
                "We've got an amazing portfolio showing our real-world impact. You can view all our Case Studies right on the /portfolio page.",
                "Absolutely! We focus on delivering top 1% designs. Head over to our Portfolio section to see our past UI/UX, SaaS tools, and AI deployments."
            ],
            "out_of_scope": [
                "I'm exclusively programmed to discuss Nivetix Agency's services, pricing, and project capabilities. Can I help you build some software instead?"
            ],
            "affirm_thanks": [
                "You're very welcome! Let me know if there's anything else I can help you with.",
                "Happy to help! Have a great day.",
                "Anytime! Feel free to reach out if you need anything else.",
                "You got it! We're here whenever you need us."
            ]
        }

    async def get_intent(self, text: str):
        return self.intent_classifier.predict(text)

    async def process_message(self, text: str, history: list) -> str:
        # Preprocess User Syntax (Fix spelling strings, map synonyms)
        cleaned_text = self.preprocessor.clean(text)
        
        # Get Intent matrix confidence strictly on the immediate query first
        intent, confidence = await self.get_intent(cleaned_text)

        # Context Ingestion Memory Routing
        # If the standalone short query fails to trigger high confidence, append previous contextual topics
        if confidence < 0.70 and len(cleaned_text.split()) < 3 and history:
            context = history[-1]
            contextualized_text = f"{context} {cleaned_text}"
            
            # Re-evaluate intent against the contextualized string
            intent, confidence = await self.get_intent(contextualized_text)
            cleaned_text = contextualized_text # Inherit context for subsequent FAISS retrieval if it still fails

        # ---------------------------------------------
        # THE DECISION: Template vs FAISS Generation
        # ---------------------------------------------
        if confidence >= 0.70 and intent in self.responses:
            return random.choice(self.responses[intent])
        
        # Fallback to FAISS Document Semantic Search retrieval
        results = self.faiss_db.search(cleaned_text, top_k=2)
        if results and "Awaiting indexing." not in results[0]:
            ans = "\n\n".join(results)
            templates = [
                f"Here's what I found regarding your question:\n{ans}",
                f"Based on our knowledge base, this might help:\n{ans}",
                f"Got it! Here is the relevant information:\n{ans}"
            ]
            return random.choice(templates)
            
        return random.choice(self.responses["out_of_scope"])
