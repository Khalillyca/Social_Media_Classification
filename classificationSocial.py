import time
import os
from glob import glob
import pandas as pd
from enum import Enum
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# =========================
# 🔐 API
# =========================
api_key = os.getenv("GROQ_API_KEY")
MODEL_ID = "groq/openai/gpt-oss-120b"
client_llama = instructor.from_provider(model=MODEL_ID)

# =========================
# 🌍 COUNTRY MAP
# =========================
country_map = {
    "BEL": "Belgium",
    "FRA": "France",
    "GER": "Germany",
    "ITA": "Italy",
    "NLD": "Netherlands",
    "PRT": "Portugal",
    "UGA": "Uganda",
    "GBR": "UK",
    "DEU": "Germany"
}

# =========================
# 🤖 ENUMS
# =========================
class SentimentLabel(str, Enum):
    negative="negative"
    neutral="neutral"
    positive="positive"

class EmotionLabel(str, Enum):
    anger = "anger"
    frustration = "frustration"
    disappointment = "disappointment"
    anxiety = "anxiety"
    confusion = "confusion"
    fear = "fear"
    sadness = "sadness"
    betrayal = "betrayal"
    relief = "relief"
    satisfaction = "satisfaction"
    joy = "joy"
    gratitude = "gratitude"
    pride = "pride"
    neutral = "neutral"
    
class ReviewMention(str, Enum):
    customer_service = "customer_service"
    customer_communications = "customer_communications"
    service_general = "service_general"
    solution = "solution"
    cancellation = "cancellation"
    payment = "payment"
    staff = "staff"
    refund = "refund"
    location = "location"
    delivery_service = "delivery_service"
    network_coverage = "network_coverage"
    data_speed = "data_speed"
    call_quality = "call_quality"
    pricing_value = "pricing_value"
    plans_bundles = "plans_bundles"
    roaming_international = "roaming_international"
    sim_activation_porting = "sim_activation_porting"
    app_website_experience = "app_website_experience"
    account_login_security = "account_login_security"
    billing_invoicing = "billing_invoicing"
    promotions_discounts = "promotions_discounts"
    fraud_scam_concerns = "fraud_scam_concerns"
    complaint_handling = "complaint_handling"
    other = "other"


class JourneyStage(str, Enum):
    acquisition = "acquisition"
    onboarding_activation = "onboarding_activation"
    everyday_usage = "everyday_usage"
    support_contact = "support_contact"
    payment_billing = "payment_billing"
    cancellation_exit = "cancellation_exit"
    post_exit_refund = "post_exit_refund"
    other = "other"
class IssueType(str, Enum):
    no_issue_pure_praise = "no_issue_pure_praise"
    network_issue = "network_issue"
    product_plan_issue = "product_plan_issue"
    billing_payment_issue = "billing_payment_issue"
    account_login_issue = "account_login_issue"
    app_website_issue = "app_website_issue"
    process_delay_issue = "process_delay_issue"
    staff_behaviour_issue = "staff_behaviour_issue"
    communication_issue = "communication_issue"
    cancellation_refund_issue = "cancellation_refund_issue"
    delivery_logistics_issue = "delivery_logistics_issue"
    other = "other"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    partially_resolved = "partially_resolved"
    unresolved = "unresolved"
    pending = "pending"
    not_applicable = "not_applicable"

class ReviewTone(str, Enum):
    complaint = "complaint"
    compliment = "compliment"
    suggestion = "suggestion"
    question = "question"
    mixed = "mixed"
    other = "other"

class ValueForMoney(str, Enum):
    very_poor = "very_poor"
    poor = "poor"
    fair = "fair"
    good = "good"
    excellent = "excellent"
    not_applicable = "not_applicable"

class ChurnRiskLabel(str, Enum):
    high="high"
    medium="medium"
    low="low"
    not_applicable="not_applicable"

class TrustpilotReviewInsights(BaseModel):
    sentiment_label: SentimentLabel
    sentiment_score: float = Field(..., ge=-1, le=1)
    primary_emotion: EmotionLabel
    primary_mention: ReviewMention
    journey_stage: JourneyStage
    primary_issue_type: IssueType
    resolution_status: ResolutionStatus
    review_tone: ReviewTone
    value_for_money: ValueForMoney
    churn_risk: ChurnRiskLabel

# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = """
You analyze public Trustpilot reviews for a telecom MVNO (Lyca Mobile).

Your job is to read a single review and fill all fields of the
TrustpilotReviewInsights schema:

- sentiment_label: negative | neutral | positive
- sentiment_score: float in [0, 1]
- primary_emotion: one dominant emotion
- primary_mention: one dominant topic / theme
- journey_stage: main stage of the customer journey
- primary_issue_type: main underlying issue (or 'no_issue_pure_praise')
- resolution_status: whether the issue is resolved or not
- review_tone: overall intent of the review
- value_for_money: how the customer feels about price vs value
- churn_risk: how likely they are to leave
- summary: one-sentence plain-English summary

General rules
-------------
- Always base your decisions ONLY on what is clearly implied in the review.
- Do NOT hallucinate specific facts (dates, amounts, names).
- When several labels could apply, choose the MOST IMPORTANT / DOMINANT one.
- Every field must have exactly ONE value from its enum (no lists, no nulls).
- If something is unclear or not mentioned, choose the safest / most neutral label
  (e.g. 'other', 'not_applicable', 'neutral', 'fair', 'medium').

Sentiment
---------
sentiment_label:
- negative: clear complaint, anger, threats to leave, strong dissatisfaction.
- neutral: mostly factual, balanced, or mixed comments without strong emotion.
- positive: praise, strong satisfaction, clear recommendation.

sentiment_score (-1 to 1):
- from -1.0 to -0.80: extremely negative.
- from -0.79 to -0.40: clearly negative.
- from -0.39 to +0.39: weak / mixed / neutral (0 means perfectly neutral).
- from +0.40 to +0.79: clearly positive.
- from +0.80 to +1.00: extremely positive.

The score reflects BOTH direction and intensity:
- negative scores = negative sentiment,
- positive scores = positive sentiment,
- scores near 0 = neutral or very weak sentiment.

Ensure the sign of sentiment_score is consistent with sentiment_label:
- if sentiment_label = "negative", sentiment_score should be < 0;
- if sentiment_label = "neutral", sentiment_score should be close to 0;
- if sentiment_label = "positive", sentiment_score should be > 0.


Primary emotion
---------------
Choose ONE EmotionLabel that best captures the dominant feeling:

Negative:
- anger, frustration, disappointment, anxiety, confusion, fear, sadness, betrayal
  (betrayal = feeling cheated, scammed, or lied to).

Positive:
- relief (finally fixed after problems), satisfaction, joy, gratitude, pride.

Neutral:
- neutral (no clear emotional tone).

If there is both weak annoyance and strong disappointment, pick the stronger
one (e.g. disappointment).

Primary mention (topic)
-----------------------
Choose ONE ReviewMention that best describes what the review is mainly about.

Generic / service:
- customer_service: interaction with agents / support quality.
- customer_communications: emails, SMS, clarity of information, notifications.
- service_general: very generic comments about “service” with no clear detail.
- solution: focus on how the problem was solved (or not).
- cancellation, payment, staff, refund, location, delivery_service.

Telco-specific:
- network_coverage: signal strength, coverage, no service in places.
- data_speed: data speed, throttling, slow internet.
- call_quality: call drops, echo, voice quality.
- pricing_value: value for money, price vs benefits.
- plans_bundles: bundle structure, allowances, fairness of plans.
- roaming_international: roaming, international usage, EU usage.
- sim_activation_porting: activation delays, number porting issues.
- app_website_experience: usability or bugs in app/website.
- account_login_security: login, password, security, OTP issues.
- billing_invoicing: bills, overcharges, unexpected fees.
- promotions_discounts: promo codes, discounts, special offers.
- fraud_scam_concerns: scams, suspicious calls, fraud experiences.
- complaint_handling: formal complaints, escalation handling.
- other: none of the above fits well.
IMPORTANT ENUM SAFETY RULES
---------------------------
- primary_mention MUST ALWAYS be chosen ONLY from the ReviewMention enum.
- NEVER use IssueType values (e.g. delivery_logistics_issue, billing_payment_issue)
  as primary_mention.
- If the issue is about SIM or delivery problems:
  - Use primary_mention = delivery_service
  - Use primary_issue_type = delivery_logistics_issue


Customer journey stage
----------------------
JourneyStage:
- acquisition: marketing, sign-up decision, first impression before use.
- onboarding_activation: SIM delivery, activation, number porting, first setup.
- everyday_usage: regular usage of calls, data, texts, roaming.
- support_contact: interacting with support/helpdesk (chat, email, call).
- payment_billing: paying, top-ups, invoices, auto-renewal.
- cancellation_exit: leaving the service, switching providers, closing account.
- post_exit_refund: refunds or issues after leaving.
- other: unclear or mixed.

Primary issue type
------------------
IssueType:
- no_issue_pure_praise: purely positive praise with no real problem.
- network_issue: coverage, outages, network instability.
- product_plan_issue: wrong plan, allowances, plan design, hidden limits.
- billing_payment_issue: charges, refunds, payment failures, overbilling.
- account_login_issue: login, account access, password, security codes.
- app_website_issue: website or app bugs, poor UX, technical errors.
- process_delay_issue: very long waiting times, delays in handling.
- staff_behaviour_issue: rude staff, unhelpful agents, attitude.
- communication_issue: misleading or unclear information, fine print issues.
- cancellation_refund_issue: difficulty cancelling, contract lock-in, refunds.
- delivery_logistics_issue: SIM or product delivery problems, courier issues.
- other: something different or unclear.

Resolution status
-----------------
ResolutionStatus:
- resolved: the review clearly says the issue is solved.
- partially_resolved: some progress but not fully resolved.
- unresolved: the issue is still not fixed.
- pending: the customer is waiting for a response or outcome.
- not_applicable: no specific problem to resolve (pure praise or general comment).

Review tone
-----------
ReviewTone:
- complaint: mainly to complain or warn others.
- compliment: mainly to praise or thank the company/staff.
- suggestion: mainly advice or ideas for improvement.
- question: mainly asking questions or seeking clarification.
- mixed: clearly includes both strong praise and strong complaints.
- other: does not fit any of the above.

Value for money
---------------
ValueForMoney:
- very_poor: feels totally ripped off, extremely bad value.
- poor: bad value, too expensive for what they get.
- fair: average, acceptable but not amazing.
- good: clearly happy with price vs value.
- excellent: extremely happy with pricing and value.
- not_applicable: price/value is not discussed or cannot be inferred.

Churn risk
----------
ChurnRiskLabel:
- high: clear statements about leaving / switching, or very strong persistent
  dissatisfaction (“I will never use them again”, “last time with Lyca”).
- medium: unhappy and may leave, but not explicitly decided.
- low: generally satisfied, minor issues only.
- not_applicable: cannot reasonably judge (e.g. generic remark or ex-customer
  already left long ago and is just describing history).

Summary
-------
summary:
- One concise sentence in plain English.
- Capture who the customer is (if implied), what happened, and how they feel.
- Do not add marketing language or apologise; just describe.

## Competitor Praise Note
If the review primarily praises or recommends a competitor instead of Lyca Mobile,
treat this as negative sentiment toward Lyca unless Lyca is also clearly praised.
Classify all fields from Lyca’s perspective.

In such cases, the summary must clearly state that the customer is praising a
competitor and implicitly or explicitly comparing Lyca unfavorably.
Do not invent competitor details beyond what is mentioned in the review.

Output format
-------------
You MUST produce values that exactly match the enums defined in the
TrustpilotReviewInsights Pydantic model and obey all constraints above.
Do NOT output lists or arrays for any field; each field must be a single value.
"""

def build_user_message(review_text: str):
    return {
        "role": "user",
        "content": f"[REVIEW]\n{review_text}"
    }

# =========================
# RETRY CLASSIFIER
# =========================
def classify_ticket(ticket_text: str):

    for attempt in range(3):

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                build_user_message(ticket_text),
            ]

            resp = client_llama.chat.completions.create(
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                response_model=TrustpilotReviewInsights,
            )

            time.sleep(0.5)
            return resp

        except Exception:
            time.sleep(1.5)
            continue

    return None

# =========================
# INPUT / OUTPUT
# =========================
input_folder = r"C:\Users\Moha8550\OneDrive - Lyca Group\Desktop\SocialMedia_datapreprocessing_folder\countrywise_output_message_only"
output_folder = r"C:\Users\Moha8550\OneDrive - Lyca Group\Desktop\SocialMedia_datapreprocessing_folder\countrywisetrustpilot_output_message_only"
os.makedirs(output_folder, exist_ok=True)

files = glob(f"{input_folder}\\*.xlsx")

d2_llm = {}

for file in files:

    print("\nProcessing:", file)

    filename = os.path.basename(file)
    raw = filename.split("(")[0].strip()

    if len(raw) == 3:
        country = country_map.get(raw.upper(), raw)
    else:
        country = raw.title()

    df = pd.read_excel(file)

    rows = []

    for _, row in df.iterrows():

        message = str(row.get("message") or row.get("Message")).strip()

        if not message or message.lower() == "nan":
            continue

        ai = classify_ticket(message)

        if ai is None:
            print("Skipped:", message[:40])
            continue

        rows.append({
            "country": country,
            "platform": row.get("platform") or row.get("Media Type"),
            "title": row.get("title") or row.get("Title"),
            "message": message,
            "link": row.get("link") or row.get("Link"),
            "created_date": row.get("created_date") or row.get("Publish Date"),
            "language": row.get("language") or row.get("Language"),
            "username": row.get("username") or row.get("User Name"),
            "gender": row.get("gender") or row.get("Gender"),
            "user_rating": row.get("user_rating") or row.get("Star Rating"),

            "sentiment": ai.sentiment_label.value,   # 🔴 UPDATED NAME
            "sentiment_score": ai.sentiment_score,
            "emotion": ai.primary_emotion.value,
            "primary_mention": ai.primary_mention.value,
            "journey_stage": ai.journey_stage.value,
            "issue_type": ai.primary_issue_type.value,
            "resolution_status": ai.resolution_status.value,
            "review_tone": ai.review_tone.value,
            "value_for_money": ai.value_for_money.value,
            "churn_risk": ai.churn_risk.value,
        })

    final_df = pd.DataFrame(rows)

    d2_llm[country] = final_df

    final_df.to_excel(f"{output_folder}\\{country}_trustpilot_llm.xlsx", index=False)

    print(f"✅ {country} Done")

print("\n🎉 ALL FILES CLASSIFIED SUCCESSFULLY")