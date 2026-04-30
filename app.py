from dotenv import load_dotenv
load_dotenv()

import os
import imaplib
import email
from email.header import decode_header

from flask import Flask, jsonify, render_template

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import (
    Features, SentimentOptions, EmotionOptions
)

app = Flask(__name__)

# ---------------------------
# Watson NLU setup
# ---------------------------
authenticator = IAMAuthenticator(os.getenv("WATSON_API_KEY"))
nlu = NaturalLanguageUnderstandingV1(
    version="2021-08-01",
    authenticator=authenticator
)
nlu.set_service_url(os.getenv("WATSON_SERVICE_URL"))


# ---------------------------
# Helpers: email decoding
# ---------------------------
def decode_subject(subject):
    if not subject:
        return "No Subject"
    decoded = decode_header(subject)
    out = ""
    for part, enc in decoded:
        if isinstance(part, bytes):
            out += part.decode(enc or "utf-8", errors="ignore")
        else:
            out += str(part)
    return out


def extract_email_body(msg):
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition") or "")
                if ctype == "text/plain" and "attachment" not in disp:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        body = payload.decode("utf-8", errors="ignore")
                    else:
                        body = str(payload or "")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = str(payload or "")
    except Exception:
        body = ""
    return (body or "").strip()


# ---------------------------
# Demo emails
# ---------------------------
def get_demo_emails():
    return [
        {
            "from": "angry.customer@example.com",
            "subject": "Terrible Service!",
            "body": "I am extremely disappointed with your product. It broke after one day. This is absolutely unacceptable and I want a full refund immediately!"
        },
        {
            "from": "concerned.user@example.com",
            "subject": "Order delay issue",
            "body": "My order has not arrived yet. I am disappointed because I expected it yesterday. Please check the status."
        },
        {
            "from": "happy.client@example.com",
            "subject": "Great support",
            "body": "Thank you so much. Your support team was very helpful and solved my issue quickly."
        },
        {
            "from": "normal.user@example.com",
            "subject": "Need information",
            "body": "Hello, can you please tell me the status of my request? Thank you."
        }
    ]


# ---------------------------
# Gmail fetch
# ---------------------------
def get_emails_from_gmail():
    emails = []

    imap_server = (os.getenv("IMAP_SERVER", "imap.gmail.com") or "").strip()
    imap_email = (os.getenv("IMAP_EMAIL") or "").strip()
    imap_password = (os.getenv("IMAP_PASSWORD") or "").strip()

    # strict: if missing creds -> empty list
    if not imap_email or not imap_password:
        return emails

    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(imap_email, imap_password)
    mail.select("INBOX")

    # last 5
    status, data = mail.search(None, "ALL")
    ids = data[0].split()[-5:] if status == "OK" and data and data[0] else []

    for num in ids:
        status, msg_data = mail.fetch(num, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            continue

        msg = email.message_from_bytes(msg_data[0][1])
        sender = msg.get("From", "unknown@email.com")
        subject = decode_subject(msg.get("Subject"))
        body = extract_email_body(msg)

        if not body:
            body = subject

        emails.append({
            "from": sender,
            "subject": subject,
            "body": body[:800]
        })

    mail.close()
    mail.logout()
    return emails


# ---------------------------
# Watson analysis + priority
# ---------------------------
def analyze_with_watson(text):
    try:
        safe_text = (text or "").strip()
        if len(safe_text) < 10:
            safe_text = safe_text + " Customer email message."

        response = nlu.analyze(
            text=safe_text,
            features=Features(
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()
            )
        ).get_result()

        sent = response["sentiment"]["document"]
        emo = response["emotion"]["document"]["emotion"]

        sentiment_label = sent.get("label", "neutral")
        sentiment_score = round(sent.get("score", 0), 2)
        anger = emo.get("anger", 0)

        if sentiment_label == "negative" and anger > 0.6:
            priority = "URGENT"
        elif sentiment_label == "negative":
            priority = "MEDIUM"
        else:
            priority = "LOW"

        dominant_emotion = max(emo, key=emo.get) if emo else "neutral"

        return {
            "sentiment": sentiment_label,
            "score": sentiment_score,
            "emotion": {k: round(v, 2) for k, v in emo.items()},
            "priority": priority,
            "dominant_emotion": dominant_emotion
        }

    except Exception as e:
        print("Watson Error:", e)
        return {
            "sentiment": "neutral",
            "score": 0,
            "emotion": {"anger": 0, "joy": 0, "sadness": 0, "fear": 0, "disgust": 0},
            "priority": "LOW",
            "dominant_emotion": "neutral"
        }


# ---------------------------
# Suggested reply (template-based)
# ---------------------------
def generate_reply(analysis):
    priority = analysis.get("priority", "LOW")
    dom = analysis.get("dominant_emotion", "neutral")

    if priority == "URGENT":
        return ("We sincerely apologize for your experience. "
                "Your concern is important to us, and we are escalating this immediately for urgent attention.")
    if dom == "anger":
        return ("I understand why this situation may be frustrating. "
                "Please allow us to review it carefully and help resolve it as soon as possible.")
    if dom == "sadness":
        return ("We are sorry this experience has been disappointing. "
                "We truly value your concern and will do our best to support you.")
    if dom == "joy":
        return ("Thank you for sharing your positive feedback. "
                "We are glad to know about your experience and appreciate your message.")

    return ("Thank you for reaching out. "
            "We have received your message and will get back to you with the next steps soon.")


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/emails")
def api_emails():
    # ✅ DEMO on Render, REAL on Local
    use_demo = os.getenv("USE_DEMO_EMAILS", "0").lower() in ("1", "true", "yes", "on")
    privacy_mode = os.getenv("PRIVACY_MODE", "1") == "1"

    if use_demo:
        source = "demo"
        raw = get_demo_emails()
        # Gmail never called on Render in this mode
    else:
        source = "gmail"
        raw = get_emails_from_gmail()
        # if Gmail empty (rare), you can decide fallback:
        if not raw:
            source = "demo"
            raw = get_demo_emails()

    processed = []
    for mail in raw:
        combined_text = f"{mail.get('subject','')}. {mail.get('body','')}"
        analysis = analyze_with_watson(combined_text)
        mail["analysis"] = analysis
        mail["suggested_reply"] = generate_reply(analysis)

        # Mask only when REAL Gmail is being used (but Render demo-only won't reach here)
        if privacy_mode and source == "gmail":
            mail["from"] = "Customer"
            mail["subject"] = "Customer message"
            mail["body"] = mail["body"]  # keep body if you want; set to "[Message hidden]" if you want masking
            # If you want full hiding, uncomment next line:
            # mail["body"] = "[Message hidden]"

        processed.append(mail)

    processed.sort(key=lambda x: {"URGENT": 0, "MEDIUM": 1, "LOW": 2}.get(x["analysis"]["priority"], 3))

    for i, m in enumerate(processed):
        m["_id"] = i

    return jsonify(processed)


@app.route("/api/reply", methods=["POST"])
def api_reply():
    data = request.json or {}
    print("=" * 60)
    print("Draft saved safely. No real email was sent.")
    print("To:", data.get("to"))
    print("Message:", data.get("message"))
    print("=" * 60)
    return jsonify({"status": "saved", "message": "Draft saved safely. No real email was sent."})


if __name__ == "__main__":
    print("=" * 60)
    print("EmpathyMail starting...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)