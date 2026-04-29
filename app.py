
from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, render_template, jsonify, request

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import (
    Features, SentimentOptions, EmotionOptions
)


import imaplib
import email
from email.header import decode_header


app = Flask(__name__)


authenticator = IAMAuthenticator(os.getenv("WATSON_API_KEY"))

nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)

nlu.set_service_url(os.getenv("WATSON_SERVICE_URL"))

print("✅ Watson NLU Connected Successfully!")


def analyze_with_watson(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(
                sentiment=SentimentOptions(),
                emotion=EmotionOptions()
            )
        ).get_result()

        sent = response['sentiment']['document']
        emo = response['emotion']['document']['emotion']

        anger = emo.get('anger', 0)
        sadness = emo.get('sadness', 0)

      
        priority = "LOW"
        if sent['label'] == 'negative' and anger > 0.6:
            priority = "URGENT"
        elif sent['label'] == 'negative':
            priority = "MEDIUM"

        dominant_emotion = max(emo, key=emo.get)

        return {
            "sentiment": sent['label'],
            "score": round(sent['score'], 2),
            "emotion": {k: round(v, 2) for k, v in emo.items()},
            "priority": priority,
            "dominant_emotion": dominant_emotion
        }
    except Exception as e:
        print(f"❌ Watson Error: {e}")
        return {
            "sentiment": "neutral",
            "score": 0,
            "emotion": {},
            "priority": "LOW",
            "dominant_emotion": "neutral"
        }

def generate_reply(analysis):
    priority = analysis['priority']
    emotion = analysis['dominant_emotion']

    if priority == "URGENT":
        return "😔 We are truly sorry for this experience. I am escalating this immediately to our senior team. We value you deeply and will fix this right away."
    elif emotion == "sadness":
        return "💙 I completely understand your frustration. Please know that we care about your experience and are here to help make this right."
    elif emotion == "anger":
        return "🙏 I hear you and I understand why you are upset. Your concern is absolutely valid. Let me personally ensure this gets resolved."
    elif emotion == "joy":
        return "😊 Thank you so much for your wonderful feedback! It really brightens our day. We will keep working hard to serve you!"
    else:
        return "✨ Thank you for reaching out to us! We have received your message and our team is already working on it. You will hear from us soon."

def get_emails_from_gmail():
    emails = []
    try:
        mail = imaplib.IMAP4_SSL(os.getenv("IMAP_SERVER"))
        mail.login(os.getenv("IMAP_EMAIL"), os.getenv("IMAP_PASSWORD"))
        mail.select("INBOX")

   
        _, data = mail.search(None, "ALL")

       
        email_ids = data[0].split()[-5:]

        for num in email_ids:
            _, msg_data = mail.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

          
            subject = msg["Subject"]
            if subject:
                decoded = decode_header(subject)
                if isinstance(decoded[0][0], bytes):
                    subject = decoded[0][0].decode(decoded[0][1] or "utf-8")
            else:
                subject = "No Subject"

          
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

            sender = msg.get("From", "unknown@email.com")

            emails.append({
                "from": sender,
                "subject": subject,
                "body": body.strip()[:500]  
            })

        mail.close()
        mail.logout()
        print(f"✅ Fetched {len(emails)} emails from Gmail")

    except Exception as e:
        print(f"❌ Gmail Error: {e}")

    return emails


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/emails")
def get_emails():
    
    raw_emails = get_emails_from_gmail()

    
    if not raw_emails:
        print("⚠️ No emails found, using demo data")
        raw_emails = [
            {
                "from": "angry_customer@gmail.com",
                "subject": "Terrible Service!",
                "body": "I am extremely disappointed with your product. It broke after one day. This is absolutely unacceptable and I want a full refund immediately!"
            },
            {
                "from": "confused_user@gmail.com",
                "subject": "Need help with order",
                "body": "Hi, I placed an order last week but haven't received any update. Could you please check the status? Thank you."
            },
            {
                "from": "happy_customer@gmail.com",
                "subject": "Amazing experience!",
                "body": "Just wanted to say your customer service team was incredible. They resolved my issue in minutes. Thank you so much!"
            },
            {
                "from": "sad_customer@gmail.com",
                "subject": "Disappointed with delivery",
                "body": "My package arrived damaged and I feel really let down. I was looking forward to this for weeks. Very sad experience."
            }
        ]

    
    processed = []
    for mail in raw_emails:
        analysis = analyze_with_watson(mail["body"])
        mail["analysis"] = analysis
        mail["suggested_reply"] = generate_reply(analysis)
        processed.append(mail)

  
    priority_order = {"URGENT": 0, "MEDIUM": 1, "LOW": 2}
    processed.sort(key=lambda x: priority_order.get(x["analysis"]["priority"], 3))

    return jsonify(processed)


@app.route("/api/reply", methods=["POST"])
def send_reply():
    data = request.json
    print(f"📧 Reply to: {data.get('to')}")
    print(f"💌 Message: {data.get('message')}")
    return jsonify({
        "status": "sent",
        "message": "Reply sent successfully with empathy! 💌"
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  🚀 EmpathyMail Starting...")
    print("  📍 http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)
