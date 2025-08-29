import os
from twilio.rest import Client

# ---------- Twilio helpers ----------
def _load_twilio_client():
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_num = os.environ.get("TWILIO_FROM")  # e.g. "+12345550123"
    to_num = os.environ.get("TWILIO_TO")      # e.g. "+12345550123"

    missing = [k for k, v in {
        "TWILIO_ACCOUNT_SID": sid,
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_FROM": from_num,
        "TWILIO_TO": to_num
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Twilio secrets: {', '.join(missing)}")

    return Client(sid, token), from_num, to_num

def send_twilio_sms(body: str):
    client, from_num, to_num = _load_twilio_client()
    client.messages.create(body=body, from_=from_num, to=to_num)

# Your Account SID and Auth Token from console.twilio.com
account_sid = "ACa8a80dd771d80903084d35d1aef78529"
auth_token = "8e0e7814cbaad6cb3177da53af73be71"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+15712718265",
    from_="+18884715413",
    body="Hello from Python!")

print(message.sid)

