"""
SOC watcher: classifies reported emails and writes one JSON line per email for Wazuh to collect.

  python soc_watcher.py <inbox_folder> <alerts_log_file>           # runs forever, checks every 5 s
  python soc_watcher.py <inbox_folder> <alerts_log_file> --once    # process what is there, then exit

Drop .eml (or .txt) files into <inbox_folder>. Processed files are moved to <inbox_folder>/processed/.
"""
import sys, os, time, json, shutil, email
from email import policy
from ai_phish_detector import analyze


def read_email(path):
    if path.lower().endswith(".eml"):
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part else ""
        return str(msg["subject"] or ""), str(msg["from"] or ""), body
    return "", "", open(path, encoding="utf-8", errors="ignore").read()


def process_folder(inbox, log_file):
    done = os.path.join(inbox, "processed"); os.makedirs(done, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(inbox)):
        path = os.path.join(inbox, name)
        if not os.path.isfile(path) or not name.lower().endswith((".eml", ".txt")):
            continue
        try:
            subject, sender, body = read_email(path)
            alert = analyze(subject + "\n" + body)
            alert.update({"file": name, "email_subject": subject[:200], "email_from": sender[:200]})
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert) + "\n")          # ONE JSON object per line (Wazuh requirement)
            shutil.move(path, os.path.join(done, name)); n += 1
            print(f"[{alert['severity']:8s}] {alert['verdict']:12s} {alert['confidence']:.2f}  {name}")
        except Exception as e:
            print("ERROR processing", name, e)
    return n


if __name__ == "__main__":
    if len(sys.argv) < 3: print(__doc__); sys.exit(1)
    inbox, log = sys.argv[1], sys.argv[2]
    os.makedirs(inbox, exist_ok=True)
    if "--once" in sys.argv: process_folder(inbox, log)
    else:
        print("Watching", inbox, "->", log)
        while True:
            process_folder(inbox, log); time.sleep(5)
