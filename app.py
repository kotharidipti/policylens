import re
import difflib
import gradio as gr
from transformers import pipeline

MODEL = "cross-encoder/nli-deberta-v3-xsmall"
THRESH = 0.7
TEMPLATE = "The company {}."

# label phrase -> (risk weight, plain-language explanation)
RISKS = {
    "sells personal data": (3, "Your data may be sold for profit."),
    "shares data with third parties or advertisers": (2, "Your data goes to other companies, often advertisers."),
    "tracks users with cookies or trackers": (2, "Cookies or trackers follow your activity."),
    "uses user data to train AI models": (2, "Your content may be used to train AI models."),
    "keeps data for a long or unlimited time": (2, "There is no clear deletion date."),
    "collects precise location": (2, "Your exact location is collected."),
    "collects children's data": (2, "Children's data is involved."),
    "lets users delete their data": (-1, "Good: you can ask for your data to be deleted."),
}
LABELS = list(RISKS)

RIGHTS = {
    "GDPR (EU/UK)": "You can access, object to, and erase your data (GDPR Arts. 15-21).",
    "CCPA/CPRA (California)": "You can opt out of sale or sharing and request deletion.",
    "DPDP Act (India)": "You can withdraw consent and request erasure (DPDP Act 2023).",
    "LGPD (Brazil)": "You can request access, correction, and deletion of your data.",
}

_clf = None


def clf():
    global _clf
    if _clf is None:
        _clf = pipeline("zero-shot-classification", model=MODEL)
    return _clf


def split_clauses(text, limit=60):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 30][:limit]


def classify(sentences):
    """Return {label: (score, sentence)} keeping the best match per label."""
    best = {}
    for s in sentences:
        out = clf()(s, LABELS, hypothesis_template=TEMPLATE, multi_label=True)
        for lab, sc in zip(out["labels"], out["scores"]):
            if sc >= THRESH and (lab not in best or sc > best[lab][0]):
                best[lab] = (sc, s)
    return best


def grade(best):
    pts = sum(RISKS[l][0] for l in best)
    for cut, g in [(1, "A"), (3, "B"), (6, "C"), (9, "D")]:
        if pts <= cut:
            return g, pts
    return "F", pts


def render(best, region):
    g, pts = grade(best)
    md = [f"## Privacy grade: {g}  (score {pts}, lower is better)\n"]
    if not best:
        md.append("No known risk patterns detected. Try a longer policy.")
    for lab, (sc, sent) in sorted(best.items(), key=lambda x: -RISKS[x[0]][0]):
        md.append(f"**{lab.capitalize()}** ({sc:.0%} confidence)  \n{RISKS[lab][1]}  \n> {sent}\n")
    md.append(f"\n**Your rights under {region}:** {RIGHTS[region]}\n\n*Informational only, not legal advice.*")
    return "\n".join(md)


def analyze(text, region):
    if not text.strip():
        return "Paste a privacy policy first."
    return render(classify(split_clauses(text)), region)


def diff(old, new, region):
    a, b = split_clauses(old, 200), split_clauses(new, 200)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("insert", "replace"):
            added += b[j1:j2]
        if tag in ("delete", "replace"):
            removed += a[i1:i2]
    if not added and not removed:
        return "No meaningful changes found."
    md = [f"### {len(added)} clauses added, {len(removed)} removed\n"]
    risky = classify(added[:40])
    if risky:
        md.append("**New risks introduced by this update:**\n")
        for lab, (sc, sent) in risky.items():
            md.append(f"- **{lab}**: {sent}")
    else:
        md.append("No new risky patterns in the added text.")
    return "\n".join(md)


OLD = "We collect your email and name. We use cookies to keep you logged in. You may request deletion of your account at any time."
NEW = OLD + " We may share your information with advertising partners. We use your content to train our machine learning models. We may sell personal information to data brokers."

with gr.Blocks(title="PolicyLens") as demo:
    gr.Markdown("# PolicyLens\nA privacy nutrition label for any policy, with your legal rights and change tracking.")
    region = gr.Dropdown(list(RIGHTS), value="GDPR (EU/UK)", label="Your region's law")
    with gr.Tab("Analyze a policy"):
        t = gr.Textbox(lines=10, label="Paste privacy policy", value=NEW)
        o = gr.Markdown()
        gr.Button("Generate label").click(analyze, [t, region], o)
    with gr.Tab("Detect policy changes"):
        old = gr.Textbox(lines=6, label="Old version", value=OLD)
        new = gr.Textbox(lines=6, label="New version", value=NEW)
        o2 = gr.Markdown()
        gr.Button("What changed?").click(diff, [old, new, region], o2)

if __name__ == "__main__":
    demo.launch()
