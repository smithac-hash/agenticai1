"""NewsAI - AI-Based Personalized News Digest Generator (Flask + SQLite)."""
import json, math, os, re, sqlite3, time, threading
from collections import Counter
from datetime import datetime, timezone
import feedparser, requests
from flask import Flask, jsonify, request, send_from_directory

DB = os.environ.get("NEWSAI_DB", "newsai.db")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # optional: LLM summaries
app = Flask(__name__, static_folder="static")

FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "ESPN": "https://www.espn.com/espn/rss/news",
}
CATEGORIES = {
    "Technology": "ai software tech app google apple microsoft startup chip cyber data robot internet phone openai",
    "Business": "market stock economy bank inflation trade company profit earnings shares investor ceo tariff",
    "Politics": "election president government minister parliament senate vote policy law court congress party",
    "Sports": "match game team league cup coach player win goal season tournament cricket football nba",
    "Science": "study research space nasa climate scientists species planet physics discovery vaccine",
    "Health": "health hospital disease cancer drug patients doctors virus medical covid mental",
    "World": "war ukraine gaza china russia europe attack ceasefire diplomat border united nations",
    "Entertainment": "film movie music actor album celebrity series festival award box office show",
}
STOP = set("the a an and or of to in on for with at by from is are was were be as it its that this has have had will not but after over new says say us".split())

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY, title TEXT, url TEXT UNIQUE,
          source TEXT, published REAL, content TEXT, category TEXT, summary TEXT, cluster INTEGER);
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT, prefs TEXT);
        CREATE TABLE IF NOT EXISTS reads(user_id INT, article_id INT, liked INT, PRIMARY KEY(user_id,article_id));
        """)

# ---------- NLP ----------
def tokens(t): return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2]

def categorize(text):
    toks = Counter(tokens(text))
    scores = {c: sum(toks[w] for w in kw.split()) for c, kw in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

def summarize(title, text, n=2):
    text = re.sub(r"\s+", " ", text).strip()
    if API_KEY and len(text) > 200:
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=20,
                headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5", "max_tokens": 150,
                      "messages": [{"role": "user", "content": f"Summarize in 2 neutral sentences:\n{title}\n{text[:3000]}"}]})
            return r.json()["content"][0]["text"].strip()
        except Exception: pass
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s) > 40]
    if not sents: return text[:240]
    freq = Counter(tokens(title + " " + text))
    score = lambda s: sum(freq[w] for w in tokens(s)) / (len(tokens(s)) ** 0.5 + 1)
    top = sorted(range(len(sents)), key=lambda i: -score(sents[i]))[:n]
    return " ".join(sents[i] for i in sorted(top))

def vec(text, idf):
    c = Counter(tokens(text)); v = {w: f * idf.get(w, 1) for w, f in c.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1
    return {w: x / norm for w, x in v.items()}

def cos(a, b):
    if len(a) > len(b): a, b = b, a
    return sum(x * b.get(w, 0) for w, x in a.items())

def cluster_all(threshold=0.4):
    """Group near-duplicate/similar stories using TF-IDF cosine similarity."""
    with db() as c:
        rows = c.execute("SELECT id,title,content FROM articles ORDER BY published DESC LIMIT 600").fetchall()
        df = Counter(w for r in rows for w in set(tokens(r["title"] + " " + r["content"])))
        idf = {w: math.log(len(rows) / (1 + d)) + 1 for w, d in df.items()}
        vs = [vec(r["title"] * 2 + " " + r["content"], idf) for r in rows]
        lab = list(range(len(rows)))
        for i in range(len(rows)):
            for j in range(i):
                if cos(vs[i], vs[j]) >= threshold: lab[i] = lab[j]; break
        for r, l in zip(rows, lab): c.execute("UPDATE articles SET cluster=? WHERE id=?", (rows[l]["id"], r["id"]))

# ---------- ingestion ----------
def ingest():
    added = 0
    for src, url in FEEDS.items():
        try: feed = feedparser.parse(url, request_headers={"User-Agent": "NewsAI/1.0"})
        except Exception: continue
        for e in feed.entries[:25]:
            raw = e.get("summary", "") or (e.get("content") or [{}])[0].get("value", "")
            body = re.sub(r"<[^>]+>", " ", raw); title = e.get("title", "").strip()
            if not title or not e.get("link"): continue
            ts = time.mktime(e.published_parsed) if e.get("published_parsed") else time.time()
            try:
                with db() as c:
                    c.execute("INSERT INTO articles(title,url,source,published,content,category,summary) VALUES(?,?,?,?,?,?,?)",
                        (title, e.link, src, ts, body, categorize(title + " " + body), summarize(title, body or title)))
                added += 1
            except sqlite3.IntegrityError: pass
    cluster_all(); return added

# ---------- personalization ----------
def rank(user_id, limit=15):
    with db() as c:
        u = c.execute("SELECT prefs FROM users WHERE id=?", (user_id,)).fetchone()
        prefs = json.loads(u["prefs"]) if u else {}
        w, kws = prefs.get("weights", {}), [k.lower() for k in prefs.get("keywords", [])]
        rows = [dict(r) for r in c.execute("SELECT * FROM articles ORDER BY published DESC LIMIT 400")]
        seen = {r["article_id"] for r in c.execute("SELECT article_id FROM reads WHERE user_id=?", (user_id,))}
    best = {}
    for a in rows:
        age_h = (time.time() - a["published"]) / 3600
        s = w.get(a["category"], 0.3) * 2 + math.exp(-age_h / 24)
        s += sum(1.5 for k in kws if k in (a["title"] + a["content"]).lower())
        if a["id"] in seen: s -= 1
        a["score"] = s
        k = a["cluster"] or a["id"]  # one representative per story cluster
        if k not in best or s > best[k]["score"]:
            best[k] = a; a["also_covered_by"] = []
        else: best[k]["also_covered_by"].append(a["source"])
    for k, a in best.items():
        a["also_covered_by"] = sorted(set(a["also_covered_by"]) - {a["source"]})
        a["content"] = ""
    return sorted(best.values(), key=lambda a: -a["score"])[:limit]

# ---------- API ----------
@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.post("/api/refresh")
def refresh(): return jsonify(added=ingest())

@app.post("/api/user")
def save_user():
    d = request.json; prefs = {"weights": {c: 1.0 for c in d.get("categories", [])}, "keywords": d.get("keywords", [])}
    with db() as c:
        if d.get("id"):
            c.execute("UPDATE users SET name=?,email=?,prefs=? WHERE id=?", (d["name"], d.get("email", ""), json.dumps(prefs), d["id"])); uid = d["id"]
        else: uid = c.execute("INSERT INTO users(name,email,prefs) VALUES(?,?,?)", (d["name"], d.get("email", ""), json.dumps(prefs))).lastrowid
    return jsonify(id=uid)

@app.get("/api/user/<int:uid>")
def get_user(uid):
    with db() as c: u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return (jsonify(dict(u, prefs=json.loads(u["prefs"]))) if u else (jsonify(error="not found"), 404))

@app.get("/api/articles")
def articles():
    cat, q = request.args.get("category"), request.args.get("q", "")
    sql, args = "SELECT id,title,url,source,published,category,summary,cluster FROM articles WHERE 1=1", []
    if cat: sql += " AND category=?"; args.append(cat)
    if q: sql += " AND (title LIKE ? OR content LIKE ?)"; args += [f"%{q}%"] * 2
    with db() as c: return jsonify([dict(r) for r in c.execute(sql + " ORDER BY published DESC LIMIT 60", args)])

@app.get("/api/digest/<int:uid>")
def digest(uid):
    items = rank(uid)
    by_cat = {}
    for a in items: by_cat.setdefault(a["category"], []).append(a)
    return jsonify(date=datetime.now(timezone.utc).strftime("%A, %d %B %Y"), sections=by_cat, total=len(items))

@app.post("/api/feedback")
def feedback():
    d = request.json
    with db() as c:
        a = c.execute("SELECT category FROM articles WHERE id=?", (d["article_id"],)).fetchone()
        u = c.execute("SELECT prefs FROM users WHERE id=?", (d["user_id"],)).fetchone()
        if not (a and u): return jsonify(error="bad ids"), 400
        p = json.loads(u["prefs"]); delta = 0.25 if d["liked"] else -0.25
        p["weights"][a["category"]] = max(0, p["weights"].get(a["category"], 0.3) + delta)  # learn interests
        c.execute("UPDATE users SET prefs=? WHERE id=?", (json.dumps(p), d["user_id"]))
        c.execute("INSERT OR REPLACE INTO reads VALUES(?,?,?)", (d["user_id"], d["article_id"], int(d["liked"])))
    return jsonify(ok=True)

def daily_job():  # refresh feeds every 3 hours
    while True:
        time.sleep(10800); ingest()

if __name__ == "__main__":
    init(); threading.Thread(target=daily_job, daemon=True).start()
    app.run(port=5000, debug=False)
