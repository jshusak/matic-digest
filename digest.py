import os
import json
import requests
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

# ── Load your API keys from the .env file ─────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

if not ANTHROPIC_API_KEY:
    raise RuntimeError(f"ANTHROPIC_API_KEY not found. Looked in: {_env_path}")

# ── Load your industry config ─────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    config = json.load(f)

# ── Set up the Claude client ──────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

ARTICLES_TARGET = config["articles_per_industry"]
ARTICLES_FETCH  = 12
ACCOUNTS_FETCH  = 5


# ── STEP 1: Fetch articles from NewsAPI ──────────────────────────────────────
def fetch_articles(industry):
    query = " OR ".join([f'"{term}"' for term in industry["search_terms"]])
    from_date = (datetime.now() - timedelta(days=config["days_back"])).strftime("%Y-%m-%d")

    response = requests.get("https://newsapi.org/v2/everything", params={
        "q":        query,
        "from":     from_date,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": ARTICLES_FETCH,
        "apiKey":   NEWS_API_KEY,
    })

    data = response.json()
    if data["status"] == "ok":
        return data["articles"]
    else:
        print(f"  ✗ NewsAPI error for {industry['name']}: {data.get('message')}")
        return []


# ── STEP 1b: Fetch articles from RSS feeds ───────────────────────────────────
def fetch_from_rss(feed_url):
    import feedparser
    feed    = feedparser.parse(feed_url)
    source  = feed.feed.get("title", feed_url)
    articles = []
    for entry in feed.entries[:ARTICLES_FETCH]:
        # Try to extract an image from media content or enclosures
        image = None
        if hasattr(entry, "media_content") and entry.media_content:
            image = entry.media_content[0].get("url")
        elif hasattr(entry, "enclosures") and entry.enclosures:
            image = entry.enclosures[0].get("url")

        articles.append({
            "title":       entry.get("title", ""),
            "url":         entry.get("link", ""),
            "description": entry.get("summary", ""),
            "content":     entry.get("summary", ""),
            "source":      {"name": source},
            "urlToImage":  image,
            "publishedAt": entry.get("published", ""),
        })
    return articles


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate_articles(candidates):
    """Remove articles whose titles are 70%+ similar to one already seen."""
    seen, unique = [], []
    for article in candidates:
        title = article.get("title", "").lower().strip()
        title_clean = "".join(c for c in title if c.isalnum() or c.isspace())
        words_a = set(title_clean.split())
        is_dup = False
        for seen_words in seen:
            if words_a and seen_words:
                overlap = len(words_a & seen_words) / max(len(words_a), len(seen_words))
                if overlap >= 0.7:
                    is_dup = True
                    break
        if not is_dup:
            seen.append(words_a)
            unique.append(article)
    return unique


# ── Date formatting ────────────────────────────────────────────────────────────
def format_date(date_str):
    """Return a human-readable relative date from a publishedAt string."""
    if not date_str:
        return ""
    from datetime import timezone
    import email.utils
    dt = None
    try:
        dt = datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    if dt is None:
        try:
            dt = email.utils.parsedate_to_datetime(date_str).astimezone(timezone.utc)
        except Exception:
            pass
    if dt is None:
        return ""
    now = datetime.now(timezone.utc)
    days = (now - dt).days
    if days == 0:
        hours = (now - dt).seconds // 3600
        return "Today" if hours == 0 else f"{hours}h ago"
    elif days == 1:
        return "Yesterday"
    elif days < 7:
        return f"{days}d ago"
    else:
        return dt.strftime("%b %d")


# ── STEP 2: Have Claude evaluate relevance and generate structured output ─────
def evaluate_and_summarize(article, industry_name):
    prompt = f"""You are a senior strategist at Matic Digital, a full-service branding, design, and technology agency.
You work with clients in the {industry_name} space.

Matic Digital's four service areas:
1. Brand & Creative — brand strategy & identity, content & messaging, brand systems & guidelines, rebranding & evolution, brand activation
2. Experience Design — personas & journey mapping, taxonomy & content strategy, design systems, UX/UI design, interaction & prototyping, user testing & validation
3. Software & Technology — website & software development, headless & monolithic CMS, platform modernization & integrations, full-stack engineering, security & compliance, ongoing support
4. Growth & Marketing Ops — market & audience intelligence, white space opportunity, go-to-market activation, SEO/GEO/AI visibility, content systems, lead generation & sales conversion, performance optimization

Evaluate this article. Is it genuinely relevant to the {industry_name} industry?

RELEVANT = meaningful news, trends, innovation, regulation, market shifts, or business developments directly in the {industry_name} industry.
NOT RELEVANT = reject any of the following without exception: stock/share price movements, investment analyst ratings, obituaries, awards or scholarship announcements, local human-interest stories, sports scores, celebrity gossip, viral social media content, political news not directly tied to {industry_name} regulation or policy, generic science/research with no clear industry application, or anything that would only loosely connect to {industry_name} with a stretch of imagination.

Article:
Title: {article['title']}
Source: {article['source']['name']}
Description: {article.get('description', '')}
Content: {article.get('content', '')}

If relevant, respond with ONLY this JSON (no other text):
{{
  "relevant": true,
  "summary": "2-3 sentences. Write like a smart colleague telling you what they just read — direct, clear, no fluff. Active voice. No jargon. Say what happened and why it matters.",
  "agency_relevance": "2-3 sentences. What does this signal for {industry_name} clients specifically? Name which of Matic's service areas are most relevant — Brand & Creative, Experience Design, Software & Technology, or Growth & Marketing Ops — and say plainly why. Sound like someone who has been in the room, not someone writing a proposal.",
  "talking_points": [
    "A natural conversation starter with a client in this space — curious, not salesy. Something you'd actually say over coffee.",
    "A specific service opportunity this news surfaces — name the Matic service area and frame it as a question or observation, not a pitch.",
    "A forward-looking provocation — the kind of question that makes a client pause and think differently about where they're headed."
  ]
}}

Tone guide:
- Write the way COLLINS, Area17, and Matic Digital write: confident, human, a little sharp.
- Short sentences. No hedging. No padding.
- Never use: leverage, solutions, deliverables, synergy, holistic, utilize, impactful.
- Talking points should sound like things a person would actually say — not bullets from a deck.

If NOT relevant, respond with ONLY:
{{"relevant": false}}"""

    try:
        message = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"relevant": False}
    except Exception as e:
        print(f"    ⚠ Evaluation error: {e}")
        return {"relevant": False}


# ── Named account tracking ────────────────────────────────────────────────────
def fetch_account_news(account):
    """Fetch recent news about a named account by exact company name (+ any aliases)."""
    names = [account["name"]] + account.get("aliases", [])
    query = " OR ".join(f'"{n}"' for n in names)
    from_date = (datetime.now() - timedelta(days=config.get("days_back", 7))).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": ACCOUNTS_FETCH,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        for a in articles:
            if not isinstance(a.get("source"), dict):
                a["source"] = {"name": "Unknown"}
            elif not a["source"].get("name"):
                a["source"]["name"] = "Unknown"
        return articles
    except Exception as e:
        print(f"    ⚠ Account fetch error ({account['name']}): {e}")
        return []


def evaluate_account_article(article, account):
    """Evaluate if an article is genuinely about this account and generate email/note."""
    is_prospect = account.get("type", "prospect").lower() == "prospect"
    context_line = f"Context: {account['context']}" if account.get("context") else ""

    if is_prospect:
        output_spec = '''"outreach_email": {{
    "subject": "A specific, natural subject line referencing the news — not clickbait, not generic",
    "body": "3-4 sentences. From a Matic Digital strategist to a senior contact at {name}. Reference the news. Make one sharp, specific observation about what it signals. Suggest a brief conversation naturally. Sound like a smart colleague, not a salesperson. No fluff, no pitch."
  }}'''.format(name=account["name"])
    else:
        output_spec = '"relationship_note": "1-2 sentences. A sharp talking point a Matic account manager can use in their next check-in with {name}. Reference the news. Connect it to something the team is likely working on. Conversational, not formal."'.format(name=account["name"])

    prompt = f"""You are a senior strategist at Matic Digital, a full-service branding, design, and technology agency.

You are reviewing a news article to see if it is genuinely and substantively about {account["name"]}, a company Matic is tracking.
{context_line}

Article:
Title: {article["title"]}
Source: {article["source"]["name"]}
Description: {article.get("description", "")}

First: is this article GENUINELY about {account["name"]} as a primary subject? Not a passing mention, not a different company with a similar name — it must be meaningfully about them.

Automatically mark as NOT relevant if the article is primarily about: stock price movements, analyst buy/sell ratings, earnings reports, investment recommendations, obituaries, awards or rankings lists, sports, celebrity gossip, or any story where {account["name"]} is only incidentally mentioned.

Only mark as relevant if the article covers something substantive — a product launch, partnership, rebrand, leadership change, strategic shift, new market entry, or meaningful business development that a branding or design agency would find genuinely useful as a conversation starter.

If YES, respond with ONLY this JSON:
{{
  "relevant": true,
  "summary": "1-2 sentences. What happened and why it matters for {account["name"]}.",
  "strategic_angle": "1-2 sentences. What does this signal from Matic's perspective — what might this company now need?",
  {output_spec}
}}

If NOT genuinely about {account["name"]}, respond with ONLY:
{{"relevant": false}}"""

    try:
        message = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"relevant": False}
    except Exception as e:
        print(f"    ⚠ Account evaluation error ({account['name']}): {e}")
        return {"relevant": False}


# ── STEP 3: Build the HTML digest page ───────────────────────────────────────
def generate_html(all_industry_articles, account_hits=[]):
    date_str  = datetime.now().strftime("%B %d, %Y")
    week_of   = datetime.now().strftime("Week of %B %d, %Y")
    total     = sum(len(ind["articles"]) for ind in all_industry_articles)

    accent_colors = {
        "Renewable Energy":        "#16a34a",
        "Health & Wellness":       "#2563eb",
        "Marketing Tech":          "#ea580c",
        "Tourism":                 "#db2777",
        "Fintech":                 "#7c3aed",
        "Artificial Intelligence": "#0891b2",
    }

    nav_links = ""
    for ind in all_industry_articles:
        anchor = ind["name"].lower().replace(" ", "-").replace("&", "and")
        nav_links += f'<a href="#{anchor}">{ind["name"]}</a>\n'

    all_urls = [a["url"] for ind in all_industry_articles for a in ind["articles"] if a.get("url")]

    sections = ""
    for i, ind in enumerate(all_industry_articles):
        anchor = ind["name"].lower().replace(" ", "-").replace("&", "and")
        accent = accent_colors.get(ind["name"], "#111111")
        num    = str(i + 1).zfill(2)
        cards  = ""

        if not ind["articles"]:
            cards = """
            <div class="card empty-card">
              <div class="card-body">
                <div class="empty-msg">Nothing noteworthy surfaced this week.</div>
              </div>
            </div>"""
        else:
            for article in ind["articles"]:
                if article.get("urlToImage"):
                    img_html = f'<img class="card-img" src="{article["urlToImage"]}" alt="" onerror="this.style.display=\'none\'">'
                else:
                    img_html = f'<div class="card-img-accent" style="background:{accent};"></div>'

                tp_html   = "".join(f"<li>{pt}</li>" for pt in article.get("talking_points", []))
                timestamp = format_date(article.get("publishedAt", ""))
                ts_html   = f'<span class="timestamp">{timestamp}</span>' if timestamp else ""

                cards += f"""
                <div class="card" style="--accent:{accent};">
                  {img_html}
                  <div class="card-body">
                    <div class="source">{article['source']['name']}{ts_html}</div>
                    <div class="headline">{article['title']}</div>
                    <div class="summary">{article.get('summary', '')}</div>

                    <div class="meta-block">
                      <div class="meta-label">Why it matters</div>
                      <div class="meta-text">{article.get('agency_relevance', '')}</div>
                    </div>

                    <div class="meta-block">
                      <div class="meta-label">Talking points</div>
                      <ul class="talking-points">{tp_html}</ul>
                    </div>

                    <a href="{article['url']}" target="_blank" class="read-more">
                      Read full article <span class="arrow">→</span>
                    </a>
                  </div>
                </div>"""

        sections += f"""
        <section id="{anchor}" style="--accent:{accent};">
          <div class="section-header">
            <div class="section-title">
              <span class="section-num">{num}</span>{ind["name"]}
            </div>
            <div class="section-meta">{len(ind["articles"])} articles this week</div>
          </div>
          <div class="grid">{cards}</div>
        </section>"""

    urls_js = json.dumps(all_urls)

    # ── On Our Radar section ──────────────────────────────────────────────────
    radar_html = ""
    if account_hits:
        radar_cards = ""
        for i, hit in enumerate(account_hits):
            acct    = hit["account"]
            art     = hit["article"]
            is_prospect = acct.get("type", "prospect").lower() == "prospect"
            badge_cls   = "badge-prospect" if is_prospect else "badge-client"
            badge_label = "Prospect" if is_prospect else "Client"
            ts          = format_date(art.get("publishedAt", ""))
            ts_str      = f" · {ts}" if ts else ""
            email_id    = f"email-body-{i}"

            if is_prospect and hit.get("outreach_email"):
                em      = hit["outreach_email"]
                subject = em.get("subject", "").replace('"', "&quot;")
                body    = em.get("body", "").replace("<", "&lt;").replace(">", "&gt;")
                action_block = f"""
                <div class="email-block">
                  <div class="email-block-header">
                    <div class="radar-meta-label">Outreach draft</div>
                    <button class="copy-email-btn" onclick="copyEmail(this, '{subject}', '{email_id}')">Copy</button>
                  </div>
                  <div class="email-subject">Subject: {subject}</div>
                  <div class="email-body" id="{email_id}">{body}</div>
                </div>"""
            elif not is_prospect and hit.get("relationship_note"):
                action_block = f"""
                <div class="radar-meta-block client-block">
                  <div class="radar-meta-label">Relationship talking point</div>
                  <div class="radar-meta-text">{hit["relationship_note"]}</div>
                </div>"""
            else:
                action_block = ""

            radar_cards += f"""
            <div class="radar-card">
              <div class="radar-card-top">
                <div class="account-name">{acct["name"]}</div>
                <span class="account-badge {badge_cls}">{badge_label}</span>
              </div>
              <div>
                <div class="radar-article-source">{art["source"]["name"]}{ts_str}</div>
                <div class="radar-article-headline">{art["title"]}</div>
              </div>
              <div class="radar-summary">{hit["summary"]}</div>
              <div class="radar-meta-block">
                <div class="radar-meta-label">Strategic angle</div>
                <div class="radar-meta-text">{hit["strategic_angle"]}</div>
              </div>
              {action_block}
            </div>"""

        radar_html = f"""
        <div class="radar-section">
          <div class="radar-header">
            <div class="radar-title">🎯 On Our Radar</div>
            <div class="radar-meta">{len(account_hits)} account{"s" if len(account_hits) != 1 else ""} in the news this week</div>
          </div>
          <div class="radar-grid">{radar_cards}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Matic Digest — {date_str}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #F3F2EF;
      --surface:   #ffffff;
      --ink:       #0f0f0f;
      --ink-2:     rgba(15,15,15,0.6);
      --ink-3:     rgba(15,15,15,0.35);
      --rule:      #E0DDD8;
      --shadow:    0 1px 2px rgba(0,0,0,.06), 0 0 0 1px rgba(0,0,0,.05);
      --shadow-up: 0 8px 40px rgba(0,0,0,.10), 0 0 0 1px rgba(0,0,0,.05);
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
      font-feature-settings: "kern" 1, "liga" 1;
    }}

    /* ── Header ── */
    .site-header {{
      background: var(--ink);
      padding: 40px 56px;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 48px;
    }}
    .brand h1 {{
      font-size: 15px;
      font-weight: 600;
      color: #fff;
      letter-spacing: -0.1px;
    }}
    .brand p {{
      font-size: 11px;
      color: rgba(255,255,255,0.3);
      margin-top: 4px;
      letter-spacing: 0.3px;
    }}
    .header-stats {{
      display: flex;
      gap: 40px;
      align-items: flex-end;
    }}
    .stat {{ text-align: right; }}
    .stat-value {{
      font-size: 36px;
      font-weight: 300;
      color: #fff;
      letter-spacing: -1.5px;
      line-height: 1;
    }}
    .stat-label {{
      font-size: 9px;
      color: rgba(255,255,255,0.3);
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-top: 5px;
    }}
    .stat-divider {{
      width: 1px;
      height: 48px;
      background: rgba(255,255,255,0.1);
    }}

    /* ── Nav ── */
    .site-nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: var(--surface);
      border-bottom: 1px solid var(--rule);
      padding: 0 56px;
      display: flex;
      align-items: stretch;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}
    .nav-links {{ display: flex; align-items: stretch; }}
    .site-nav a {{
      display: flex;
      align-items: center;
      padding: 0 16px;
      height: 48px;
      font-size: 10px;
      font-weight: 600;
      color: var(--ink-3);
      text-decoration: none;
      white-space: nowrap;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      border-bottom: 2px solid transparent;
      transition: color .15s, border-color .15s;
    }}
    .site-nav a:first-child {{ padding-left: 0; }}
    .site-nav a:hover {{ color: var(--ink); border-bottom-color: var(--ink); }}
    .nav-action {{
      margin-left: auto;
      display: flex;
      align-items: center;
      padding-left: 32px;
    }}
    .notebook-btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: var(--ink);
      color: #fff;
      border: none;
      cursor: pointer;
      padding: 9px 18px;
      border-radius: 2px;
      font-size: 10px;
      font-family: 'Inter', sans-serif;
      font-weight: 600;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      transition: opacity .15s;
    }}
    .notebook-btn:hover {{ opacity: 0.75; }}

    /* ── Main ── */
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 72px 56px 96px;
    }}

    /* ── Section ── */
    section {{ margin-bottom: 96px; }}
    .section-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      border-top: 3px solid var(--accent, #111);
      padding-top: 22px;
      margin-bottom: 32px;
    }}
    .section-title {{
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.8px;
      color: var(--ink);
      display: flex;
      align-items: baseline;
      gap: 14px;
    }}
    .section-num {{
      font-size: 14px;
      font-weight: 400;
      color: var(--accent, rgba(15,15,15,0.3));
    }}
    .section-meta {{
      font-size: 11px;
      color: var(--ink-3);
    }}

    /* ── Grid ── */
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      align-items: start;
    }}

    /* ── Card ── */
    .card {{
      background: var(--surface);
      border-radius: 3px;
      overflow: hidden;
      box-shadow: var(--shadow);
      transition: box-shadow .25s, transform .25s;
      display: flex;
      flex-direction: column;
    }}
    .card:hover {{
      box-shadow: var(--shadow-up);
      transform: translateY(-3px);
    }}
    .card-img {{
      width: 100%;
      height: 160px;
      object-fit: cover;
      display: block;
    }}
    .card-img-accent {{
      height: 4px;
      width: 100%;
    }}
    .card-body {{
      padding: 22px 22px 24px;
      display: flex;
      flex-direction: column;
      flex: 1;
      gap: 14px;
    }}
    .source {{
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--ink-3);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .timestamp {{
      font-size: 9px;
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
      color: var(--ink-3);
      flex-shrink: 0;
    }}
    .headline {{
      font-size: 15px;
      font-weight: 600;
      line-height: 1.4;
      color: var(--ink);
      letter-spacing: -0.1px;
    }}
    .summary {{
      font-size: 13px;
      line-height: 1.75;
      color: var(--ink-2);
    }}
    .meta-block {{
      background: rgba(0,0,0,.02);
      border-left: 2px solid var(--accent, #ddd);
      padding: 13px 15px;
      border-radius: 0 2px 2px 0;
    }}
    .meta-label {{
      font-size: 8px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      color: var(--accent, #999);
      margin-bottom: 7px;
    }}
    .meta-text {{
      font-size: 12px;
      line-height: 1.7;
      color: var(--ink-2);
    }}
    .talking-points {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .talking-points li {{
      font-size: 12px;
      line-height: 1.65;
      color: var(--ink-2);
      padding-left: 14px;
      position: relative;
    }}
    .talking-points li::before {{
      content: '—';
      position: absolute;
      left: 0;
      color: var(--accent, #ccc);
    }}
    .read-more {{
      margin-top: auto;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      font-weight: 600;
      color: var(--ink);
      text-decoration: none;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      transition: gap .2s;
    }}
    .read-more:hover {{ gap: 10px; }}

    /* ── On Our Radar ── */
    .radar-section {{
      margin-bottom: 80px;
      padding: 36px 40px;
      background: var(--surface);
      border-radius: 4px;
      border-left: 4px solid #d97706;
      box-shadow: var(--shadow);
    }}
    .radar-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 28px;
    }}
    .radar-title {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.4px;
      color: var(--ink);
    }}
    .radar-meta {{
      font-size: 11px;
      color: var(--ink-3);
    }}
    .radar-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .radar-card {{
      background: var(--bg);
      border-radius: 3px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      border: 1px solid var(--rule);
    }}
    .radar-card-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .account-name {{
      font-size: 16px;
      font-weight: 700;
      color: var(--ink);
      letter-spacing: -0.2px;
    }}
    .account-badge {{
      font-size: 8px;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .badge-prospect {{
      background: #fef3c7;
      color: #92400e;
    }}
    .badge-client {{
      background: #dcfce7;
      color: #14532d;
    }}
    .radar-article-headline {{
      font-size: 14px;
      font-weight: 600;
      color: var(--ink);
      line-height: 1.45;
      letter-spacing: -0.1px;
    }}
    .radar-article-source {{
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--ink-3);
    }}
    .radar-summary {{
      font-size: 13px;
      line-height: 1.7;
      color: var(--ink-2);
    }}
    .radar-meta-block {{
      background: rgba(217,119,6,0.06);
      border-left: 2px solid #d97706;
      padding: 12px 14px;
      border-radius: 0 2px 2px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .radar-meta-block.client-block {{
      background: rgba(22,163,74,0.06);
      border-left-color: #16a34a;
    }}
    .radar-meta-label {{
      font-size: 8px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      color: #d97706;
    }}
    .radar-meta-block.client-block .radar-meta-label {{
      color: #16a34a;
    }}
    .radar-meta-text {{
      font-size: 12px;
      line-height: 1.65;
      color: var(--ink-2);
    }}
    .email-block {{
      background: var(--surface);
      border: 1px solid var(--rule);
      border-radius: 3px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .email-block-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .copy-email-btn {{
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      background: var(--ink);
      color: #fff;
      border: none;
      border-radius: 2px;
      padding: 5px 10px;
      cursor: pointer;
      font-family: 'Inter', sans-serif;
      transition: opacity .15s;
    }}
    .copy-email-btn:hover {{ opacity: 0.7; }}
    .email-subject {{
      font-size: 11px;
      font-weight: 600;
      color: var(--ink);
    }}
    .email-body {{
      font-size: 12px;
      line-height: 1.75;
      color: var(--ink-2);
      white-space: pre-wrap;
    }}
    @media (max-width: 1024px) {{
      .radar-grid {{ grid-template-columns: 1fr; }}
      .radar-section {{ padding: 28px 28px; }}
    }}
    @media (max-width: 640px) {{
      .radar-section {{ padding: 20px 16px; margin-bottom: 48px; }}
      .radar-title {{ font-size: 18px; }}
    }}

    /* ── Empty card ── */
    .empty-card {{
      grid-column: 1 / -1;
      background: transparent;
      box-shadow: none;
      border: 1px dashed var(--rule);
    }}
    .empty-card:hover {{
      box-shadow: none;
      transform: none;
    }}
    .empty-msg {{
      font-size: 13px;
      color: var(--ink-3);
      padding: 32px 22px;
      font-style: italic;
    }}

    /* ── Footer ── */
    .site-footer {{
      border-top: 1px solid var(--rule);
      padding: 32px 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
    }}
    .footer-left {{
      display: flex;
      align-items: center;
      gap: 20px;
    }}
    .footer-logo {{
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
    }}
    .footer-divider {{
      width: 1px;
      height: 14px;
      background: var(--rule);
    }}
    .footer-tagline {{
      font-size: 11px;
      color: var(--ink-3);
    }}
    .footer-meta {{
      font-size: 11px;
      color: var(--ink-3);
    }}

    /* ── Tablet (2 columns) ── */
    @media (max-width: 1024px) {{
      .grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
      .site-header {{
        padding: 32px 32px;
      }}
      .stat-value {{
        font-size: 28px;
      }}
      main {{
        padding: 56px 32px 72px;
      }}
      .site-nav {{
        padding: 0 32px;
      }}
      .site-nav a:first-child {{
        padding-left: 16px;
      }}
      .site-footer {{
        padding: 28px 32px;
      }}
    }}

    /* ── Mobile (1 column) ── */
    @media (max-width: 640px) {{
      .site-header {{
        padding: 24px 20px;
        flex-direction: column;
        align-items: flex-start;
        gap: 20px;
      }}
      .header-stats {{
        width: 100%;
        gap: 0;
        justify-content: space-between;
        align-items: flex-end;
      }}
      .stat {{
        text-align: left;
        flex: 1;
      }}
      .stat-value {{
        font-size: 26px;
        letter-spacing: -1px;
      }}
      .stat-divider {{
        display: none;
      }}
      .site-nav {{
        padding: 0 16px;
      }}
      .site-nav a {{
        padding: 0 10px;
        font-size: 9px;
        height: 44px;
      }}
      .site-nav a:first-child {{
        padding-left: 0;
      }}
      .nav-action {{
        padding-left: 12px;
      }}
      .notebook-btn .btn-label {{
        display: none;
      }}
      .notebook-btn {{
        font-size: 14px;
        padding: 8px 10px;
        letter-spacing: 0;
      }}
      main {{
        padding: 32px 16px 56px;
      }}
      .section-header {{
        flex-direction: column;
        gap: 4px;
      }}
      .section-title {{
        font-size: 22px;
        letter-spacing: -0.4px;
      }}
      .section-meta {{
        font-size: 10px;
      }}
      .grid {{
        grid-template-columns: 1fr;
        gap: 12px;
      }}
      .card-img {{
        height: 180px;
      }}
      .card-body {{
        padding: 16px 16px 20px;
        gap: 12px;
      }}
      .headline {{
        font-size: 15px;
      }}
      .summary {{
        font-size: 13px;
      }}
      .meta-text {{
        font-size: 12px;
      }}
      .talking-points li {{
        font-size: 12px;
      }}
      .site-footer {{
        padding: 24px 16px;
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
      }}
      .footer-left {{
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
      }}
      .footer-divider {{ display: none; }}
    }}
  </style>
</head>
<body>

<header class="site-header">
  <div class="brand">
    <h1>Matic Digest</h1>
    <p>{week_of} · Industry Intelligence · Internal</p>
  </div>
  <div class="header-stats">
    <div class="stat">
      <div class="stat-value">{total}</div>
      <div class="stat-label">Articles</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat">
      <div class="stat-value">{len(all_industry_articles)}</div>
      <div class="stat-label">Industries</div>
    </div>
  </div>
</header>

<nav class="site-nav">
  <div class="nav-links">
    {nav_links}
  </div>
  <div class="nav-action">
    <button class="notebook-btn" onclick="copyURLs(this)">↗<span class="btn-label"> Copy URLs for NotebookLM</span></button>
  </div>
</nav>

<main>
  {radar_html}
  {sections}
</main>

<footer class="site-footer">
  <div class="footer-left">
    <div class="footer-logo">Matic Digital</div>
    <div class="footer-divider"></div>
    <div class="footer-tagline">Weekly Industry Intelligence</div>
  </div>
  <div class="footer-meta">{week_of} · Internal Use Only</div>
</footer>

<script>
  function copyURLs(btn) {{
    const urls = {urls_js};
    navigator.clipboard.writeText(urls.join("\\n")).then(() => {{
      const orig = btn.textContent;
      btn.textContent = "✓ Copied";
      setTimeout(() => btn.textContent = orig, 2500);
    }});
  }}
  function copyEmail(btn, subject, bodyId) {{
    const body = document.getElementById(bodyId).textContent;
    const full = "Subject: " + subject + "\\n\\n" + body;
    navigator.clipboard.writeText(full).then(() => {{
      btn.textContent = "✓ Copied";
      setTimeout(() => btn.textContent = "Copy", 2000);
    }});
  }}
</script>
</body>
</html>"""


# ── STEP 4: Wait for GitHub Pages to deploy ───────────────────────────────────
def wait_for_deployment(url, timeout=300):
    import time
    date_str = datetime.now().strftime("%B %d, %Y")
    print("⏳ Waiting for GitHub Pages to deploy", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and date_str in r.text:
                print(" ✓")
                return True
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(15)
    print("\n⚠ Timed out — posting to Slack anyway.")
    return False


# ── STEP 5: Generate Morning Brew-style Slack briefing ───────────────────────
INDUSTRY_EMOJI = {
    "Renewable Energy":      "⚡",
    "Health & Wellness":     "🩺",
    "Marketing Tech":        "🎯",
    "Tourism":               "✈️",
    "Fintech":               "💳",
    "Artificial Intelligence": "🤖",
}

CTA_VARIANTS = [
    "That's the week. Full breakdown is one click away.",
    "Dig deeper in the full digest — it's worth the scroll.",
    "More signal, less noise. The rest is in the digest.",
    "Everything above plus the details that didn't fit — hit the button.",
    "That's your preview. Full read is right below.",
    "Now you're caught up. The full digest has the rest.",
    "Good week to pay attention. Full digest below.",
]

def generate_slack_briefing(all_industry_articles, page_url, account_hits=[]):
    import random
    articles_text = ""
    for ind in all_industry_articles:
        articles_text += f"\n{ind['name'].upper()}\n"
        for a in ind["articles"]:
            articles_text += f"  - {a['title']}: {a.get('summary', '')}\n"

    cta = random.choice(CTA_VARIANTS)

    # Build emoji map string for the prompt
    emoji_map = "\n".join(f"  {name}: {emoji}" for name, emoji in INDUSTRY_EMOJI.items())

    prompt = f"""You are writing the weekly Matic Digest briefing for Slack.

Matic Digital is a full-service branding, design, and technology agency with four service areas:
1. Brand & Creative — brand strategy, identity, messaging, brand systems, rebranding
2. Experience Design — UX/UI, journey mapping, design systems, prototyping, user testing
3. Software & Technology — web & software development, CMS, platform modernization, full-stack engineering
4. Growth & Marketing Ops — go-to-market, SEO/GEO/AI visibility, audience intelligence, lead generation

This briefing goes to the internal strategy team before their week starts.

Write a Morning Brew-style executive briefing covering this week's industry news.
Tone: smart, conversational, a little sharp — like a well-informed colleague catching you up before a Monday meeting. Not corporate. Not stiff.

FORMAT RULES:
- Open with a single punchy line that sets the tone for the week (no industry prefix, just a strong opener)
- Then one section per industry, structured exactly like this:

[emoji] *Industry Name*
• One sentence on the most interesting story — weave in which Matic service area it connects to naturally
• One sentence on another notable story or trend
• (add a third bullet only if there's a genuinely distinct third angle worth calling out)

Use these emojis per industry:
{emoji_map}

- End with this exact line (do not change it): {cta}
- Use Slack markdown: *bold*, _italic_ where it adds punch
- Keep bullets tight — one sentence each, no padding
- Total length: readable in under 90 seconds

This week's articles:
{articles_text}"""

    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    briefing = message.content[0].text.strip()

    # Prepend "On Our Radar" block if there are account hits
    if account_hits:
        radar_lines = ["🎯 *On Our Radar*"]
        for hit in account_hits:
            acct  = hit["account"]
            label = "PROSPECT" if acct.get("type", "prospect").lower() == "prospect" else "CLIENT"
            note  = hit.get("strategic_angle") or hit.get("relationship_note") or ""
            # Trim to one sentence
            note  = note.split(".")[0] + "." if "." in note else note
            radar_lines.append(f"• *{acct['name']}* [{label}] — {hit['article']['title']}. {note}")
        briefing = "\n".join(radar_lines) + "\n\n" + briefing

    return briefing


# ── STEP 6: Post to Slack ─────────────────────────────────────────────────────
def post_to_slack(url, total, briefing):
    date_str = datetime.now().strftime("%B %d, %Y")

    # Slack section blocks have a 3000-character limit — split if needed
    MAX_BLOCK = 2900
    briefing_blocks = []
    while briefing:
        chunk = briefing[:MAX_BLOCK]
        # Don't cut mid-word
        if len(briefing) > MAX_BLOCK:
            cut = chunk.rfind("\n")
            if cut == -1:
                cut = chunk.rfind(" ")
            if cut > 0:
                chunk = briefing[:cut]
        briefing_blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": chunk}
        })
        briefing = briefing[len(chunk):].lstrip()

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Matic Digest — {date_str}"}
            },
            *briefing_blocks,
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"{total} articles · {len(config['industries'])} industries · Internal use only"}
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Read Full Digest →"},
                        "url": url,
                        "style": "primary"
                    }
                ]
            }
        ]
    }
    r = requests.post(SLACK_WEBHOOK_URL, json=payload)
    print("✓ Posted to Slack" if r.status_code == 200 else f"✗ Slack failed ({r.status_code}): {r.text}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("🔍 Starting Matic Digest...\n")
    all_industry_articles = []
    total = 0

    for industry in config["industries"]:
        print(f"  {industry['name']} — fetching candidates...")
        candidates = fetch_articles(industry)
        for feed_url in industry.get("rss_feeds", []):
            rss = fetch_from_rss(feed_url)
            print(f"    + {len(rss)} articles from RSS ({feed_url})")
            candidates.extend(rss)
        candidates = deduplicate_articles(candidates)
        print(f"    Pulled {len(candidates)} unique candidates. Evaluating relevance...")

        good_articles = []
        for article in candidates:
            if len(good_articles) >= ARTICLES_TARGET:
                break
            result = evaluate_and_summarize(article, industry["name"])
            if result.get("relevant"):
                article["summary"]          = result.get("summary", "")
                article["agency_relevance"] = result.get("agency_relevance", "")
                article["talking_points"]   = result.get("talking_points", [])
                good_articles.append(article)
                print(f"    ✓ {article['title'][:65]}...")
            else:
                print(f"    ✗ Skipped: {article['title'][:60]}...")

        print(f"    → {len(good_articles)} articles kept.\n")
        all_industry_articles.append({"name": industry["name"], "articles": good_articles})
        total += len(good_articles)

    # ── Named account tracking ────────────────────────────────────────────────
    account_hits = []
    if config.get("accounts"):
        print("🎯 Checking named accounts...\n")
        for account in config["accounts"]:
            label = account.get("type", "prospect").upper()
            print(f"  {account['name']} ({label})...")
            candidates = fetch_account_news(account)
            found = False
            for article in candidates:
                result = evaluate_account_article(article, account)
                if result.get("relevant"):
                    account_hits.append({
                        "account":           account,
                        "article":           article,
                        "summary":           result.get("summary", ""),
                        "strategic_angle":   result.get("strategic_angle", ""),
                        "outreach_email":    result.get("outreach_email"),
                        "relationship_note": result.get("relationship_note"),
                    })
                    print(f"    ✓ {article['title'][:65]}...")
                    found = True
                    break
            if not found:
                print(f"    — Nothing newsworthy this week.")
        print(f"\n  → {len(account_hits)} account hit(s) found.\n")

    print("📄 Building HTML digest...")
    html = generate_html(all_industry_articles, account_hits)

    dated_filename = f"digest_{datetime.now().strftime('%Y-%m-%d')}.html"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for filename in [dated_filename, "index.html"]:
        with open(os.path.join(base_dir, filename), "w") as f:
            f.write(html)

    print(f"✓ Saved: {dated_filename} + index.html")

    print("\n📡 Pushing to GitHub...")
    try:
        subprocess.run(["git", "-C", base_dir, "add", dated_filename, "index.html"], check=True)
        subprocess.run(["git", "-C", base_dir, "commit", "-m", f"Digest {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "-C", base_dir, "push", "origin", "main"], check=True)
        print("✓ Pushed to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"✗ Git error: {e}")
        return

    page_url = "https://jshusak.github.io/matic-digest/"
    wait_for_deployment(page_url)

    print("💬 Writing Slack briefing...")
    briefing = generate_slack_briefing(all_industry_articles, page_url, account_hits)

    print("💬 Posting to Slack...")
    post_to_slack(page_url, total, briefing)

    print(f"\n✓ {total} articles across {len(config['industries'])} industries.")
    print(f"🌐 Live at: {page_url}")


if __name__ == "__main__":
    main()
