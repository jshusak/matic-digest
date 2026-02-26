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

RELEVANT = meaningful news, trends, innovation, regulation, market shifts, or business developments in {industry_name}.
NOT RELEVANT = tangentially related, off-topic, political news unrelated to the industry, too generic, or clickbait.

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


# ── STEP 3: Build the HTML digest page ───────────────────────────────────────
def generate_html(all_industry_articles):
    date_str  = datetime.now().strftime("%B %d, %Y")
    month_day = datetime.now().strftime("%b %d")
    year      = datetime.now().strftime("%Y")
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

        for article in ind["articles"]:
            if article.get("urlToImage"):
                img_html = f'<img class="card-img" src="{article["urlToImage"]}" alt="" onerror="this.style.display=\'none\'">'
            else:
                img_html = f'<div class="card-img-accent" style="background:{accent};"></div>'

            tp_html = "".join(f"<li>{pt}</li>" for pt in article.get("talking_points", []))

            cards += f"""
            <div class="card" style="--accent:{accent};">
              {img_html}
              <div class="card-body">
                <div class="source">{article['source']['name']}</div>
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

    /* ── Footer ── */
    .site-footer {{
      border-top: 1px solid var(--rule);
      padding: 32px 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
    }}
    .footer-logo {{
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
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
        padding: 0 20px;
      }}
      .site-nav a {{
        padding: 0 12px;
        font-size: 9px;
      }}
      .site-nav a:first-child {{
        padding-left: 0;
      }}
      .nav-action {{
        padding-left: 16px;
      }}
      .notebook-btn {{
        font-size: 9px;
        padding: 8px 12px;
      }}
      main {{
        padding: 36px 20px 56px;
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
        gap: 14px;
      }}
      .card-img {{
        height: 200px;
      }}
      .card-body {{
        padding: 18px 18px 20px;
        gap: 12px;
      }}
      .headline {{
        font-size: 16px;
      }}
      .summary {{
        font-size: 14px;
      }}
      .meta-text {{
        font-size: 13px;
      }}
      .talking-points li {{
        font-size: 13px;
      }}
      .site-footer {{
        padding: 24px 20px;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
      }}
    }}
  </style>
</head>
<body>

<header class="site-header">
  <div class="brand">
    <h1>Matic Digest</h1>
    <p>Industry Intelligence · Internal</p>
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
    <div class="stat-divider"></div>
    <div class="stat">
      <div class="stat-value">{month_day}</div>
      <div class="stat-label">{year}</div>
    </div>
  </div>
</header>

<nav class="site-nav">
  <div class="nav-links">
    {nav_links}
  </div>
  <div class="nav-action">
    <button class="notebook-btn" onclick="copyURLs(this)">↗ Copy URLs for NotebookLM</button>
  </div>
</nav>

<main>
  {sections}
</main>

<footer class="site-footer">
  <div class="footer-logo">Matic Digital</div>
  <div class="footer-meta">Industry Digest · {date_str} · Internal Use Only</div>
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
    "The full breakdown is one click away — {url}",
    "Dig into the details in this week's full digest → {url}",
    "All 24 articles, fully briefed. Worth the scroll → {url}",
    "More signal, less noise — full digest here: {url}",
    "Everything above, plus the details that didn't fit. {url}",
    "Pull up the full digest when you get a minute → {url}",
    "That's the week in preview. Full read here: {url}",
]

def generate_slack_briefing(all_industry_articles, page_url):
    import random
    articles_text = ""
    for ind in all_industry_articles:
        articles_text += f"\n{ind['name'].upper()}\n"
        for a in ind["articles"]:
            articles_text += f"  - {a['title']}: {a.get('summary', '')}\n"

    cta = random.choice(CTA_VARIANTS).format(url=page_url)

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
    return message.content[0].text.strip()


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
        print(f"    Pulled {len(candidates)} total candidates. Evaluating relevance...")

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

    print("📄 Building HTML digest...")
    html = generate_html(all_industry_articles)

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
    briefing = generate_slack_briefing(all_industry_articles, page_url)

    print("💬 Posting to Slack...")
    post_to_slack(page_url, total, briefing)

    print(f"\n✓ {total} articles across {len(config['industries'])} industries.")
    print(f"🌐 Live at: {page_url}")


if __name__ == "__main__":
    main()
