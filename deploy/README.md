# Hetzner deployment: cricket signal helper (screenshot-driven)

A Telegram bot that receives your Betfair screenshots, extracts the market
data with Anthropic's vision API, runs the saved cricket model, and replies
with lay-bet recommendations. You place every bet manually on your phone.

**Why this design:** Betfair restricts data-centre IPs. The bot never talks
to Betfair — only to Telegram and the Anthropic API, both of which Hetzner
can reach freely. You provide the market data by screenshotting whenever
you're interested in a match.

## What you need before starting

1. **A Hetzner Cloud server.** CX22 / CPX11 is plenty (1 vCPU, 2 GB RAM).
2. **A Telegram bot.**
   - In Telegram, message [@BotFather](https://t.me/botfather) → `/newbot`
   - Save the bot token
   - Send any message to your new bot, then visit
     `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat_id
3. **An Anthropic API key.** Sign up at https://console.anthropic.com and
   create an API key. Vision is included on Sonnet/Haiku; usage on this
   bot is roughly £1–3/month even at heavy use.

## Step 1: server setup

```bash
# As root on Ubuntu 24.04
apt update && apt install -y python3.11 python3.11-venv python3-pip git
adduser --system --group --home /opt/cricket cricket
mkdir -p /opt/cricket /etc/cricket
chown -R cricket:cricket /opt/cricket
```

## Step 2: clone & install

```bash
sudo -u cricket bash <<'EOF'
cd /opt/cricket
git clone https://github.com/<your-fork>/cricket.git .
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
EOF
```

## Step 3: bring the trained model over

The bot loads the model from `models/` on the server. Either retrain on
the server (needs Cricsheet + Betfair data — large) or just scp from your
dev machine:

```bash
# From your dev machine
scp -r models/ root@<hetzner-ip>:/opt/cricket/models/
ssh root@<hetzner-ip> "chown -R cricket:cricket /opt/cricket/models"
```

Required files in `models/`:
- `full_innings_gbm.lgb`
- `full_innings_meta.json`
- `full_innings_bat_pp.json`
- `full_innings_bowl_pp.json`
- `full_innings_venue_par.json`

You also need the league-rosters module — but that ships with the repo.

## Step 4: secrets

```bash
cp /opt/cricket/deploy/live.env.example /etc/cricket/live.env
$EDITOR /etc/cricket/live.env       # fill in real values
chmod 600 /etc/cricket/live.env
chown root:root /etc/cricket/live.env
```

Required vars:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `ANTHROPIC_API_KEY`

## Step 5: dry-run test from a shell

```bash
sudo -u cricket bash -c '
  cd /opt/cricket
  set -a; . /etc/cricket/live.env; set +a
  .venv/bin/python -m scripts.live_signals
'
```

You should see a Telegram message: "*Cricket signal helper online*". Send
the bot any message in Telegram — it should respond. Stop with Ctrl-C.

## Step 6: install and start as systemd service

```bash
cp /opt/cricket/deploy/cricket-live.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cricket-live
journalctl -u cricket-live -f
```

## How to use it

Open the Betfair app on your phone, find an **Innings Runs** market for
a T20 match (1st or 2nd Innings Runs — the multi-runner ladder, not the
single Line market). Screenshot the visible ladder, send to the bot.

Add a quick caption — conversational, no fixed format:

- "MI v KKR"
- "Big Bash, Heat vs Sixers, batting first"
- "innings 2 now"  *(just text, no screenshot needed for context updates)*
- "venue is Wankhede"
- "reset"  *(forget the current match)*
- "status"  *(show what the bot knows)*

The bot will reply with the extracted ladder + the model's view + any lay
recommendations. Confirm with "yes" / "ok" / "✓" or just send the next
screenshot.

### Example exchange

You send: a screenshot + caption "MI v KKR innings 1"

Bot replies:

```
Read it: Mumbai Indians v Kolkata Knight Riders, innings 1
(vision confidence: high)

  140 or more   lay 1.10  (implied 91%) model 92%  edge +1.0pp
  150 or more   lay 1.35  (implied 74%) model 80%  edge +6.4pp
  160 or more   lay 1.85  (implied 54%) model 51%  edge -3.0pp
  170 or more   lay 2.40  (implied 42%) model 33%  edge -8.4pp ← LAY
  180 or more   lay 3.50  (implied 29%) model 22%  edge -6.7pp ← LAY
  190 or more   lay 5.50  (implied 18%) model 14%  edge -3.7pp ← LAY
  200 or more   lay 9.00  (implied 11%) model 8%   edge -3.0pp ← LAY

Signals:
Cricket LAY signal (ipl)
1st Innings Runs   innings 1
Runner: 170 Runs or more
Lay at 2.40 (implied 41.7%)
Model says 33.0% -> edge -8.4%
£10 stake -> liability £14.00 / win £9.50
```

You decide which (if any) to place on Betfair.

## Day-to-day

- **Logs**: `journalctl -u cricket-live -f`
- **State file**: `/opt/cricket/data/live/chat_state.json` (current match, history)
- **Restart**: `systemctl restart cricket-live`
- **Stop**: `systemctl stop cricket-live`
- **Update code**: `cd /opt/cricket && sudo -u cricket git pull && systemctl restart cricket-live`

## Retraining the model

Monthly-ish, run `scripts.train_and_save_full_innings` on your dev machine
with fresh Cricsheet + Betfair archives, then scp the new `models/*` to
the server and restart the service.

## Costs

- Hetzner CX22: ~€4/month
- Anthropic API: ~£1–3/month at typical usage
- Telegram: free

## Security

- `/etc/cricket/live.env` is root-only (chmod 600)
- The `cricket` user has no shell, no sudo, restricted FS writes via systemd
- Anthropic key is sensitive (can be used to incur cost) — rotate if leaked
- Telegram bot token: rotate via @BotFather → `/revoke` if leaked

## What this deliberately does NOT do

- **Place bets.** You bet manually; the bot only recommends.
- **Auto-poll Betfair.** You drive when analysis happens by sending screenshots.
- **Hold position state.** It doesn't know what bets you actually placed
  or what your bankroll is. That stays in your head and on Betfair.

If/when you want to graduate to auto-execution, you'll need the IP-tunnel
or home-Pi setup described in earlier notes. Don't do that until you've
seen 100+ live signals match simulated outcomes.
