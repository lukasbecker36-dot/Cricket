# Hetzner deployment: live cricket signals

This deploys the Betfair signal runner as a systemd service on a Hetzner
Ubuntu/Debian server. The runner polls Betfair, applies the saved model
to upcoming T20 matches, and sends Telegram alerts when it finds a lay
opportunity that exceeds the validated edge threshold.

## What you need before starting

1. **A Hetzner Cloud server** (the cheapest CX22 / CPX11 is more than enough — 1 vCPU, 2 GB RAM)
2. **Betfair account** with API access enabled
   - Apply for an "Application Key" at https://developer.betfair.com/
   - You'll get a `live` app key (free, rate-limited) and optionally a `delayed` one
3. **Telegram bot**
   - In Telegram, message [@BotFather](https://t.me/botfather) and `/newbot` to create one
   - Save the bot token
   - Send any message to your new bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat ID

## Step 1: server setup

```bash
# As root on a fresh Hetzner Ubuntu 24.04 box
apt update && apt install -y python3.11 python3.11-venv git
adduser --system --group --home /opt/cricket cricket
mkdir -p /opt/cricket /etc/cricket /opt/cricket/data /opt/cricket/models
chown -R cricket:cricket /opt/cricket
```

## Step 2: clone repo & install

```bash
sudo -u cricket bash <<'EOF'
cd /opt/cricket
git clone https://github.com/<your-fork>/cricket.git .
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
EOF
```

## Step 3: bring over the trained model

The model files live in `models/` and need to be present on the server:

```
models/full_innings_gbm.lgb
models/full_innings_meta.json
models/full_innings_bat_pp.json
models/full_innings_bowl_pp.json
models/full_innings_venue_par.json
```

**Either** retrain on the server (requires bringing data over too — large),
**or** just `scp` the `models/` directory from your dev machine to the server:

```bash
scp -r models/ root@<hetzner-ip>:/opt/cricket/models/
chown -R cricket:cricket /opt/cricket/models
```

## Step 4: secrets

```bash
cp /opt/cricket/deploy/live.env.example /etc/cricket/live.env
$EDITOR /etc/cricket/live.env          # fill in real values
chmod 600 /etc/cricket/live.env
chown root:root /etc/cricket/live.env
```

## Step 5: dry-run test

```bash
sudo -u cricket bash -c "
  cd /opt/cricket
  set -a; . /etc/cricket/live.env; set +a
  LIVE_DRY_RUN=1 .venv/bin/python -m scripts.live_signals
"
```

This logs signals to journal without sending Telegram. Should see:
- "logged in to Betfair; session refreshed"
- Periodic poll logs (every 60s)
- If a T20 match starts in the next 15 min with edge, "scan: N new signals"

Stop with Ctrl-C.

## Step 6: install and start the service

```bash
cp /opt/cricket/deploy/cricket-live.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cricket-live
journalctl -u cricket-live -f
```

You should see "Cricket live signal runner started" in Telegram.

## Day-to-day

- **Logs:** `journalctl -u cricket-live -f`
- **Recent signals:** `tail -f /opt/cricket/data/live/signals/$(date +%F).jsonl`
- **Telegram commands** (just send to the bot):
  - `/status` — last poll time, signals sent today, etc.
  - `/pause` / `/resume` — stop / start sending alerts (still logs)
  - `/clearcache` — clear de-dup cache (re-alert known signals; useful after model retrain)
- **Stop service:** `systemctl stop cricket-live`
- **Restart after model update:** `systemctl restart cricket-live`

## Retraining

Periodically (e.g. monthly), retrain on fresh data:

1. Update Cricsheet and Betfair archives on your dev box
2. Re-run `python -m scripts.train_and_save_full_innings`
3. `scp models/*.lgb models/*.json` to the server
4. `systemctl restart cricket-live`

## Security notes

- `/etc/cricket/live.env` is chmod 600, root-only. Don't put secrets in the repo
- The `cricket` user has no shell login by default, no sudo
- ProtectSystem=strict in the unit file restricts file system writes to data/models only
- Rotate Betfair password periodically; refresh app key if compromised
- Rotate Telegram token if leaked (BotFather → `/revoke`)

## What this doesn't do (yet)

- **Does not place real bets.** Signals are alerts only. You execute manually
  by following the alert in the Betfair app. This is Phase 1 deliberately —
  validate the strategy live before risking auto-execution
- **Does not handle multi-bet positions or risk limits.** You're responsible
  for daily exposure
- **Does not retry on Betfair API outages** beyond simple sleep-and-continue
- **No web UI** — Telegram is the interface

These are Phase 2 work after you've seen 100-200 live signal outcomes.
