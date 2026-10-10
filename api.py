"""Small web service for the player recommender"""

import argparse

from flask import Flask, jsonify, request

import config
from recommender import PlayerRecommender

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KG Basketball Player Recommender</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1b1f24}
 h1{font-size:1.5rem} .row{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0}
 input,select,button{font-size:1rem;padding:.5rem;border:1px solid #bbb;border-radius:6px}
 input{flex:1;min-width:220px} button{background:#1f6feb;color:#fff;border-color:#1f6feb;cursor:pointer}
 table{border-collapse:collapse;width:100%} th,td{padding:.5rem;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}
 .tag{display:inline-block;background:#eef3fb;border-radius:10px;padding:1px 8px;margin:1px;font-size:.8rem}
 .muted{color:#666;font-size:.9rem} .err{color:#b00020}
</style></head><body>
<h1>Find players with a similar playstyle</h1>
<p class="muted">Knowledge graph + Datalog rules + TransE embeddings, NBA 2025-26 regular season.</p>
<div class="row">
  <input id="player" list="names" placeholder="Player name, e.g. Luka Doncic" autocomplete="off">
  <datalist id="names"></datalist>
  <select id="n"><option>3</option><option selected>5</option><option>10</option></select>
  <select id="method"><option value="embedding">Embedding (TransE)</option><option value="rules">Rules (Jaccard)</option></select>
  <button onclick="go()">Recommend</button>
</div>
<div id="profile" class="muted"></div><div id="out"></div>
<script>
const q = s => document.querySelector(s);
q('#player').addEventListener('input', async e => {
  if (e.target.value.length < 2) return;
  const r = await fetch('/api/players?q=' + encodeURIComponent(e.target.value));
  q('#names').innerHTML = (await r.json()).players.map(p => `<option value="${p.name}">`).join('');
});
q('#player').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
async function go() {
  const url = `/api/recommend?player=${encodeURIComponent(q('#player').value)}&n=${q('#n').value}&method=${q('#method').value}`;
  const r = await fetch(url), d = await r.json();
  if (!r.ok) { q('#out').innerHTML = `<p class="err">${d.error}</p>`; q('#profile').innerHTML = ''; return; }
  q('#profile').innerHTML = `<b>${d.target.name}</b> (${d.target.team}) &mdash; ` + d.target.traits.map(t => `<span class="tag">${t}</span>`).join('');
  q('#out').innerHTML = '<table><tr><th>#</th><th>Player</th><th>Score</th><th>PTS / REB / AST</th><th>Shared traits</th></tr>' +
    d.recommendations.map(x => `<tr><td>${x.rank}</td><td>${x.name}<br><span class="muted">${x.team} &middot; ${x.pos}</span></td>
    <td>${x.score}</td><td>${x.pts} / ${x.trb} / ${x.ast}</td>
    <td>${x.shared_traits.map(t => `<span class="tag">${t}</span>`).join('')}</td></tr>`).join('') + '</table>';
}
</script></body></html>"""


def create_app(model_dir: str = config.ARTIFACTS_DIR, triples_csv_path: str = config.KG_CSV) -> Flask:
    app = Flask(__name__)
    rec = PlayerRecommender(model_dir=model_dir, triples_csv_path=triples_csv_path)
    app.config["RECOMMENDER"] = rec
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", players=len(rec.player_ids), embeddings=rec._emb_unit is not None,
                       model=rec.meta.get("model"))

    @app.get("/api/players")
    def players():
        return jsonify(players=rec.search(request.args.get("q", ""), int(request.args.get("limit", 10))))

    @app.get("/api/player/<player_id>")
    def player(player_id):
        try:
            return jsonify(rec.profile(player_id))
        except KeyError as exc:
            return jsonify(error=str(exc.args[0])), 404

    @app.get("/api/recommend")
    def recommend():
        query = request.args.get("player", "").strip()
        if not query:
            return jsonify(error="Query parameter 'player' is required."), 400
        try:
            n = int(request.args.get("n", 5))
        except ValueError:
            return jsonify(error="'n' must be an integer."), 400
        if not 1 <= n <= 25:
            return jsonify(error="'n' must be between 1 and 25."), 400
        default = "embedding" if rec._emb_unit is not None else "rules"
        method = request.args.get("method", default)
        if method not in ("embedding", "rules"):
            return jsonify(error="'method' must be 'embedding' or 'rules'."), 400
        try:
            return jsonify(target=rec.profile(query), method=method,
                           recommendations=rec.recommend(query, n, method))
        except KeyError as exc:
            return jsonify(error=str(exc.args[0])), 404
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=config.ARTIFACTS_DIR)
    parser.add_argument("--data", default=config.KG_CSV)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.artifacts, args.data).run(port=args.port)
