"""GitDB REST API Server — full CRUD for vectors, documents, and tables."""

import json
import re
import threading
import time
import traceback
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# ─── HTML/CSS/JS Dashboard ──────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GitDB Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#1a1b1e;--bg2:#222326;--bg3:#2a2b2f;--border:#333538;
  --text:#e0e0e0;--dim:#888;--green:#4ade80;--blue:#60a5fa;
  --red:#f87171;--yellow:#fbbf24;--cyan:#22d3ee;
  --hot:#ef4444;--cold:#3b82f6;
}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Cascadia Code','JetBrains Mono',Consolas,monospace;font-size:13px;line-height:1.5}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}

/* Layout */
.shell{display:flex;flex-direction:column;height:100vh}
header{background:var(--bg2);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;gap:20px;flex-shrink:0}
header .logo{font-size:18px;font-weight:700;letter-spacing:1px;color:var(--green)}
header .logo span{color:var(--blue)}
header .meta{color:var(--dim);font-size:12px;display:flex;gap:16px}
header .meta .branch{color:var(--green)}
header .meta .hash{color:var(--yellow)}

.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:220px;background:var(--bg2);border-right:1px solid var(--border);padding:16px;overflow-y:auto;flex-shrink:0}
.content{flex:1;overflow-y:auto;padding:24px}

/* Search */
.search-box{position:relative;margin-bottom:20px}
.search-box input{width:100%;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:10px 14px 10px 36px;border-radius:6px;font-family:inherit;font-size:13px;outline:none;transition:border-color .2s}
.search-box input:focus{border-color:var(--blue)}
.search-box .icon{position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--dim);font-size:14px}
.search-box .hint{position:absolute;right:12px;top:50%;transform:translateY(-50%);color:var(--dim);font-size:11px;background:var(--bg);border:1px solid var(--border);padding:1px 6px;border-radius:3px}

/* Cards */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:14px 16px;margin-bottom:10px;animation:fadeIn .3s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.card .doc-preview{color:var(--dim);margin-top:6px;white-space:pre-wrap;overflow:hidden;max-height:60px;font-size:12px}
.card .meta-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.card .id{color:var(--cyan);font-size:12px}

.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge-score{background:var(--bg3)}

/* Commit timeline */
.timeline{position:relative;padding-left:24px}
.timeline::before{content:'';position:absolute;left:8px;top:0;bottom:0;width:2px;background:var(--border)}
.commit{position:relative;padding:8px 0 16px}
.commit .dot{position:absolute;left:-20px;top:12px;width:10px;height:10px;border-radius:50%;background:var(--blue);border:2px solid var(--bg)}
.commit .hash{color:var(--yellow);font-size:12px;cursor:pointer}
.commit .hash:hover{text-decoration:underline}
.commit .msg{margin-top:2px}
.commit .ts{color:var(--dim);font-size:11px}
.commit .stats{font-size:11px;margin-top:2px}
.commit .stats .add{color:var(--green)}
.commit .stats .del{color:var(--red)}
.commit .stats .mod{color:var(--yellow)}
.commit .branch-label{display:inline-block;background:var(--green);color:#000;padding:0 6px;border-radius:3px;font-size:10px;font-weight:700;margin-left:6px;vertical-align:middle}

/* Sidebar */
.sidebar h3{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.branch-list{list-style:none}
.branch-list li{padding:5px 8px;border-radius:4px;cursor:pointer;font-size:12px;transition:background .15s}
.branch-list li:hover{background:var(--bg3)}
.branch-list li.current{color:var(--green);font-weight:700}
.branch-list li.current::before{content:'● '}
.tag-list{list-style:none;margin-top:16px}
.tag-list li{padding:3px 8px;font-size:12px;color:var(--yellow)}
.tag-list li::before{content:'⚑ '}

/* AC Heatmap */
.heatmap-section{margin-top:24px}
.heatmap-section h3{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.heatmap-section h3 .live{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.heatmap{display:flex;flex-wrap:wrap;gap:2px}
.heatmap .cell{width:14px;height:14px;border-radius:2px;transition:background .3s}
.ac-stats{color:var(--dim);font-size:11px;margin-top:8px}

/* Tabs */
.tabs{display:flex;gap:2px;margin-bottom:16px}
.tabs button{background:var(--bg2);border:1px solid var(--border);border-bottom:none;color:var(--dim);padding:6px 16px;cursor:pointer;font-family:inherit;font-size:12px;border-radius:4px 4px 0 0;transition:all .15s}
.tabs button.active{background:var(--bg3);color:var(--text);border-color:var(--blue)}
.tabs button:hover{color:var(--text)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* Footer */
footer{background:var(--bg2);border-top:1px solid var(--border);padding:8px 24px;font-size:11px;color:var(--dim);display:flex;gap:24px;flex-shrink:0}
footer span{display:flex;align-items:center;gap:4px}

/* Responsive */
@media(max-width:768px){
  .sidebar{display:none}
  .content{padding:12px}
  header{flex-wrap:wrap;padding:8px 12px}
}

/* Loading */
.spinner{display:inline-block;width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{color:var(--dim);padding:40px;text-align:center}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="logo">Git<span>DB</span></div>
    <div class="meta">
      <span class="path" id="hdr-path"></span>
      <span>branch: <span class="branch" id="hdr-branch"></span></span>
      <span>HEAD: <span class="hash" id="hdr-head"></span></span>
    </div>
  </header>

  <div class="main">
    <aside class="sidebar">
      <h3>Branches</h3>
      <ul class="branch-list" id="branch-list"></ul>
      <h3 style="margin-top:20px">Tags</h3>
      <ul class="tag-list" id="tag-list"></ul>

      <div class="heatmap-section" id="ac-section" style="display:none">
        <h3>Emirati AC <span class="live"></span></h3>
        <div class="heatmap" id="ac-heatmap"></div>
        <div class="ac-stats" id="ac-stats"></div>
      </div>
    </aside>

    <div class="content">
      <div class="search-box">
        <span class="icon">&#x1F50D;</span>
        <input type="text" id="search-input" placeholder="Semantic search..." autocomplete="off">
        <span class="hint">/</span>
      </div>

      <div id="search-results"></div>

      <div class="tabs">
        <button class="active" data-tab="commits">Commits</button>
        <button data-tab="query">Query Builder</button>
      </div>

      <div id="tab-commits" class="tab-panel active">
        <div class="timeline" id="commit-log"></div>
      </div>

      <div id="tab-query" class="tab-panel">
        <div class="card">
          <div style="margin-bottom:8px;color:var(--dim)">SELECT (structured query)</div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
            <input id="q-where" placeholder='where: {"key":"val"}' style="flex:1;min-width:200px;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:4px;font-family:inherit;font-size:12px">
            <input id="q-fields" placeholder="fields: id,document" style="width:180px;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:4px;font-family:inherit;font-size:12px">
            <input id="q-limit" type="number" placeholder="limit: 20" value="20" style="width:80px;background:var(--bg3);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:4px;font-family:inherit;font-size:12px">
            <button onclick="runSelect()" style="background:var(--blue);color:#000;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;font-family:inherit;font-weight:600">Run</button>
          </div>
          <pre id="q-results" style="color:var(--dim);font-size:12px;max-height:400px;overflow:auto;white-space:pre-wrap"></pre>
        </div>
      </div>
    </div>
  </div>

  <footer>
    <span id="ft-vectors"></span>
    <span id="ft-commits"></span>
    <span id="ft-dim"></span>
    <span id="ft-device"></span>
  </footer>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// ── API helpers ──
async function api(path, opts) {
  try {
    const r = await fetch(path, opts);
    return await r.json();
  } catch(e) { console.error(path, e); return null; }
}

// ── Init ──
async function init() {
  const [status, branches, tags] = await Promise.all([
    api('/api/status'), api('/api/branches'), api('/api/tags')
  ]);
  if (status && status.ok) {
    const d = status.data;
    $('#hdr-branch').textContent = d.branch || '—';
    $('#hdr-head').textContent = d.head ? d.head.slice(0,8) : 'none';
    $('#hdr-path').textContent = d.path || '';
    $('#ft-vectors').textContent = 'vectors: ' + (d.active_rows || 0);
    $('#ft-dim').textContent = 'dim: ' + (d.dim || '?');
    $('#ft-device').textContent = 'device: ' + (d.device || '?');
  }
  if (branches && branches.ok) renderBranches(branches.data);
  if (tags && tags.ok) renderTags(tags.data);
  loadLog();
  checkAC();
}

// ── Branches ──
function renderBranches(data) {
  const ul = $('#branch-list');
  ul.innerHTML = '';
  const current = data.current;
  for (const [name, hash] of Object.entries(data.branches || {})) {
    const li = document.createElement('li');
    li.textContent = name;
    if (name === current) li.className = 'current';
    li.onclick = () => loadLog(name);
    ul.appendChild(li);
  }
}

// ── Tags ──
function renderTags(data) {
  const ul = $('#tag-list');
  ul.innerHTML = '';
  for (const [name, hash] of Object.entries(data.tags || {})) {
    const li = document.createElement('li');
    li.textContent = name + ' → ' + hash.slice(0,8);
    ul.appendChild(li);
  }
  if (!Object.keys(data.tags || {}).length) {
    ul.innerHTML = '<li style="color:var(--dim)">no tags</li>';
  }
}

// ── Commit log ──
let totalCommits = 0;
async function loadLog(branch) {
  const el = $('#commit-log');
  el.innerHTML = '<div class="spinner"></div>';
  const resp = await api('/api/log?limit=100' + (branch ? '&branch=' + branch : ''));
  if (!resp || !resp.ok) {
    el.innerHTML = '<div class="empty">No commits yet</div>';
    totalCommits = 0;
    $('#ft-commits').textContent = 'commits: 0';
    return;
  }
  const data = resp.data;
  if (!data.commits || !data.commits.length) {
    el.innerHTML = '<div class="empty">No commits yet</div>';
    totalCommits = 0;
    $('#ft-commits').textContent = 'commits: 0';
    return;
  }
  totalCommits = data.commits.length;
  $('#ft-commits').textContent = 'commits: ' + totalCommits;

  // Build branch→hash map for labels
  const brResp = await api('/api/branches');
  const branchMap = {};
  if (brResp && brResp.ok && brResp.data.branches) {
    for (const [name, hash] of Object.entries(brResp.data.branches)) {
      if (!branchMap[hash]) branchMap[hash] = [];
      branchMap[hash].push(name);
    }
  }

  el.innerHTML = '';
  for (const c of data.commits) {
    const div = document.createElement('div');
    div.className = 'commit';
    const ts = new Date(c.timestamp * 1000).toISOString().replace('T',' ').slice(0,16);
    let labels = '';
    if (branchMap[c.hash]) {
      labels = branchMap[c.hash].map(b => '<span class="branch-label">' + b + '</span>').join('');
    }
    div.innerHTML =
      '<div class="dot"></div>' +
      '<span class="hash" onclick="showCommit(\'' + c.hash + '\')">' + c.hash.slice(0,8) + '</span>' +
      labels +
      '<div class="msg">' + esc(c.message) + '</div>' +
      '<div class="ts">' + ts + '</div>' +
      '<div class="stats">' +
        (c.stats.added ? '<span class="add">+' + c.stats.added + '</span> ' : '') +
        (c.stats.removed ? '<span class="del">-' + c.stats.removed + '</span> ' : '') +
        (c.stats.modified ? '<span class="mod">~' + c.stats.modified + '</span>' : '') +
      '</div>';
    el.appendChild(div);
  }
}

// ── Show commit ──
async function showCommit(ref) {
  const resp = await api('/api/show/' + ref);
  if (!resp || !resp.ok) return;
  const data = resp.data;
  const ts = new Date(data.timestamp * 1000).toISOString().replace('T',' ').slice(0,19);
  alert('commit ' + data.hash + '\nParent: ' + (data.parent || 'none') +
    '\nDate: ' + ts + '\n\n' + data.message +
    '\n\n+' + data.stats.added + ' -' + data.stats.removed + ' ~' + data.stats.modified +
    '\n' + data.tensor_rows + ' rows | delta ' + data.delta_size_bytes + ' bytes');
}

// ── Search ──
$('#search-input').addEventListener('keydown', async e => {
  if (e.key !== 'Enter') return;
  const q = e.target.value.trim();
  if (!q) return;
  const el = $('#search-results');
  el.innerHTML = '<div class="spinner"></div> Searching...';
  const resp = await api('/api/vectors/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: q, k: 10})
  });
  if (!resp || !resp.ok || !resp.data.ids || !resp.data.ids.length) {
    el.innerHTML = '<div class="card">No results</div>';
    return;
  }
  const data = resp.data;
  el.innerHTML = '';
  for (let i = 0; i < data.ids.length; i++) {
    const card = document.createElement('div');
    card.className = 'card';
    const score = data.scores[i];
    const color = score > 0.7 ? 'var(--green)' : score > 0.4 ? 'var(--yellow)' : 'var(--dim)';
    card.innerHTML =
      '<div class="meta-row">' +
        '<span class="badge badge-score" style="color:' + color + '">' + score.toFixed(4) + '</span>' +
        '<span class="id">' + esc(data.ids[i]) + '</span>' +
      '</div>' +
      (data.documents[i] ? '<div class="doc-preview">' + esc(data.documents[i].slice(0,200)) + '</div>' : '') +
      (data.metadata[i] && Object.keys(data.metadata[i]).length ? '<div style="color:var(--dim);font-size:11px;margin-top:4px">' + esc(JSON.stringify(data.metadata[i])) + '</div>' : '');
    el.appendChild(card);
  }
});

// ── Select (query builder) ──
async function runSelect() {
  const whereStr = $('#q-where').value.trim();
  const fieldsStr = $('#q-fields').value.trim();
  const limit = parseInt($('#q-limit').value) || 20;
  let where = null;
  if (whereStr) {
    try { where = JSON.parse(whereStr); } catch(e) {
      $('#q-results').textContent = 'Invalid JSON in where clause';
      return;
    }
  }
  const fields = fieldsStr ? fieldsStr.split(',').map(s => s.trim()) : null;
  const resp = await api('/api/docs/find', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({where, fields, limit})
  });
  $('#q-results').textContent = resp ? JSON.stringify(resp, null, 2) : 'Error';
}

// ── AC Heatmap ──
let acRunning = false;
async function checkAC() {
  const resp = await api('/api/ac/status');
  if (!resp || !resp.ok) return;
  acRunning = resp.data.running;
  if (acRunning) {
    $('#ac-section').style.display = '';
    refreshHeatmap();
    setInterval(refreshHeatmap, 3000);
  }
}

async function refreshHeatmap() {
  const resp = await api('/api/ac/heatmap');
  if (!resp || !resp.ok) return;
  const data = resp.data;
  const el = $('#ac-heatmap');
  el.innerHTML = '';
  const activations = data.activations || {};
  const entries = Object.entries(activations).sort((a,b) => b[1] - a[1]);
  const maxCells = 200;
  const shown = entries.slice(0, maxCells);
  for (const [idx, level] of shown) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const r = Math.round(level * 239 + (1-level) * 59);
    const g = Math.round(level * 68 + (1-level) * 130);
    const b = Math.round(level * 68 + (1-level) * 246);
    cell.style.background = 'rgb(' + r + ',' + g + ',' + b + ')';
    cell.title = 'idx ' + idx + ': ' + level.toFixed(3);
    el.appendChild(cell);
  }
  const stats = data.stats || {};
  $('#ac-stats').innerHTML =
    'cycles: ' + (stats.cycles || 0) +
    ' &middot; active: ' + (stats.active_vectors || 0) +
    ' &middot; peak: ' + (stats.peak_active || 0) +
    ' &middot; top: ' + (stats.top_score || 0);
}

// ── Tabs ──
for (const btn of $$('.tabs button')) {
  btn.onclick = () => {
    $$('.tabs button').forEach(b => b.classList.remove('active'));
    $$('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('#tab-' + btn.dataset.tab).classList.add('active');
  };
}

// ── Keyboard shortcut ──
document.addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    $('#search-input').focus();
  }
});

// ── Helpers ──
function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

init();
</script>
</body>
</html>"""


# ─── Threaded HTTP Server ─────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


# ─── Route patterns ───────────────────────────────────────────────

# Pre-compiled patterns for dynamic routes
_RE_DIFF = re.compile(r"^/api/diff/([^/]+)/([^/]+)$")
_RE_SHOW = re.compile(r"^/api/show/([^/]+)$")
_RE_TABLE_ACTION = re.compile(r"^/api/tables/([^/]+)/(insert|select|update|delete)$")
_RE_TABLE_DROP = re.compile(r"^/api/tables/([^/]+)$")


# ─── Request Handler ─────────────────────────────────────────────

class GitDBHandler(BaseHTTPRequestHandler):
    """Full REST API handler for GitDB."""

    db = None  # Set by serve()

    def log_message(self, fmt, *args):
        """Suppress default request logging."""
        pass

    # ── Response helpers ──

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_ok(self, data, status=200):
        body = json.dumps({"ok": True, "data": data}, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _json_err(self, error, status=400):
        body = json.dumps({"ok": False, "error": str(error)}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _params(self):
        return parse_qs(urlparse(self.path).query)

    def _route(self):
        return urlparse(self.path).path

    # ── CORS preflight ──

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── GET ──

    def do_GET(self):
        route = self._route()
        params = self._params()

        try:
            if route == "/":
                self._send_html(DASHBOARD_HTML)

            elif route == "/api/health":
                self._json_ok({"status": "healthy", "timestamp": time.time()})

            elif route == "/api/status":
                self._h_status()

            elif route == "/api/log":
                limit = int(params.get("limit", [50])[0])
                self._h_log(limit)

            elif route == "/api/branches":
                self._h_branches()

            elif route == "/api/tags":
                self._h_tags()

            elif route == "/api/tables":
                self._h_list_tables()

            elif route == "/api/stash/list":
                self._h_stash_list()

            else:
                # Dynamic GET routes
                m = _RE_DIFF.match(route)
                if m:
                    self._h_diff(m.group(1), m.group(2))
                    return
                m = _RE_SHOW.match(route)
                if m:
                    self._h_show(m.group(1))
                    return

                # Legacy compat
                if route == "/api/diff":
                    a = params.get("a", [None])[0]
                    b = params.get("b", [None])[0]
                    self._h_diff(a, b)
                elif route == "/api/show":
                    ref = params.get("ref", ["HEAD"])[0]
                    self._h_show(ref)
                elif route == "/api/ac/status":
                    self._h_ac_status()
                elif route == "/api/ac/heatmap":
                    self._h_ac_heatmap()
                else:
                    self._json_err("not found", 404)

        except Exception as e:
            self._json_err(str(e), 500)

    # ── POST ──

    def do_POST(self):
        route = self._route()

        try:
            body = self._read_body()

            # --- Git Operations ---
            if route == "/api/commit":
                self._h_commit(body)
            elif route == "/api/branch":
                self._h_branch_create(body)
            elif route == "/api/switch":
                self._h_switch(body)
            elif route == "/api/merge":
                self._h_merge(body)
            elif route == "/api/stash":
                self._h_stash(body)
            elif route == "/api/stash/pop":
                self._h_stash_pop(body)
            elif route == "/api/reset":
                self._h_reset(body)
            elif route == "/api/cherry-pick":
                self._h_cherry_pick(body)
            elif route == "/api/revert":
                self._h_revert(body)

            # --- Vector Operations ---
            elif route == "/api/vectors/add":
                self._h_vectors_add(body)
            elif route == "/api/vectors/query":
                self._h_vectors_query(body)

            # --- Document Operations ---
            elif route == "/api/docs/insert":
                self._h_docs_insert(body)
            elif route == "/api/docs/find":
                self._h_docs_find(body)
            elif route == "/api/docs/update":
                self._h_docs_update(body)
            elif route == "/api/docs/delete":
                self._h_docs_delete(body)
            elif route == "/api/docs/count":
                self._h_docs_count(body)
            elif route == "/api/docs/aggregate":
                self._h_docs_aggregate(body)

            # --- Table Operations ---
            elif route == "/api/tables/create":
                self._h_table_create(body)

            # Legacy compat
            elif route == "/api/query":
                self._h_vectors_query(body)
            elif route == "/api/select":
                self._h_legacy_select(body)

            else:
                # Dynamic POST routes: /api/tables/{name}/{action}
                m = _RE_TABLE_ACTION.match(route)
                if m:
                    table_name = m.group(1)
                    action = m.group(2)
                    if action == "insert":
                        self._h_table_insert(table_name, body)
                    elif action == "select":
                        self._h_table_select(table_name, body)
                    elif action == "update":
                        self._h_table_update(table_name, body)
                    elif action == "delete":
                        self._h_table_delete(table_name, body)
                else:
                    self._json_err("not found", 404)

        except Exception as e:
            self._json_err(str(e), 500)

    # ── DELETE ──

    def do_DELETE(self):
        route = self._route()

        try:
            # DELETE /api/vectors
            if route == "/api/vectors":
                body = self._read_body()
                self._h_vectors_delete(body)
            else:
                # DELETE /api/tables/{name}
                m = _RE_TABLE_DROP.match(route)
                if m and route != "/api/tables":
                    self._h_table_drop(m.group(1))
                else:
                    self._json_err("not found", 404)

        except Exception as e:
            self._json_err(str(e), 500)

    # ═══════════════════════════════════════════════════════════════
    # Handler implementations
    # ═══════════════════════════════════════════════════════════════

    # ── Health / Status ──

    def _h_status(self):
        db = self.db
        s = db.status()
        s["path"] = str(db.path)
        s["dim"] = db.dim
        s["device"] = db.device
        self._json_ok(s)

    # ── Git: Log ──

    def _h_log(self, limit):
        db = self.db
        entries = db.log(limit=limit)
        commits = []
        for c in entries:
            commits.append({
                "hash": c.hash,
                "parent": c.parent,
                "message": c.message,
                "timestamp": c.timestamp,
                "stats": {
                    "added": c.stats.added,
                    "removed": c.stats.removed,
                    "modified": c.stats.modified,
                },
            })
        self._json_ok({"commits": commits})

    # ── Git: Branches ──

    def _h_branches(self):
        db = self.db
        branches = db.branches()
        current = db.refs.current_branch
        self._json_ok({"current": current, "branches": branches})

    # ── Git: Tags ──

    def _h_tags(self):
        db = self.db
        tags = {}
        tags_dir = db.refs.tags_dir
        if tags_dir.exists():
            for p in tags_dir.rglob("*"):
                if p.is_file():
                    name = str(p.relative_to(tags_dir))
                    tags[name] = p.read_text().strip()
        self._json_ok({"tags": tags})

    # ── Git: Commit ──

    def _h_commit(self, body):
        message = body.get("message")
        if not message:
            self._json_err("message is required", 400)
            return
        try:
            commit_hash = self.db.commit(message)
            self._json_ok({"hash": commit_hash}, 201)
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Branch create ──

    def _h_branch_create(self, body):
        name = body.get("name")
        if not name:
            self._json_err("name is required", 400)
            return
        ref = body.get("ref", "HEAD")
        try:
            self.db.branch(name, ref=ref)
            self._json_ok({"branch": name}, 201)
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Switch ──

    def _h_switch(self, body):
        branch = body.get("branch")
        if not branch:
            self._json_err("branch is required", 400)
            return
        try:
            self.db.switch(branch)
            self._json_ok({"branch": branch, "head": self.db.refs.get_head_commit()})
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Merge ──

    def _h_merge(self, body):
        branch = body.get("branch")
        if not branch:
            self._json_err("branch is required", 400)
            return
        strategy = body.get("strategy", "union")
        try:
            result = self.db.merge(branch, strategy=strategy)
            self._json_ok({
                "commit_hash": result.commit_hash,
                "strategy": result.strategy,
                "conflicts": result.conflicts,
            })
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Stash ──

    def _h_stash(self, body):
        message = body.get("message", "WIP")
        try:
            self.db.stash(message=message)
            self._json_ok({"message": message})
        except ValueError as e:
            self._json_err(str(e), 400)

    def _h_stash_pop(self, body):
        index = body.get("index", -1)
        try:
            entry = self.db.stash_pop(index=index)
            self._json_ok({
                "message": entry.message,
                "timestamp": entry.timestamp,
            })
        except (ValueError, IndexError) as e:
            self._json_err(str(e), 400)

    def _h_stash_list(self):
        entries = self.db.stash_list()
        result = []
        for e in entries:
            result.append({
                "message": e.message,
                "timestamp": e.timestamp,
                "branch": e.branch,
            })
        self._json_ok({"stashes": result})

    # ── Git: Reset ──

    def _h_reset(self, body):
        ref = body.get("ref", "HEAD")
        try:
            self.db.reset(ref)
            self._json_ok({"ref": ref, "head": self.db.refs.get_head_commit()})
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Cherry-pick ──

    def _h_cherry_pick(self, body):
        ref = body.get("ref")
        if not ref:
            self._json_err("ref is required", 400)
            return
        try:
            commit_hash = self.db.cherry_pick(ref)
            self._json_ok({"hash": commit_hash})
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Revert ──

    def _h_revert(self, body):
        ref = body.get("ref")
        if not ref:
            self._json_err("ref is required", 400)
            return
        try:
            commit_hash = self.db.revert(ref)
            self._json_ok({"hash": commit_hash})
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Diff ──

    def _h_diff(self, a, b):
        if not a or not b:
            self._json_err("two refs required", 400)
            return
        try:
            d = self.db.diff(a, b)
            self._json_ok({
                "added_count": d.added_count,
                "removed_count": d.removed_count,
                "modified_count": d.modified_count,
                "added_ids": d.added_ids,
                "removed_ids": d.removed_ids,
                "modified_ids": d.modified_ids,
            })
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Git: Show ──

    def _h_show(self, ref):
        try:
            info = self.db.show(ref)
            self._json_ok(info)
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Vectors: Add ──

    def _h_vectors_add(self, body):
        import torch

        texts = body.get("texts")
        embeddings_raw = body.get("embeddings")
        documents = body.get("documents")
        metadata = body.get("metadata")
        ids = body.get("ids")

        try:
            if texts is not None:
                indices = self.db.add(
                    texts=texts,
                    documents=documents,
                    metadata=metadata,
                    ids=ids,
                )
            elif embeddings_raw is not None:
                embeddings = torch.tensor(embeddings_raw, dtype=torch.float32)
                indices = self.db.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadata=metadata,
                    ids=ids,
                )
            else:
                self._json_err("texts or embeddings required", 400)
                return

            self._json_ok({"indices": indices, "count": len(indices)}, 201)
        except (ValueError, RuntimeError) as e:
            self._json_err(str(e), 400)

    # ── Vectors: Query ──

    def _h_vectors_query(self, body):
        import torch

        text = body.get("text")
        vector_raw = body.get("vector")
        k = body.get("k", 10)
        where = body.get("where")

        try:
            if text is not None:
                results = self.db.query_text(text, k=k, where=where)
            elif vector_raw is not None:
                vector = torch.tensor(vector_raw, dtype=torch.float32)
                results = self.db.query(vector, k=k, where=where)
            else:
                self._json_err("text or vector required", 400)
                return

            self._json_ok({
                "ids": results.ids,
                "scores": [round(s, 6) for s in results.scores],
                "documents": results.documents,
                "metadata": results.metadata,
            })
        except (ValueError, RuntimeError) as e:
            self._json_err(str(e), 400)

    # ── Vectors: Delete ──

    def _h_vectors_delete(self, body):
        ids = body.get("ids")
        if not ids:
            self._json_err("ids is required", 400)
            return
        try:
            removed = self.db.remove(ids=ids)
            self._json_ok({"removed": removed, "count": len(removed)})
        except (ValueError, IndexError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Insert ──

    def _h_docs_insert(self, body):
        doc = body.get("doc")
        docs = body.get("docs")
        try:
            if docs is not None:
                ids = self.db.insert(docs)
                self._json_ok({"ids": ids, "count": len(ids)}, 201)
            elif doc is not None:
                _id = self.db.insert(doc)
                self._json_ok({"id": _id}, 201)
            else:
                self._json_err("doc or docs required", 400)
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Find ──

    def _h_docs_find(self, body):
        where = body.get("where")
        limit = body.get("limit")
        skip = body.get("skip", 0)
        try:
            rows = self.db.find(where=where, limit=limit, skip=skip)
            self._json_ok({"docs": rows, "count": len(rows)})
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Update ──

    def _h_docs_update(self, body):
        where = body.get("where")
        set_fields = body.get("set")
        if where is None or set_fields is None:
            self._json_err("where and set are required", 400)
            return
        try:
            count = self.db.update_docs(where, set_fields)
            self._json_ok({"updated": count})
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Delete ──

    def _h_docs_delete(self, body):
        where = body.get("where")
        if where is None:
            self._json_err("where is required", 400)
            return
        try:
            count = self.db.delete_docs(where)
            self._json_ok({"deleted": count})
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Count ──

    def _h_docs_count(self, body):
        where = body.get("where")
        try:
            count = self.db.count_docs(where=where)
            self._json_ok({"count": count})
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Documents: Aggregate ──

    def _h_docs_aggregate(self, body):
        group_by = body.get("group_by")
        if not group_by:
            self._json_err("group_by is required", 400)
            return
        agg_fn = body.get("agg_fn", "count")
        agg_field = body.get("agg_field")
        where = body.get("where")
        try:
            result = self.db.aggregate_docs(
                group_by=group_by,
                agg_field=agg_field,
                agg_fn=agg_fn,
                where=where,
            )
            self._json_ok({"groups": result})
        except (ValueError, TypeError) as e:
            self._json_err(str(e), 400)

    # ── Tables: Create ──

    def _h_table_create(self, body):
        name = body.get("name")
        if not name:
            self._json_err("name is required", 400)
            return
        columns = body.get("columns")
        try:
            self.db.create_table(name, columns)
            self._json_ok({"table": name}, 201)
        except ValueError as e:
            self._json_err(str(e), 400)

    # ── Tables: List ──

    def _h_list_tables(self):
        tables = self.db.list_tables()
        self._json_ok({"tables": tables})

    # ── Tables: Insert ──

    def _h_table_insert(self, table_name, body):
        row = body.get("row")
        rows = body.get("rows")
        try:
            if rows is not None:
                ids = self.db.insert_into(table_name, rows)
                self._json_ok({"ids": ids, "count": len(ids)}, 201)
            elif row is not None:
                _id = self.db.insert_into(table_name, row)
                self._json_ok({"id": _id}, 201)
            else:
                self._json_err("row or rows required", 400)
        except (ValueError, KeyError) as e:
            self._json_err(str(e), 400)

    # ── Tables: Select ──

    def _h_table_select(self, table_name, body):
        columns = body.get("columns")
        where = body.get("where")
        order_by = body.get("order_by")
        desc = body.get("desc", False)
        limit = body.get("limit")
        offset = body.get("offset", 0)
        try:
            rows = self.db.select_from(
                table_name,
                columns=columns,
                where=where,
                order_by=order_by,
                desc=desc,
                limit=limit,
                offset=offset,
            )
            self._json_ok({"rows": rows, "count": len(rows)})
        except (ValueError, KeyError) as e:
            self._json_err(str(e), 400)

    # ── Tables: Update ──

    def _h_table_update(self, table_name, body):
        where = body.get("where")
        set_fields = body.get("set")
        if where is None or set_fields is None:
            self._json_err("where and set are required", 400)
            return
        try:
            count = self.db.update_table(table_name, where, set_fields)
            self._json_ok({"updated": count})
        except (ValueError, KeyError) as e:
            self._json_err(str(e), 400)

    # ── Tables: Delete rows ──

    def _h_table_delete(self, table_name, body):
        where = body.get("where")
        if where is None:
            self._json_err("where is required", 400)
            return
        try:
            count = self.db.delete_from(table_name, where)
            self._json_ok({"deleted": count})
        except (ValueError, KeyError) as e:
            self._json_err(str(e), 400)

    # ── Tables: Drop ──

    def _h_table_drop(self, table_name):
        try:
            self.db.drop_table(table_name)
            self._json_ok({"dropped": table_name})
        except (ValueError, KeyError) as e:
            self._json_err(str(e), 400)

    # ── AC (Ambient Cognition) ──

    def _h_ac_status(self):
        stats = self.db.ac.stats()
        self._json_ok(stats)

    def _h_ac_heatmap(self):
        activations = self.db.ac.get_activation_map()
        stats = self.db.ac.stats()
        act_out = {str(k): round(v, 4) for k, v in activations.items()}
        self._json_ok({"activations": act_out, "stats": stats})

    # ── Legacy compat ──

    def _h_legacy_select(self, body):
        where = body.get("where")
        fields = body.get("fields")
        limit = body.get("limit", 20)
        try:
            rows = self.db.select(fields=fields, where=where, limit=limit)
            self._json_ok({"rows": rows, "count": len(rows)})
        except Exception as e:
            self._json_err(str(e), 500)


# ─── Server startup ──────────────────────────────────────────────

def serve(db, port=7474, open_browser=True):
    """Start the GitDB REST API server.

    Args:
        db: GitDB instance
        port: Port number (default 7474)
        open_browser: Open browser automatically
    """
    GitDBHandler.db = db
    server = ThreadedHTTPServer(("0.0.0.0", port), GitDBHandler)
    url = f"http://localhost:{port}"
    print(f"GitDB server running at {url}")
    print(f"  Dashboard:  {url}/")
    print(f"  API:        {url}/api/health")
    print(f"  Vectors:    {url}/api/vectors/add")
    print(f"  Documents:  {url}/api/docs/insert")
    print(f"  Tables:     {url}/api/tables")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


def cmd_serve(args):
    """CLI entry point for `gitdb serve`."""
    from gitdb.cli import load_db
    db = load_db(args)
    port = getattr(args, "port", 7474) or 7474
    no_browser = getattr(args, "no_browser", False)
    serve(db, port=port, open_browser=not no_browser)
