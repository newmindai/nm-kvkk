"""Local demo: entities + relations extraction page for any GLiNER2 checkpoint of this repository.

Stdlib-only HTTP server in one file (the page is embedded below): loads the checkpoint once, serves the UI, runs
on-device inference (CUDA, MPS or CPU). Editable label / relation chips, threshold slider, span highlighting,
per-relation confidence, and the type-constraint filter (configs/relation_constraints.json: the (head label,
tail label) pairs seen in training per relation type).

Usage:
  python scripts/demo.py                                    # models/gliner2.5-kvkk-tr-v1, then open http://localhost:8765
  python scripts/demo.py --model models/gliner2.5-mursit-kvkk-tr-v1 --port 8765 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"

logger = logging.getLogger("demo")

INDEX_HTML = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KVKK Varlık + İlişki Demo</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root { color-scheme: light dark;
    --page:#f7f8f9; --surface:#fdfdfc; --ink:#101112; --ink-2:#4e5459; --muted:#86898d;
    --border:rgba(16,17,18,.12); --accent:#2a78d6; --tile:#eef2f6; --good:#006300; }
  @media (prefers-color-scheme: dark) { :root {
    --page:#0e0f10; --surface:#1a1b1c; --ink:#f4f4f2; --ink-2:#b9bcb8; --muted:#85888c;
    --border:rgba(255,255,255,.12); --accent:#3987e5; --tile:#202224; --good:#0ca30c; } }
  * { box-sizing:border-box; }
  body { background:var(--page); color:var(--ink); margin:0; font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif; }
  main { max-width:1060px; margin:0 auto; padding:36px 24px 80px; }
  h1 { font-family:"IBM Plex Serif",Georgia,serif; font-size:28px; margin:6px 0 2px; letter-spacing:-.01em; }
  .sub { color:var(--ink-2); font-size:13.5px; margin:0 0 20px; }
  .sub code { font-family:"IBM Plex Mono",monospace; background:var(--tile); border-radius:4px; padding:1px 5px; font-size:.9em; }
  .eyebrow { font:500 11px/1 "IBM Plex Mono",monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
  .panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; margin-top:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  textarea { width:100%; min-height:170px; background:var(--page); color:var(--ink); border:1px solid var(--border);
    border-radius:8px; padding:12px; font:14px/1.6 "IBM Plex Sans",sans-serif; resize:vertical; }
  textarea:focus, input:focus, button:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
  .examples button { background:var(--tile); border:1px solid var(--border); color:var(--ink-2); border-radius:16px;
    padding:6px 13px; font:12.5px "IBM Plex Sans",sans-serif; cursor:pointer; }
  .examples button:hover { color:var(--ink); border-color:var(--accent); }
  h3 { margin:0 0 8px; font:600 11.5px/1 "IBM Plex Sans",sans-serif; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { display:inline-flex; align-items:center; gap:6px; background:var(--tile); border:1px solid var(--border);
    border-radius:14px; padding:4px 6px 4px 11px; font-size:13px; }
  .chip .dot { width:9px; height:9px; border-radius:3px; }
  .chip button { border:0; background:none; color:var(--muted); cursor:pointer; font-size:14px; line-height:1; padding:0 3px; }
  .chip button:hover { color:var(--ink); }
  .adder { display:flex; gap:6px; margin-top:8px; }
  .adder input { flex:1; max-width:280px; background:var(--page); color:var(--ink); border:1px solid var(--border);
    border-radius:7px; padding:6px 10px; font:13px "IBM Plex Sans",sans-serif; }
  .adder button, #run { background:var(--accent); color:#fff; border:0; border-radius:7px; padding:7px 14px;
    font:600 13px "IBM Plex Sans",sans-serif; cursor:pointer; }
  #run { font-size:14.5px; padding:10px 26px; }
  #run:disabled { opacity:.55; cursor:wait; }
  .meta { color:var(--muted); font-size:12.5px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:860px){ .grid2 { grid-template-columns:1fr; } }
  #doc { white-space:pre-wrap; font:14px/1.85 "IBM Plex Sans",sans-serif; }
  .ent { border-radius:4px; padding:1px 3px; border-bottom:2px solid; cursor:default; }
  .ent .lbl { font:500 9.5px/1 "IBM Plex Mono",monospace; letter-spacing:.02em; opacity:.8; margin-left:3px; vertical-align:super; }
  .rel-row { display:flex; align-items:center; gap:8px; padding:7px 4px; border-bottom:1px solid var(--border); font-size:13.5px; flex-wrap:wrap; }
  .rel-row:last-child { border-bottom:0; }
  .rel-type { font:500 11.5px "IBM Plex Mono",monospace; background:var(--tile); border-radius:5px; padding:3px 8px; color:var(--ink-2); }
  .arrow { color:var(--muted); }
  .surf { font-weight:500; }
  details { margin-top:14px; } summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  pre { background:var(--tile); border-radius:8px; padding:12px; overflow-x:auto; font:12px/1.5 "IBM Plex Mono",monospace; color:var(--ink-2); }
  input[type=range] { accent-color:var(--accent); vertical-align:middle; }
  .empty { color:var(--muted); font-size:13px; }
  #err { color:#d03b3b; font-size:13px; margin-top:8px; white-space:pre-wrap; }
</style>
</head>
<body>
<main>
  <div class="eyebrow">rele07tr · gliner2.5 · yerel çıkarım (on-device)</div>
  <h1>KVKK Varlık + İlişki Demo</h1>
  <p class="sub">Model: <code id="modelinfo">yükleniyor…</code> — KVKK F1 0.923 · varlıklar 0.962 · ilişkiler 0.693 (Türkçe şema).
     Metni yapıştır, etiketleri düzenle, <b>Çıkar</b>'a bas. Tüm çıkarım bu cihazda çalışır.</p>

  <div class="panel">
    <div class="row examples" id="examples"><span class="meta">Örnekler:</span></div>
    <textarea id="text" spellcheck="false" placeholder="Türkçe hukuki metni buraya yapıştırın…"></textarea>
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>Varlık etiketleri</h3>
      <div class="chips" id="entchips"></div>
      <div class="adder"><input id="entadd" placeholder="yeni etiket (ör. pasaport numarası)"><button onclick="addChip('ent')">ekle</button></div>
    </div>
    <div class="panel">
      <h3>İlişki türleri</h3>
      <div class="chips" id="relchips"></div>
      <div class="adder"><input id="reladd" list="allrels" placeholder="yeni ilişki (ör. drivers_license_of)"><button onclick="addChip('rel')">ekle</button></div>
      <datalist id="allrels"></datalist>
    </div>
  </div>

  <div class="panel row">
    <button id="run" onclick="run()">Çıkar</button>
    <label class="meta">eşik <input type="range" id="thr" min="0.1" max="0.9" step="0.05" value="0.5"
      oninput="thrv.textContent=this.value"> <span id="thrv">0.5</span></label>
    <label class="meta"><input type="checkbox" id="typefilter" checked onchange="rerender()"> tip filtresi</label>
    <label class="meta"><input type="checkbox" id="deepscan" checked> derin tarama (nadir ilişkileri bulur, daha yavaş)</label>
    <span class="meta" id="timing"></span>
    <div id="err"></div>
  </div>

  <div class="grid2">
    <div class="panel"><h3>Belge — vurgulanmış varlıklar</h3><div id="doc" class="empty">Henüz çıkarım yapılmadı.</div></div>
    <div class="panel"><h3>İlişkiler</h3><div id="rels" class="empty">Henüz çıkarım yapılmadı.</div></div>
  </div>

  <details><summary>ham JSON çıktısı</summary><pre id="raw"></pre></details>
</main>

<script>
const PALETTE = ["#2a78d6","#eb6834","#1baf7a","#c98500","#d55181","#008300","#7a6de0","#d03b3b"];
let ents = [], rels = [];
const colorOf = label => {
  const i = ents.indexOf(label);
  const base = PALETTE[(i >= 0 ? i : 0) % PALETTE.length];
  return base;
};

function renderChips(kind) {
  const list = kind === "ent" ? ents : rels;
  const el = document.getElementById(kind + "chips");
  el.innerHTML = "";
  list.forEach((name, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = (kind === "ent" ? `<span class="dot" style="background:${PALETTE[i % PALETTE.length]}"></span>` : "")
      + `<span>${name}</span><button title="kaldır">×</button>`;
    chip.querySelector("button").onclick = () => { list.splice(i, 1); renderChips(kind); };
    el.appendChild(chip);
  });
}
function addChip(kind) {
  const input = document.getElementById(kind + "add");
  const value = input.value.trim();
  const list = kind === "ent" ? ents : rels;
  if (value && !list.includes(value)) { list.push(value); renderChips(kind); }
  input.value = "";
}
["ent","rel"].forEach(kind => document.getElementById(kind + "add")
  .addEventListener("keydown", event => { if (event.key === "Enter") addChip(kind); }));

async function boot() {
  const d = await (await fetch("/defaults")).json();
  ents = d.entities; rels = d.relations;
  renderChips("ent"); renderChips("rel");
  document.getElementById("modelinfo").textContent = "rele07_final · " + d.device;
  const ex = document.getElementById("examples");
  d.examples.forEach(e => {
    const b = document.createElement("button");
    b.textContent = e.name;
    b.onclick = () => { document.getElementById("text").value = e.text; };
    ex.appendChild(b);
  });
  const dl = document.getElementById("allrels");
  (d.all_relations || []).forEach(r => { const o = document.createElement("option"); o.value = r; dl.appendChild(o); });
}

function escapeHtml(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// Length-preserving Turkish-safe casefold: JS "İ".toLowerCase() expands to
// TWO chars (i + combining dot), which shifts every later highlight index.
// Map İ->i and I->ı first so lowering never changes string length.
const fold = s => s.replace(/İ/g, "i").replace(/I/g, "ı").toLowerCase();

function highlight(text, entities) {
  // collect all occurrences of every mention, prefer longer mentions on overlap
  const marks = [];
  const lower = fold(text);
  for (const [label, mentions] of Object.entries(entities)) {
    for (const mention of mentions) {
      const m = fold(String(mention));
      if (!m) continue;
      let at = 0;
      while ((at = lower.indexOf(m, at)) !== -1) {
        marks.push({ start: at, end: at + m.length, label });
        at += Math.max(m.length, 1);
      }
    }
  }
  marks.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
  const kept = [];
  let cursorEnd = -1;
  for (const mark of marks) if (mark.start >= cursorEnd) { kept.push(mark); cursorEnd = mark.end; }
  let html = "", cursor = 0;
  for (const mark of kept) {
    const color = colorOf(mark.label);
    html += escapeHtml(text.slice(cursor, mark.start));
    html += `<span class="ent" style="background:${color}22;border-color:${color}" title="${mark.label}">`
          + escapeHtml(text.slice(mark.start, mark.end))
          + `<span class="lbl" style="color:${color}">${escapeHtml(mark.label)}</span></span>`;
    cursor = mark.end;
  }
  return html + escapeHtml(text.slice(cursor));
}

let lastRelations = {};
function renderRelations(relations) {
  lastRelations = relations;
  const el = document.getElementById("rels");
  const filterOn = document.getElementById("typefilter").checked;
  const rows = [];
  let filtered = 0;
  for (const [type, pairs] of Object.entries(relations)) {
    for (const pair of pairs) {
      const head = pair[0], tail = pair[1], conf = pair[2], violation = pair[3];
      if (violation && filterOn) { filtered++; continue; }
      const style = violation ? ' style="opacity:.5;text-decoration:line-through"'
                  : (conf != null && conf < 0.5 ? ' style="opacity:.55"' : "");
      const badge = conf != null ? `<span class="meta" style="margin-left:auto">${(conf*100).toFixed(0)}%${violation ? " · tip ihlali" : ""}</span>` : "";
      rows.push(`<div class="rel-row"${style}><span class="surf">${escapeHtml(String(head))}</span>`
        + `<span class="arrow">—</span><span class="rel-type">${escapeHtml(type)}</span><span class="arrow">→</span>`
        + `<span class="surf">${escapeHtml(String(tail))}</span>${badge}</div>`);
    }
  }
  if (filtered) rows.push(`<div class="rel-row meta">${filtered} ilişki tip filtresiyle gizlendi (kutuyu kaldırıp görebilirsiniz)</div>`);
  el.classList.toggle("empty", rows.length === 0);
  el.innerHTML = rows.length ? rows.join("") : "İlişki bulunamadı (eşiği düşürmeyi deneyin).";
}
function rerender() { renderRelations(lastRelations); }

async function run() {
  const button = document.getElementById("run");
  const err = document.getElementById("err");
  err.textContent = "";
  button.disabled = true; button.textContent = "Çıkarılıyor…";
  try {
    const response = await fetch("/extract", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: document.getElementById("text").value,
        entities: ents, relations: rels,
        threshold: parseFloat(document.getElementById("thr").value),
        deep: document.getElementById("deepscan").checked,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    const doc = document.getElementById("doc");
    doc.classList.remove("empty");
    doc.innerHTML = highlight(document.getElementById("text").value, data.entities || {});
    renderRelations(data.relations || {});
    document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
    document.getElementById("timing").textContent = `${data.elapsed_s} sn · ${data.device}`;
  } catch (error) {
    err.textContent = "Hata: " + error.message;
  } finally {
    button.disabled = false; button.textContent = "Çıkar";
  }
}
boot();
</script>
</body>
</html>
"""

DEFAULT_ENTITIES = [
    "ad soyad",
    "adres",
    "telefon numarası",
    "e-posta adresi",
    "kimlik numarası",
    "IBAN",
    "kredi kartı numarası",
    "doğum tarihi",
    "araç plakası",
    "şirket adı",
    "dava dosya numarası",
    "vergi kimlik numarası",
]
DEFAULT_RELATIONS = [
    "ikamet adresi sahibi",
    "telefon numarası sahibi",
    "e-posta adresi sahibi",
    "kimlik numarası sahibi",
    "banka hesabı sahibi",
    "kart sahibi",
    "araç plakası sahibi",
    "doğum tarihi sahibi",
    "işvereni olduğu kişi",
    "tarafı olduğu dosya",
    "vergi kimlik numarası sahibi",
    "kullanıcı hesabı sahibi",
    "eşi olduğu kişi",
]

HANDCRAFTED_EXAMPLE = (
    "İstanbul Anadolu 5. Asliye Hukuk Mahkemesi'nin 2025/312 E. sayılı dosyasında "
    "davacı Ayşe Nur Çelik (TC Kimlik No: 12345678901), Bağdat Caddesi No:112 D:4 "
    "Kadıköy/İstanbul adresinde ikamet etmekte olup kendisine 0532 447 18 92 numaralı "
    "telefondan ve ayse.celik@ornek.com adresinden ulaşılabilmektedir. Davalı Mehmet Kaya, "
    "Yılmaz İnşaat A.Ş. bünyesinde çalışmakta olup maaşı TR33 0006 1005 1978 6457 8413 26 "
    "IBAN numaralı hesaba yatırılmaktadır. Tanık beyanına göre 34 KL 2921 plakalı araç "
    "olay günü Mehmet Kaya tarafından kullanılmıştır."
)


class ModelHolder:
    def __init__(self, checkpoint: Path, device: str):
        import torch

        from gliner2 import AutoExtractor

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        logger.info("loading %s on %s ...", checkpoint, device)
        started = time.time()
        self.model = AutoExtractor.from_pretrained(str(checkpoint), map_location=device)
        self.model.eval()
        self.device = device
        logger.info("model ready in %.1f s", time.time() - started)
        self.constraints = load_relation_constraints()

    def extract(
        self, text: str, entities: list, relations: list, descriptions: dict, threshold: float, deep: bool = True
    ) -> dict:
        started = time.time()
        if relations and deep:
            # Deep scan: one entity pass, then each relation type in its own
            # focused pass — broad schemas dilute rare relations (a spouse link
            # scores 0.85 alone but vanishes among 13 competing types).
            out = self.model.extract(
                text, self.model.create_schema().entities(entities), threshold=threshold, include_confidence=True
            )
            out.setdefault("relation_extraction", {})
            for rtype in relations:
                one = self.model.extract(
                    text,
                    self.model.create_schema().entities(entities).relations({rtype: descriptions.get(rtype, "")}),
                    threshold=threshold,
                    include_confidence=True,
                )
                out["relation_extraction"][rtype] = (one.get("relation_extraction") or {}).get(rtype, [])
        else:
            schema = self.model.create_schema().entities(entities)
            if relations:
                schema = schema.relations({r: descriptions.get(r, "") for r in relations})
            out = self.model.extract(text, schema, threshold=threshold, include_confidence=True)
        elapsed = time.time() - started
        relations = {}
        for rtype, pairs in (out.get("relation_extraction") or {}).items():
            norm = []
            for pair in pairs:
                # normalise across output shapes: (head, tail) tuples or dicts with confidence
                if isinstance(pair, dict):
                    head = pair.get("head")
                    tail = pair.get("tail")
                    head_text = head.get("text") if isinstance(head, dict) else head
                    tail_text = tail.get("text") if isinstance(tail, dict) else tail
                    conf = pair.get("confidence")
                    if conf is None and isinstance(head, dict):
                        conf = (
                            min(head.get("confidence", 1.0), (tail or {}).get("confidence", 1.0))
                            if isinstance(tail, dict)
                            else head.get("confidence")
                        )
                    norm.append([head_text, tail_text, round(float(conf), 3) if conf is not None else None])
                else:
                    norm.append([pair[0], pair[1], round(float(pair[2]), 3) if len(pair) > 2 else None])
            norm.sort(key=lambda x: -(x[2] or 0))
            relations[rtype] = norm
        ent_out = out.get("entities", {})
        # entities may also come back as dicts under include_confidence
        entities = {}
        for label, mentions in (ent_out or {}).items():
            entities[label] = [m.get("text") if isinstance(m, dict) else m for m in mentions]
        flag_type_violations(entities, relations, self.constraints)
        return {
            "entities": entities,
            "relations": relations,
            "elapsed_s": round(elapsed, 2),
            "device": self.device,
        }


def flag_type_violations(entities: dict, relations: dict, constraints: dict) -> None:
    """Append a violation flag to each [head, tail, conf] pair, in place.

    Constraint source: the (head_label, tail_label) combinations observed in
    training per relation type. A side of a pair is judged:
      - valid    if its surface belongs to one of the labels that side allows;
      - invalid  if every label that side allows was among the REQUESTED
                 entity labels (so we would have seen it) and the surface
                 belongs to none of them — "spouse needs a person name, you
                 asked for person names, 1837 is not one";
      - unknown  otherwise (an allowed label was not requested) — fail open.
    A pair violates if either side is invalid, or both sides are typed and
    the combination never occurs in training.
    """
    requested = set(entities.keys())
    surface2label = {}
    for label, mentions in entities.items():
        for mention in mentions:
            surface2label.setdefault(str(mention).lower(), label)

    def side_state(surface, allowed_labels):
        label = surface2label.get(str(surface).lower())
        if label in allowed_labels:
            return "valid", label
        if allowed_labels <= requested:
            return "invalid", label
        return "unknown", label

    for rtype, pairs in relations.items():
        allowed = constraints.get(rtype)
        for pair in pairs:
            violation = False
            if allowed:
                head_allowed = {h for h, _ in allowed}
                tail_allowed = {t for _, t in allowed}
                head_state, head_label = side_state(pair[0], head_allowed)
                tail_state, tail_label = side_state(pair[1], tail_allowed)
                if head_state == "invalid" or tail_state == "invalid":
                    violation = True
                elif head_label and tail_label and (head_label, tail_label) not in allowed:
                    violation = True
            pair.append(violation)


def load_relation_constraints() -> dict:
    """relation type -> list of allowed [head_entity_label, tail_entity_label] pairs (derived from training data)."""
    f = CONFIGS / "relation_constraints.json"
    if not f.exists():
        return {}
    return {k: {tuple(p) for p in v} for k, v in json.loads(f.read_text(encoding="utf-8")).items()}


def load_relation_descriptions() -> dict:
    local = CONFIGS / "relation_descriptions.json"  # the 92 Turkish relation names of the production model
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    mapping = PROJECT_ROOT / "configs/labels/e07_mapping.json"
    if mapping.exists():
        return json.loads(mapping.read_text(encoding="utf-8")).get("relation_descriptions", {})
    return {}


def default_model_path() -> Path:
    return PROJECT_ROOT / "models/gliner2.5-kvkk-tr-v1"


def load_examples() -> list:
    """The built-in example, plus the first few e07 test documents when that (gitignored) file is present."""
    examples = [{"name": "Dava dilekçesi (el yazımı örnek)", "text": HANDCRAFTED_EXAMPLE}]
    test_file = PROJECT_ROOT / "datasets/kvkk_relations/e07/entities_test_e07.jsonl"
    if test_file.exists():
        with test_file.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= 4:
                    break
                record = json.loads(line)
                examples.append({"name": f"e07 test {record['id']}", "text": record["input"]})
    return examples


class Handler(BaseHTTPRequestHandler):
    holder: ModelHolder = None
    descriptions: dict = {}
    examples: list = []

    def _send(self, payload, status=200, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
        elif self.path == "/defaults":
            self._send(
                {
                    "entities": DEFAULT_ENTITIES,
                    "relations": DEFAULT_RELATIONS,
                    "all_relations": sorted(self.descriptions),
                    "examples": self.examples,
                    "device": self.holder.device,
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/extract":
            self._send({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            text = req.get("text", "").strip()
            if not text:
                self._send({"error": "empty text"}, status=400)
                return
            result = self.holder.extract(
                text,
                [e for e in req.get("entities", []) if e.strip()],
                [r for r in req.get("relations", []) if r.strip()],
                self.descriptions,
                float(req.get("threshold", 0.5)),
                deep=bool(req.get("deep", True)),
            )
            self._send(result)
        except Exception as error:  # surface errors to the UI instead of dying
            logger.exception("extract failed")
            self._send({"error": f"{type(error).__name__}: {error}"}, status=500)

    def log_message(self, fmt, *args):
        logger.info("%s " + fmt, self.address_string(), *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    Handler.holder = ModelHolder(args.model, args.device)
    Handler.descriptions = load_relation_descriptions()
    Handler.examples = load_examples()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("demo ready → http://localhost:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
