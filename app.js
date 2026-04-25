/* ====================================================================
   ONCOVISION dashboard logic
   ==================================================================== */

const API = "";  // same origin

const $ = (id) => document.getElementById(id);

const els = {
  dropzone:    $("dropzone"),
  fileInput:   $("file-input"),
  demoBtn:     $("demo-btn"),
  filename:    $("filename-display"),

  statusPill:  $("status-pill"),
  statusText:  $("status-text"),
  deviceText:  $("device-text"),
  paramsText:  $("params-text"),
  timeText:    $("time-text"),
  footerTime:  $("footer-time"),

  viewerStage: $("viewer-stage"),
  viewerTitle: $("viewer-title"),
  layerBase:   $("layer-base"),
  layerOverlay:$("layer-overlay"),
  layerBtns:   $("layer-btns"),
  opacity:     $("opacity"),
  opacityVal:  $("opacity-val"),
  crosshair:   $("crosshair"),
  coord:       $("coord"),

  verdictCard: $("verdict-card"),
  verdictClass:$("verdict-class"),
  verdictConf: $("verdict-conf"),
  verdictBand: $("verdict-band"),

  probBenign:    $("prob-benign"),
  probBenignVal: $("prob-benign-val"),
  probMalignant: $("prob-malignant"),
  probMalignantVal: $("prob-malignant-val"),

  ecgPath: $("ecg-path"),

  mArea:     $("m-area"),
  mAreaPct:  $("m-area-pct"),
  mCentroid: $("m-centroid"),
  mCompact:  $("m-compact"),
  mIrreg:    $("m-irreg"),
  latency:   $("latency-val"),
  reqid:     $("reqid-val"),

  explainBody: $("explain-body"),
  caseList: $("case-list"),

  loader:     $("loader"),
  loaderStep: $("loader-step"),
};

let state = {
  cases: [],          // {id, ts, name, thumbBase, payload}
  active: null,       // active case id
  currentLayer: "segmentation",
};


// ---------- Clock + health ----------
function tick() {
  const d = new Date();
  const z = (n) => n.toString().padStart(2, "0");
  const utc = `${z(d.getUTCHours())}:${z(d.getUTCMinutes())}:${z(d.getUTCSeconds())}Z`;
  els.timeText.textContent = utc;
  els.footerTime.textContent = `${d.getUTCFullYear()}-${z(d.getUTCMonth()+1)}-${z(d.getUTCDate())} ${utc}`;
}
setInterval(tick, 1000); tick();

async function loadHealth() {
  try {
    const res = await fetch(`${API}/api/model`);
    const data = await res.json();
    els.deviceText.textContent = (data.device || "—").toUpperCase();
    els.paramsText.textContent = data.n_parameters_human || "—";
    els.statusPill.classList.add(data.loaded ? "ok" : "err");
    els.statusText.textContent = data.loaded ? "Model online" : "Demo mode (no checkpoint)";
  } catch (err) {
    els.statusPill.classList.add("err");
    els.statusText.textContent = "API offline";
  }
}
loadHealth();


// ---------- File upload ----------
function pickFile() { els.fileInput.click(); }

els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) handleFile(f);
});

["dragenter","dragover"].forEach(evt =>
  els.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);
["dragleave","drop"].forEach(evt =>
  els.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);
els.dropzone.addEventListener("drop", e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});


// Synthetic demo button — generates a simulated lesion image client-side
els.demoBtn.addEventListener("click", () => {
  const c = document.createElement("canvas");
  c.width = c.height = 224;
  const ctx = c.getContext("2d");
  // ultrasound-ish backdrop
  const grad = ctx.createRadialGradient(112,112,10,112,112,140);
  grad.addColorStop(0, "#3a3a3a");
  grad.addColorStop(1, "#0a0a0a");
  ctx.fillStyle = grad; ctx.fillRect(0,0,224,224);
  // grain
  const img = ctx.getImageData(0,0,224,224);
  for (let i=0;i<img.data.length;i+=4){
    const n = (Math.random()*70-35);
    img.data[i]   = Math.max(0,Math.min(255,img.data[i]+n));
    img.data[i+1] = Math.max(0,Math.min(255,img.data[i+1]+n));
    img.data[i+2] = Math.max(0,Math.min(255,img.data[i+2]+n));
  }
  ctx.putImageData(img,0,0);
  // irregular dark "lesion"
  ctx.fillStyle = "rgba(0,0,0,0.85)";
  ctx.beginPath();
  const cx=120+Math.random()*30, cy=110+Math.random()*30;
  for (let k=0;k<14;k++){
    const ang = (k/14)*Math.PI*2;
    const r = 28+Math.random()*22;
    const x = cx + r*Math.cos(ang), y = cy + r*Math.sin(ang);
    if (k===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.closePath(); ctx.fill();
  c.toBlob(b => {
    const f = new File([b], "synthetic_demo.png", {type:"image/png"});
    handleFile(f);
  }, "image/png");
});


async function handleFile(file) {
  els.filename.classList.add("has-file");
  els.filename.textContent = `→ ${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
  showLoader(true);
  cycleLoaderSteps();

  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const payload = await res.json();
    const caseId = `c${Date.now().toString(36)}`;
    const caseObj = {
      id: caseId,
      ts: new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit"}),
      name: file.name,
      thumb: payload.images.original,
      payload,
    };
    state.cases.unshift(caseObj);
    renderCaseList();
    activate(caseId);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    showLoader(false);
  }
}


function showLoader(on) { els.loader.classList.toggle("active", on); }
function cycleLoaderSteps() {
  const steps = [
    "Encoding patches…",
    "Running self-attention…",
    "Decoding segmentation…",
    "Computing rollout…",
    "Generating Grad-CAM…",
  ];
  let i = 0;
  const t = setInterval(() => {
    if (!els.loader.classList.contains("active")) { clearInterval(t); return; }
    els.loaderStep.textContent = steps[i % steps.length];
    i++;
  }, 600);
}


// ---------- Render ----------
function renderCaseList() {
  if (!state.cases.length) {
    els.caseList.innerHTML = `<div class="case-empty">No analyses yet — submit an image to begin.</div>`;
    return;
  }
  els.caseList.innerHTML = state.cases.map(c => {
    const cls = c.payload.prediction.class_name;
    const conf = (c.payload.prediction.confidence*100).toFixed(1);
    return `
      <div class="case-item ${c.id===state.active?'active':''}" data-id="${c.id}">
        <img class="case-thumb" src="${c.thumb}" alt="" />
        <div class="case-meta">
          <span class="case-class ${cls.toLowerCase()}">${cls}</span>
          <span class="case-time">${c.ts} · ${c.name.slice(0,18)}</span>
        </div>
        <div class="case-conf">${conf}%</div>
      </div>`;
  }).join("");
  els.caseList.querySelectorAll(".case-item").forEach(node => {
    node.addEventListener("click", () => activate(node.dataset.id));
  });
}

function activate(id) {
  const c = state.cases.find(x => x.id === id);
  if (!c) return;
  state.active = id;
  renderCaseList();
  renderResult(c.payload);
}

function renderResult(p) {
  // viewer
  els.viewerStage.classList.add("loaded");
  els.viewerStage.querySelector(".placeholder").style.display = "none";
  els.layerBase.src = p.images.original;
  els.layerBase.classList.add("visible");
  els.viewerTitle.textContent = `${p.prediction.class_name} · ${(p.prediction.confidence*100).toFixed(1)}%`;
  setLayer(state.currentLayer, p);

  // verdict
  const isMal = p.prediction.class_index === 1;
  els.verdictCard.classList.remove("benign","malignant");
  els.verdictCard.classList.add(isMal ? "malignant" : "benign");
  els.verdictClass.textContent = p.prediction.class_name.toUpperCase();
  els.verdictConf.textContent = `${(p.prediction.confidence*100).toFixed(1)}%`;
  els.verdictBand.textContent = p.prediction.confidence_band || "—";

  // bars
  const pb = p.prediction.probabilities.Benign * 100;
  const pm = p.prediction.probabilities.Malignant * 100;
  setTimeout(() => {
    els.probBenign.style.width = `${pb}%`;
    els.probMalignant.style.width = `${pm}%`;
  }, 50);
  els.probBenignVal.textContent = `${pb.toFixed(1)}%`;
  els.probMalignantVal.textContent = `${pm.toFixed(1)}%`;

  // ECG signature: derive a unique waveform from the prediction
  drawEcg(p);

  // metrics
  const m = p.lesion;
  els.mArea.textContent = m.area_px || "0";
  els.mAreaPct.textContent = m.area_pct !== undefined ? `(${m.area_pct}%)` : "";
  els.mCentroid.textContent = m.centroid ? `${m.centroid[0]} , ${m.centroid[1]}` : "—";
  els.mCompact.textContent = m.compactness != null ? m.compactness.toFixed(2) : "—";
  els.mIrreg.textContent = m.irregularity != null ? m.irregularity.toFixed(2) : "—";
  els.latency.textContent = `${p.elapsed_ms} ms`;
  els.reqid.textContent = `#${p.request_id}`;

  // explanation
  els.explainBody.innerHTML = formatExplanation(p.explanation);
}

function formatExplanation(text) {
  return (text || "")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
}


// ---------- Layer toggling ----------
els.layerBtns.querySelectorAll("button").forEach(btn => {
  btn.addEventListener("click", () => {
    state.currentLayer = btn.dataset.layer;
    els.layerBtns.querySelectorAll("button").forEach(b => b.classList.toggle("active", b===btn));
    const c = state.cases.find(x=>x.id===state.active);
    if (c) setLayer(state.currentLayer, c.payload);
  });
});

function setLayer(layer, payload) {
  const src = payload.images[layer];
  if (!src) { els.layerOverlay.classList.remove("visible"); return; }
  els.layerOverlay.src = src;
  els.layerOverlay.classList.add("visible");
}

els.opacity.addEventListener("input", () => {
  const v = els.opacity.value;
  els.opacityVal.textContent = `${v}%`;
  els.layerOverlay.style.opacity = v / 100;
});
els.opacity.dispatchEvent(new Event("input"));


// ---------- Crosshair ----------
els.viewerStage.addEventListener("mousemove", (e) => {
  if (!els.viewerStage.classList.contains("loaded")) return;
  const r = els.viewerStage.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  els.crosshair.style.setProperty("--x", `${x}px`);
  els.crosshair.style.setProperty("--y", `${y}px`);
  // map to 224x224 image space
  const ix = Math.round(x / r.width * 224);
  const iy = Math.round(y / r.height * 224);
  els.coord.textContent = `X ${ix.toString().padStart(3,"0")}  Y ${iy.toString().padStart(3,"0")}`;
});


// ---------- ECG signature ----------
// Build a waveform whose height encodes confidence and whose shape
// encodes the predicted class.
function drawEcg(p) {
  const W = 400, H = 60, pts = [];
  const conf = p.prediction.confidence;
  const isMal = p.prediction.class_index === 1;
  const irr = (p.lesion?.irregularity ?? 0);
  const compact = (p.lesion?.compactness ?? 0);

  let x = 0; pts.push([0, H/2]);
  while (x < W) {
    if (x > 60 && x < 120 && isMal) {
      // QRS-like spike
      pts.push([x+2, H/2 - 4]);
      pts.push([x+5, H/2 + 18]);
      pts.push([x+9, H/2 - 28 - conf*15]);  // tall peak when malignant + confident
      pts.push([x+13, H/2 + 14]);
      pts.push([x+18, H/2 - 4]);
      x += 20;
    } else if (x > 200 && x < 240 && !isMal) {
      // smooth bump for benign
      pts.push([x+5, H/2 - 8 - conf*10]);
      pts.push([x+10, H/2 - 12 - conf*12]);
      pts.push([x+16, H/2 - 6]);
      x += 18;
    } else {
      // baseline jitter scaled by irregularity
      const j = (Math.sin(x*0.27) + Math.sin(x*0.13)) * (1 + irr*5);
      pts.push([x+4, H/2 + j]);
      x += 4;
    }
  }
  pts.push([W, H/2]);
  const d = pts.map((p,i) => `${i?'L':'M'}${p[0]} ${p[1]}`).join(" ");
  // Force re-animation by clearing then re-setting
  els.ecgPath.setAttribute("d", d);
  els.ecgPath.style.animation = "none";
  void els.ecgPath.offsetWidth;
  els.ecgPath.style.animation = "draw 2s ease forwards";
}
