/* FaceWorkbench Pro — Frontend */
(function () {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  let state = {
    config: {},
    lastAnalysis: null,
    lastMatches: null,
    studioImg: null,
    layers: { bbox: true, kps5: true, mesh106: false, mesh3d: false },
    mode: "search",
  };

  // ---- API helpers ----
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  }

  async function postFile(path, file, extra = {}) {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(extra)) fd.append(k, v);
    return api(path, { method: "POST", body: fd });
  }

  // ---- Navigation ----
  $$(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".nav-item").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".workspace").forEach((w) => w.classList.remove("active"));
      const id = "ws-" + btn.dataset.ws;
      $(`#${id}`)?.classList.add("active");
      if (btn.dataset.ws === "gallery") loadPeople();
      if (btn.dataset.ws === "system") loadModels();
      if (btn.dataset.ws === "osint") loadEngines();
    });
  });

  // ---- Health / config ----
  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      $("#statusDot").style.background = "var(--status-emerald)";
      $("#statusText").textContent = "Online";
      $("#lanIp").textContent = h.lan_ip ? `LAN ${h.lan_ip}` : "";
      $("#thrLabel").textContent = `θ=${Number(h.threshold).toFixed(2)}`;
      state.config = await api("/api/config");
      syncSettingsUI();
    } catch {
      $("#statusDot").style.background = "var(--status-rose)";
      $("#statusText").textContent = "Offline";
    }
  }

  function syncSettingsUI() {
    const c = state.config;
    if (!c) return;
    $("#cfgThr").value = c.match_threshold ?? 0.45;
    $("#thrVal").textContent = Number(c.match_threshold ?? 0.45).toFixed(2);
    $("#cfgDetSize").value = c.detection_size ?? 640;
    $("#cfgDetector").value = c.primary_detector ?? "yunet";
    $("#cfgThreads").value = c.cpu_threads ?? 4;
    $("#thrThreads").textContent = c.cpu_threads ?? 4;
  }

  $("#cfgThr")?.addEventListener("input", (e) => {
    $("#thrVal").textContent = Number(e.target.value).toFixed(2);
  });
  $("#cfgThreads")?.addEventListener("input", (e) => {
    $("#thrThreads").textContent = e.target.value;
  });

  $("#btnSaveCfg")?.addEventListener("click", async () => {
    const body = {
      match_threshold: parseFloat($("#cfgThr").value),
      detection_size: parseInt($("#cfgDetSize").value, 10),
      primary_detector: $("#cfgDetector").value,
      cpu_threads: parseInt($("#cfgThreads").value, 10),
    };
    state.config = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    refreshHealth();
    alert("Settings saved");
  });

  // ---- Dropzone helper ----
  function bindDropzone(zone, input, onFiles) {
    zone?.addEventListener("click", () => input?.click());
    zone?.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone?.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone?.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
    });
    input?.addEventListener("change", () => {
      if (input.files?.length) onFiles(input.files);
    });
  }

  // ---- Studio ----
  bindDropzone($("#studioDrop"), $("#studioFile"), async (files) => {
    const file = files[0];
    if (state.mode === "search") await runSearch(file);
    else await runAnalyzeOnly(file);
  });

  $("#btnStudioMode")?.addEventListener("click", () => {
    state.mode = "search";
    $("#matchPanel").classList.remove("hidden");
    $("#verifyPanel").classList.add("hidden");
  });
  $("#btnVerifyMode")?.addEventListener("click", () => {
    state.mode = "verify";
    $("#matchPanel").classList.add("hidden");
    $("#verifyPanel").classList.remove("hidden");
  });

  async function runAnalyzeOnly(file) {
    const data = await postFile("/api/analyze", file);
    state.lastAnalysis = data;
    drawStudio(file, data.faces?.[0]);
    renderQuality(data.faces?.[0]);
  }

  async function runSearch(file) {
    const data = await postFile("/api/search", file);
    state.lastMatches = data;
    state.lastAnalysis = { faces: data.probe ? [data.probe] : [] };
    drawStudio(file, data.probe);
    renderQuality(data.probe);
    renderMatches(data.matches || [], data.threshold);
  }

  function drawStudio(file, face) {
    const canvas = $("#studioCanvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      state.studioImg = img;
      const maxW = 640, maxH = 480;
      let w = img.width, h = img.height;
      const scale = Math.min(maxW / w, maxH / h, 1);
      w = Math.round(w * scale);
      h = Math.round(h * scale);
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(img, 0, 0, w, h);
      if (face?.bbox) {
        const [x, y, bw, bh] = face.bbox;
        const sx = w / img.width, sy = h / img.height;
        if (state.layers.bbox) {
          ctx.strokeStyle = "#06B6D4";
          ctx.lineWidth = 2;
          ctx.strokeRect(x * sx, y * sy, bw * sx, bh * sy);
        }
        if (state.layers.kps5 && face.landmarks5) {
          ctx.fillStyle = "#10B981";
          face.landmarks5.forEach((p) => {
            ctx.beginPath();
            ctx.arc(p[0] * sx, p[1] * sy, 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }
    };
    img.src = URL.createObjectURL(file);
  }

  $$("#studioToggles .toggle-pill").forEach((p) => {
    p.addEventListener("click", () => {
      p.classList.toggle("active");
      state.layers[p.dataset.layer] = p.classList.contains("active");
    });
  });

  function renderQuality(probe) {
    const q = probe?.quality || {};
    const attrs = probe?.attributes || {};
    const pct = Math.round((q.quality || 0) * 100);
    $("#qualityRadials").innerHTML = `
      <div class="radial" style="--pct:${pct}"><span>${pct}%</span></div>
      <div>
        <div class="muted" style="font-size:11px">Composite Quality</div>
        <div class="mono">sharp ${Number(q.sharpness||0).toFixed(0)} · bright ${Number(q.brightness||0).toFixed(0)}</div>
      </div>`;
    const chips = [];
    chips.push(chip("Size", q.size_ok));
    chips.push(chip("Sharp", q.sharp_ok));
    chips.push(chip("Exposure", q.bright_ok));
    chips.push(chip("Yaw", q.yaw_ok));
    $("#qualityChips").innerHTML = chips.join("");
    const ac = [];
    if (attrs.age != null) ac.push(`<span class="chip">Age ${Math.round(attrs.age)}</span>`);
    if (attrs.gender) ac.push(`<span class="chip">${attrs.gender}</span>`);
    if (attrs.emotion) ac.push(`<span class="chip">${attrs.emotion}</span>`);
    if (attrs.liveness != null) ac.push(`<span class="chip ${attrs.is_live ? "ok" : "bad"}">Live ${(attrs.liveness*100).toFixed(0)}%</span>`);
    $("#attrChips").innerHTML = ac.join("");

    // behaviour tab mirrors
    if (q.yaw != null) {
      $("#poseYaw").textContent = `${Number(q.yaw).toFixed(1)}°`;
      $("#poseRoll").textContent = `${Number(q.roll||0).toFixed(1)}°`;
      $("#posePitch").textContent = `${Number(q.pitch||0).toFixed(1)}°`;
    }
  }

  function chip(label, ok) {
    return `<span class="chip ${ok ? "ok" : "warn"}">${label}</span>`;
  }

  function renderMatches(matches, thr) {
    if (!matches.length) {
      $("#matchResults").innerHTML = `<p class="muted">No matches above threshold.</p>`;
      return;
    }
    $("#matchResults").innerHTML = matches
      .map((m) => {
        const bars = Object.entries(m.per_model || {})
          .map(([k, v]) => {
            const pct = Math.max(0, Math.min(100, v * 100));
            const thrPct = (thr || 0.45) * 100;
            return `<div class="score-row"><span class="name">${k}</span>
              <div class="score-bar"><div class="fill" style="width:${pct}%"></div>
              <div class="thr" style="left:${thrPct}%"></div></div>
              <span class="val">${v.toFixed(3)}</span></div>`;
          })
          .join("");
        return `<div class="mb" style="padding:10px 0;border-bottom:1px solid var(--border-subtle)">
          <div class="row gap"><strong>${m.person_name}</strong>
            <span class="chip ${m.above_threshold ? "ok" : "warn"}">${m.fused_score.toFixed(3)}</span>
            <span class="muted mono">P=${(m.stacker_prob||0).toFixed(2)}</span>
          </div>${bars}</div>`;
      })
      .join("");
  }

  // Verify
  let verifyFiles = { a: null, b: null };
  bindDropzone($("#verifyDropA"), $("#verifyFileA"), (f) => {
    verifyFiles.a = f[0];
    const p = $("#verifyDropA")?.querySelector("p");
    if (p) p.textContent = f[0].name;
  });
  bindDropzone($("#verifyDropB"), $("#verifyFileB"), (f) => {
    verifyFiles.b = f[0];
    const p = $("#verifyDropB")?.querySelector("p");
    if (p) p.textContent = f[0].name;
  });
  $("#btnRunVerify")?.addEventListener("click", async () => {
    if (!verifyFiles.a || !verifyFiles.b) return alert("Select both images");
    const fd = new FormData();
    fd.append("file_a", verifyFiles.a);
    fd.append("file_b", verifyFiles.b);
    const data = await api("/api/verify", { method: "POST", body: fd });
    const el = $("#verifyResults");
    if (!data.ok) { el.innerHTML = `<p class="text-rose">${data.message}</p>`; return; }
    el.innerHTML = `<div class="row gap">
      <span class="chip ${data.match ? "ok" : "bad"}">${data.match ? "MATCH" : "NO MATCH"}</span>
      <span class="mono">fused ${data.fused_score.toFixed(4)}</span>
      <span class="mono">P=${data.stacker_prob.toFixed(3)}</span>
      <span class="muted">θ=${data.threshold}</span>
    </div>
    ${Object.entries(data.per_model||{}).map(([k,v]) =>
      `<div class="score-row"><span class="name">${k}</span>
       <div class="score-bar"><div class="fill" style="width:${v.cosine*100}%"></div></div>
       <span class="val">${v.cosine.toFixed(3)}</span></div>`).join("")}`;
  });

  // ---- Gallery ----
  async function loadPeople(q = "") {
    const people = await api("/api/people?q=" + encodeURIComponent(q));
    const grid = $("#peopleGrid");
    if (!people.length) {
      grid.innerHTML = `<p class="muted">No identities yet. Enroll someone to get started.</p>`;
      return;
    }
    grid.innerHTML = people
      .map(
        (p) => `<div class="person-card" data-id="${p.id}">
        <div class="avatar" style="background:linear-gradient(135deg,#1e293b,#0f172a);display:flex;align-items:center;justify-content:center;font-size:28px;color:#64748b">${(p.name||"?")[0]}</div>
        <div class="name">${p.name}</div>
        <div class="meta">${p.sample_count||0} samples · Q ${Number(p.avg_quality||0).toFixed(2)}</div>
      </div>`
      )
      .join("");
    $$(".person-card", grid).forEach((c) =>
      c.addEventListener("click", () => openPerson(c.dataset.id))
    );
  }

  $("#peopleSearch")?.addEventListener("input", (e) => loadPeople(e.target.value));

  async function openPerson(id) {
    const data = await api(`/api/people/${id}`);
    $("#personInspector").classList.remove("hidden");
    $("#inspName").textContent = data.person.name;
    $("#inspName").dataset.id = id;
    $("#inspFaces").innerHTML = (data.faces || [])
      .map((f) => `<div class="person-card"><div class="meta mono">${f.id.slice(0,8)}…</div>
        <div class="meta">Q ${Number(f.quality||0).toFixed(2)}</div></div>`)
      .join("") || `<p class="muted">No faces</p>`;
  }

  $("#inspClose")?.addEventListener("click", () => $("#personInspector").classList.add("hidden"));
  $("#inspDelete")?.addEventListener("click", async () => {
    const id = $("#inspName").dataset.id;
    if (!id || !confirm("Delete this identity?")) return;
    await api(`/api/people/${id}`, { method: "DELETE" });
    $("#personInspector").classList.add("hidden");
    loadPeople();
  });
  $("#inspRename")?.addEventListener("click", async () => {
    const id = $("#inspName").dataset.id;
    const name = prompt("New name", $("#inspName").textContent);
    if (!name) return;
    const fd = new FormData();
    fd.append("name", name);
    await api(`/api/people/${id}`, { method: "PATCH", body: fd });
    loadPeople();
    openPerson(id);
  });

  // Enroll modal
  $("#btnEnroll")?.addEventListener("click", () => $("#enrollModal").classList.remove("hidden"));
  $("#enrollCancel")?.addEventListener("click", () => $("#enrollModal").classList.add("hidden"));
  let enrollFileList = [];
  bindDropzone($("#enrollDrop"), $("#enrollFiles"), (files) => {
    enrollFileList = [...files];
    $("#enrollPreview").textContent = `${enrollFileList.length} file(s) selected`;
  });
  $("#enrollSubmit")?.addEventListener("click", async () => {
    const name = $("#enrollName").value.trim();
    if (!name || !enrollFileList.length) return alert("Name and at least one photo required");
    const fd = new FormData();
    fd.append("name", name);
    enrollFileList.forEach((f) => fd.append("files", f));
    const res = await api("/api/enroll", { method: "POST", body: fd });
    $("#enrollModal").classList.add("hidden");
    $("#enrollName").value = "";
    enrollFileList = [];
    alert(`Enrolled ${res.count} face(s)`);
    loadPeople();
  });

  $("#btnExportZip, #btnExportZip2")?.addEventListener?.("click", () => {
    window.location.href = "/api/export/gallery";
  });
  // both buttons
  ["btnExportZip", "btnExportZip2"].forEach((id) => {
    $(`#${id}`)?.addEventListener("click", () => { window.location.href = "/api/export/gallery"; });
  });

  $("#btnRebuildFaiss")?.addEventListener("click", async () => {
    const r = await api("/api/indexes/rebuild", { method: "POST" });
    alert(`Rebuilt ${r.rebuilt} embeddings`);
  });

  // ---- Calibration ----
  $("#btnCalibrate")?.addEventListener("click", async () => {
    const far = parseFloat($("#targetFar").value);
    const r = await api(`/api/calibrate?target_far=${far}`, { method: "POST" });
    $("#calResult").textContent = r.ok
      ? `θ=${r.threshold.toFixed(4)}  TAR@FAR=${JSON.stringify(r.tar_at_far)}`
      : r.message;
    if (r.roc) drawROC(r.roc);
    refreshHealth();
  });

  $("#btnTrainStacker")?.addEventListener("click", async () => {
    const r = await api("/api/stacker/train", { method: "POST" });
    $("#stackerResult").textContent = r.ok
      ? `acc=${r.accuracy.toFixed(3)} n=${r.n_samples} keys=${r.model_keys}`
      : r.message;
  });

  $("#btnHardNeg")?.addEventListener("click", async () => {
    const rows = await api("/api/hard-negatives");
    if (!rows.length) {
      $("#hardNegTable").innerHTML = `<p class="muted">No hard negatives found.</p>`;
      return;
    }
    $("#hardNegTable").innerHTML = `<table class="data"><thead><tr><th>Person A</th><th>Person B</th><th>Sim</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td class="mono">${r.person_a.slice(0,8)}</td><td class="mono">${r.person_b.slice(0,8)}</td><td>${r.similarity.toFixed(3)}</td></tr>`).join("")}</tbody></table>`;
  });

  function drawROC(roc) {
    const c = $("#rocCanvas");
    if (!c || !roc) return;
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.fillStyle = "#070B14";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(148,163,184,0.2)";
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * W, y = (i / 10) * H;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.strokeStyle = "#06B6D4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    (roc.fpr || []).forEach((f, i) => {
      const x = f * W;
      const y = H - (roc.tpr[i] || 0) * H;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // ---- Video ----
  bindDropzone($("#videoDrop"), $("#videoFile"), () => {});
  $("#btnRedact")?.addEventListener("click", async () => {
    const f = $("#videoFile").files?.[0];
    if (!f) return alert("Select a video");
    $("#videoProgress").textContent = "Processing… this may take a while";
    const fd = new FormData();
    fd.append("file", f);
    fd.append("mode", $("#redactMode").value);
    fd.append("target_ids", $("#targetIds").value);
    try {
      const res = await fetch("/api/video/redact", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "redacted.mp4";
      a.click();
      $("#videoProgress").textContent = "Done — download started";
    } catch (e) {
      $("#videoProgress").textContent = "Error: " + e.message;
    }
  });

  // ---- OSINT ----
  async function loadEngines() {
    const engines = await api("/api/osint/engines");
    $("#engineCards").innerHTML = engines
      .map(
        (e) => `<div class="engine-card">
        <h4>${e.name}</h4><p>${e.description}</p>
        <button class="btn sm" data-url="${e.url}">Open</button></div>`
      )
      .join("");
    $$("#engineCards button").forEach((b) =>
      b.addEventListener("click", () => window.open(b.dataset.url, "_blank"))
    );
  }

  bindDropzone($("#exifDrop"), $("#exifFile"), async (files) => {
    const data = await postFile("/api/osint/exif", files[0]);
    $("#exifOut").textContent = JSON.stringify(data, null, 2);
  });

  $("#btnOpenUrls")?.addEventListener("click", async () => {
    const fd = new FormData();
    fd.append("urls", $("#urlBox").value);
    const r = await api("/api/osint/open", { method: "POST", body: fd });
    alert(`Opened ${r.opened} URL(s)`);
  });

  // ---- Models ----
  async function loadModels() {
    const rows = await api("/api/models");
    const tb = $("#modelTable tbody");
    tb.innerHTML = rows
      .map(
        (r) => `<tr>
        <td class="mono">${r.key}</td>
        <td>${r.family}</td>
        <td>${r.dim || "—"}</td>
        <td><span class="chip ${r.available ? "ok" : "warn"}">${r.status}</span></td>
        <td>${r.available ? "—" : r.has_urls ? `<button class="btn sm" data-dl="${r.key}">Fetch</button>` : "Offline"}</td>
      </tr>`
      )
      .join("");
    $$("[data-dl]", tb).forEach((b) =>
      b.addEventListener("click", async () => {
        b.disabled = true;
        b.textContent = "…";
        await api(`/api/models/${b.dataset.dl}/download`, { method: "POST" });
        loadModels();
      })
    );
  }

  // ---- Behaviour AU meters (demo) ----
  const AU_LABELS = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"];
  function renderAU(vals) {
    $("#auMeters").innerHTML = AU_LABELS.map((lab, i) => {
      const v = vals?.[i] ?? Math.random() * 2;
      const pct = Math.min(100, (v / 5) * 100);
      return `<div class="au-bar"><span class="mono">${lab}</span>
        <div class="fill-wrap"><div class="fill" style="width:${pct}%"></div></div>
        <span class="mono">${v.toFixed(1)}</span></div>`;
    }).join("");
  }
  renderAU();

  // Gaze compass
  function drawGaze(gx = 0, gy = 0) {
    const c = $("#gazeCanvas");
    if (!c) return;
    const ctx = c.getContext("2d");
    const cx = c.width / 2, cy = c.height / 2, r = 90;
    ctx.fillStyle = "#070B14";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.strokeStyle = "rgba(148,163,184,0.3)";
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();
    const px = cx + gx * r * 0.8;
    const py = cy + gy * r * 0.8;
    ctx.fillStyle = "#06B6D4";
    ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill();
    $("#gazeX").textContent = `gx: ${gx.toFixed(2)}`;
    $("#gazeY").textContent = `gy: ${gy.toFixed(2)}`;
  }
  drawGaze(0, 0);

  bindDropzone($("#behDrop"), $("#behFile"), async (files) => {
    const data = await postFile("/api/analyze", files[0]);
    const f = data.faces?.[0];
    if (f?.quality) {
      $("#poseYaw").textContent = `${Number(f.quality.yaw||0).toFixed(1)}°`;
      $("#poseRoll").textContent = `${Number(f.quality.roll||0).toFixed(1)}°`;
      $("#posePitch").textContent = `${Number(f.quality.pitch||0).toFixed(1)}°`;
      drawGaze((f.quality.yaw || 0) / 45, (f.quality.pitch || 0) / 45);
    }
    renderAU();
  });

  // Lookalikes
  $("#topK")?.addEventListener("input", (e) => { $("#topKVal").textContent = e.target.value; });
  bindDropzone($("#lookDrop"), $("#lookFile"), async (files) => {
    const k = $("#topK").value;
    const data = await postFile(`/api/search?top_k=${k}`, files[0]);
    // force show even below threshold
    renderLook(data.matches || []);
  });
  function renderLook(matches) {
    $("#lookResults").innerHTML = matches
      .map((m) => `<div class="chip ${m.above_threshold?"ok":""}">${m.person_name} · ${m.fused_score.toFixed(3)}</div>`)
      .join(" ") || `<p class="muted">No results</p>`;
  }

  // Export JSON
  $("#btnExportJson")?.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state.lastAnalysis || state.lastMatches || {}, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "faceworkbench_result.json";
    a.click();
  });

  // Webcam (simple capture)
  $("#btnWebcam")?.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      await video.play();
      const c = document.createElement("canvas");
      c.width = video.videoWidth;
      c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      stream.getTracks().forEach((t) => t.stop());
      c.toBlob(async (blob) => {
        const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
        await runSearch(file);
      }, "image/jpeg");
    } catch (e) {
      alert("Camera access failed: " + e.message);
    }
  });

  // boot
  refreshHealth();
  setInterval(refreshHealth, 30000);
})();
