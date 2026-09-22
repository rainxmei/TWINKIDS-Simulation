/* =========================================================
   ANTARAKALA — Physical Device Simulation
   Satu halaman LCD (sesuai referensi), 3 tombol (kiri/pilih/kanan).
   Device adalah "sumber kebenaran": saat merekam titik di device,
   perangkat memutar sampel suara paru ICBHI nyata yang sudah diproses
   oleh rekonstruksi CNN, lalu mengirim hasil + RR + Grad-CAM ke app.js.
   ========================================================= */
(function(){
  "use strict";

  const POINT_NAMES = [
    "Posterior Atas Kiri","Posterior Atas Kanan","Posterior Bawah Kiri",
    "Posterior Bawah Kanan","Anterior Atas Kiri","Anterior Atas Kanan"
  ];
  const REC_SECONDS = 9;         // durasi rekam per titik
  const BAD_SIGNAL_CHANCE = 0.16; // peluang kualitas sinyal rendah per rekaman
  const RESULT_COLORS = { crackle:"#D9364A", wheeze:"#C98A00", normal:"#00A3AE" };
  const RESULT_LABELS = { crackle:"CRACKLE", wheeze:"WHEEZE", normal:"NORMAL" };

  const D = {
    state: "idle",   // idle | recording | badsignal | complete | allDone
    cursor: 0,
    done: [false,false,false,false,false,false],
    results: [null,null,null,null,null,null], // {result, confidence}
    recElapsed: 0,
    recTimer: null,
    phoneReady: false, // true only when phone/HP is on the "proses-auskultasi" screen
    probeDockedPoint: null, // indeks titik yang benar-benar ditempeli kepala stetoskop
    activeSample: null,
    usedSampleIds: [],
    audioEl: null,
    maxState: "idle", // idle | prompt | measuring | done
    maxTimer: null,
    spo2: null,
    hr: null,
  };

  const $ = (sel) => document.querySelector(sel);
  const lcd = () => $("#deviceLcd");

  function emit(name, detail){
    document.dispatchEvent(new CustomEvent(name, { detail: detail || {} }));
  }

  function sampleCatalog(){
    return Array.isArray(window.ANTARAKALA_LUNG_SAMPLES) ? window.ANTARAKALA_LUNG_SAMPLES : [];
  }

  function pickRecordingSample(){
    const catalog = sampleCatalog();
    if(!catalog.length) return null;
    let available = catalog.filter(x=>!D.usedSampleIds.includes(x.id));
    if(!available.length){ D.usedSampleIds = []; available = catalog.slice(); }
    const sample = available[Math.floor(Math.random()*available.length)];
    D.usedSampleIds.push(sample.id);
    return sample;
  }

  function stopSampleAudio(){
    if(!D.audioEl) return;
    try{ D.audioEl.pause(); D.audioEl.currentTime = 0; }catch(e){}
    D.audioEl = null;
  }

  function playSampleAudio(sample){
    stopSampleAudio();
    if(!sample || !sample.audio) return;
    try{
      const audio = new Audio(sample.audio);
      audio.loop = true;
      audio.volume = 0.9;
      D.audioEl = audio;
      const promise = audio.play();
      if(promise && typeof promise.catch === "function") promise.catch(()=>{});
    }catch(e){}
  }

  /* ---------- body diagram SVG ---------- */
  function bodySVG(){
    const positions = [
      {n:1, x:44, y:40}, {n:2, x:70, y:40},
      {n:3, x:42, y:62}, {n:4, x:72, y:62},
      {n:5, x:178, y:46}, {n:6, x:152, y:46},
    ];
    const circles = positions.map((p,i)=>{
      const isActive = i === D.cursor;
      const res = D.results[i];
      let fill = "#fff", stroke = "#9AA39D", strokeW = 1.4, textFill = "#3A423F", glow = "";

      if(res){ fill = "#00A3AE"; stroke = fill; textFill = "#fff"; }
      if(isActive && D.state === "idle"){ stroke = "#1C7FD6"; strokeW = 2.4; if(!res){ textFill = "#1C7FD6"; } }
      if(isActive && (D.state === "recording")){
        fill = "#E8720C"; stroke = "#E8720C"; textFill = "#fff";
        glow = `<circle cx="${p.x}" cy="${p.y}" r="12" fill="#E8720C" opacity="0.35"><animate attributeName="r" values="9;14;9" dur="1.1s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.45;0.05;0.45" dur="1.1s" repeatCount="indefinite"/></circle>`;
      }
      if(isActive && D.state === "badsignal"){
        fill = "#fff"; stroke = "#C98A00"; strokeW = 2.4; textFill = "#C98A00";
        glow = `<circle cx="${p.x}" cy="${p.y}" r="12" fill="none" stroke="#C98A00" stroke-width="1.6" opacity="0.7"><animate attributeName="r" values="9;15;9" dur=".8s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.8;0.1;0.8" dur=".8s" repeatCount="indefinite"/></circle>`;
      }
      return `${glow}<circle cx="${p.x}" cy="${p.y}" r="7.5" fill="${fill}" stroke="${stroke}" stroke-width="${strokeW}"/><text x="${p.x}" y="${p.y+3}" font-size="8" font-weight="800" text-anchor="middle" fill="${textFill}" font-family="Poppins, sans-serif">${p.n}</text>`;
    }).join("");

    return `<svg class="dlcd-diagram" viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">
      <circle cx="55" cy="16" r="11" fill="none" stroke="#9AA39D" stroke-width="1.4"/>
      <path d="M28 38 Q55 22 82 38 L79 76 Q55 84 31 76 Z" fill="none" stroke="#9AA39D" stroke-width="1.4"/>
      <circle cx="165" cy="16" r="11" fill="none" stroke="#9AA39D" stroke-width="1.4"/>
      <path d="M138 40 Q165 26 192 40 L189 74 Q165 80 141 74 Z" fill="none" stroke="#9AA39D" stroke-width="1.4"/>
      ${circles}
    </svg>`;
  }


  function median(values){
    const arr = values.filter(Number.isFinite).slice().sort((a,b)=>a-b);
    if(!arr.length) return NaN;
    const m = Math.floor(arr.length/2);
    return arr.length % 2 ? arr[m] : (arr[m-1] + arr[m]) / 2;
  }

  function currentRR(){
    const rrValues = D.results.map(r => r ? Number(r.rr) : NaN).filter(Number.isFinite);
    const rr = median(rrValues);
    return Number.isFinite(rr) ? Math.round(rr) : null;
  }

  function battWifi(){
    return `<div class="dlcd-status">
      <div class="dlcd-batt"><i></i><i></i><i></i><span class="cap"></span></div>
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#3DC4C0" stroke-width="2.6" stroke-linecap="round"><path d="M9 17H7a5 5 0 010-10h2"/><path d="M15 7h2a5 5 0 010 10h-2"/><path d="M8 12h8"/></svg>
    </div>`;
  }

  function updateLED(){
    const led = $("#deviceLed");
    if(!led) return;
    const isRecording = D.state === "recording";
    led.classList.toggle("led-green", isRecording);
    led.classList.toggle("led-red", !isRecording);
  }

  /* ---------- single-page render ---------- */
  function render(){
    const el = lcd();
    if(!el) return;
    updateLED();
    syncPointVisuals();

    if(!D.phoneReady && D.state === "idle"){
      el.innerHTML = `
        <div class="dlcd-header"><b>Twinkids</b>${battWifi()}</div>
        <div class="dlcd-gate">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/></svg>
          <b>Menunggu Perangkat</b>
          <span>Isi data pasien, suhu, dan kondisi klinis terlebih dahulu. Alat aktif saat halaman auskultasi dimulai</span>
        </div>`;
      return;
    }

    if(D.state === "allDone"){
      if(D.maxState === "prompt"){
        el.innerHTML = `
          <div class="dlcd-header"><b>Twinkids</b>${battWifi()}</div>
          <div class="dlcd-max-wrap">
            <div class="dlcd-max-icon">☝</div>
            <b>Tempelkan Jari</b>
            <span>Tempelkan jari telunjuk pada oksimeter.</span>
            <small>${currentRR() !== null ? `RR terdeteksi: ${currentRR()} x/menit` : ""}</small>
            <em>Tekan PILIH setelah jari terpasang.</em>
          </div>`;
        return;
      }
      if(D.maxState === "measuring"){
        el.innerHTML = `
          <div class="dlcd-header"><b>Twinkids</b>${battWifi()}</div>
          <div class="dlcd-max-wrap">
            <div class="dlcd-max-pulse">♥</div>
            <b>MENGUKUR...</b>
            <span>Pertahankan jari tetap diam pada sensor.</span>
            <small>${currentRR() !== null ? `RR: ${currentRR()} x/menit` : ""}</small>
            <div class="dlcd-max-progress"><i></i></div>
          </div>`;
        return;
      }
      if(D.maxState === "done"){
        el.innerHTML = `
          <div class="dlcd-header"><b>Twinkids</b>${battWifi()}</div>
          <div class="dlcd-max-result">
            <div><span>SpO₂</span><b>${D.spo2}%</b></div>
            <div><span>HR</span><b>${D.hr} bpm</b></div>
            <div><span>RR</span><b>${currentRR() !== null ? currentRR() + " x/menit" : "—"}</b></div>
            <p>✓ Pengukuran selesai</p>
            <small>Lanjutkan analisis di Web Lokal</small>
          </div>`;
        return;
      }
    }

    const allDone = D.state === "allDone";
    const headerLabel = allDone ? "Selesai 6/6" : `Titik ${D.cursor+1}/6`;

    // progress bar
    let pct = 0, timeLabel = `${REC_SECONDS} detik`;
    if(D.state === "recording"){
      pct = Math.min(100, (D.recElapsed/REC_SECONDS)*100);
      timeLabel = `${Math.max(0, REC_SECONDS - Math.floor(D.recElapsed))} detik`;
    } else if(D.state === "complete" || D.state === "badsignal" || allDone){
      pct = 100; timeLabel = "0 detik";
    }

    // bottom-left status block
    let resultHTML;
    const res = D.results[D.cursor];
    if(allDone){
      resultHTML = `<b style="color:#00A3AE; font-size:11px;">✓ SELESAI</b><span>Lanjut pengukuran sensor</span>`;
    } else if(D.state === "recording"){
      resultHTML = `<b style="color:#3A423F;">MEREKAM…</b><span>Jangan gerakkan stetoskop</span>`;
    } else if(D.state === "badsignal"){
      resultHTML = `<b style="color:#C98A00;">SINYAL RENDAH</b><span>Mengulang otomatis…</span>`;
    } else if(res){
      resultHTML = `<b style="color:#00A3AE;">✓ TITIK SELESAI</b><span>Hasil disimpan untuk analisis akhir</span>`;
    } else {
      const probeReady = D.probeDockedPoint === D.cursor;
      resultHTML = probeReady
        ? `<b style="color:#3A423F; font-size:8px;">SIAP MEREKAM</b><span>Tekan PILIH untuk mulai</span>`
        : `<b style="color:#3A423F; font-size:8px;">SIAP MEREKAM</b><span>Tempelkan stetoskop ke titik yang dipilih</span>`;
    }

    const pointLabel = allDone
      ? `<span>6/6 TITIK</span><span>TEREKAM</span>`
      : (()=>{
          const parts = POINT_NAMES[D.cursor].toUpperCase().split(" ");
          return `<span>${parts[0]}</span><span>${parts.slice(1).join(" ")}</span>`;
        })();

    el.innerHTML = `
      <div class="dlcd-header"><b>${headerLabel}</b>${battWifi()}</div>
      <div class="dlcd-bar-row" style="padding:0 6%;">
        <div class="dlcd-bar"><div class="dlcd-bar-fill" style="width:${pct}%"></div></div>
        <b>${timeLabel}</b>
      </div>
      <div class="dlcd-body">
        ${bodySVG()}
        <div class="dlcd-labels"><span>Punggung</span><span>Dada</span></div>
      </div>
      <div class="dlcd-resultrow">
        <div class="dlcd-result">${resultHTML}</div>
        <div class="dlcd-pointname">${pointLabel}</div>
      </div>`;
  }

  /* ---------- transitions ---------- */
  function flashDenied(){
    const el = $(".device");
    if(!el) return;
    el.classList.remove("denied");
    void el.offsetWidth; // restart animation
    el.classList.add("denied");
  }

  function pressKiri(){
    if(!D.phoneReady){ flashDenied(); return; }
    if(D.state !== "idle") return;
    D.cursor = (D.cursor + 5) % 6;
    D.probeDockedPoint = null;
    autoSwitchViewForCursor();
    render();
    requestAnimationFrame(()=>resetProbePosition(true));
  }
  function pressKanan(){
    if(!D.phoneReady){ flashDenied(); return; }
    if(D.state !== "idle") return;
    D.cursor = (D.cursor + 1) % 6;
    D.probeDockedPoint = null;
    autoSwitchViewForCursor();
    render();
    requestAnimationFrame(()=>resetProbePosition(true));
  }
  function pressPilih(){
    if(!D.phoneReady){ flashDenied(); return; }
    if(D.state === "idle"){
      if(D.probeDockedPoint !== D.cursor){
        flashDenied();
        return;
      }
      startRecording();
      return;
    }
    if(D.state === "allDone"){
      if(D.maxState === "prompt") startMaxMeasurement();
      else render();
      return;
    }
    // recording / badsignal: tombol tidak berfungsi, otomatis berjalan
  }

  function startRecording(){
    if(!D.activeSample) D.activeSample = pickRecordingSample();
    D.state = "recording";
    D.recElapsed = 0;
    playSampleAudio(D.activeSample);
    render();
    emit("antarakala:point-start", { index: D.cursor, name: POINT_NAMES[D.cursor], duration: REC_SECONDS, sampleId:D.activeSample && D.activeSample.id });

    clearInterval(D.recTimer);
    D.recTimer = setInterval(()=>{
      D.recElapsed += 0.1;
      if(D.recElapsed >= REC_SECONDS){
        clearInterval(D.recTimer);
        finishRecording();
        return;
      }
      render();
    }, 100);
  }

  function finishRecording(){
    stopSampleAudio();
    const badSignal = Math.random() < BAD_SIGNAL_CHANCE;
    if(badSignal){
      D.state = "badsignal";
      render();
      emit("antarakala:signal-warning", { index: D.cursor });
      setTimeout(()=>{ startRecording(); }, 1500);
      return;
    }

    const sample = D.activeSample || pickRecordingSample();
    const result = sample ? sample.label : "normal";
    const confidence = sample && Number.isFinite(Number(sample.confidence)) ? Math.round(Number(sample.confidence)) : 50;
    const sampleMeta = sample ? {
      sampleId:sample.id, rr:Number(sample.rr), gradcam:sample.gradcam, audio:sample.audio,
      sourceRecording:sample.source_recording, annotationCycle:sample.annotation_cycle,
      probabilities:sample.probabilities || null
    } : {};
    D.results[D.cursor] = { result, confidence, ...sampleMeta };
    D.done[D.cursor] = true;
    D.state = "complete";
    render();
    emit("antarakala:point-result", { index: D.cursor, name: POINT_NAMES[D.cursor], result, confidence, ...sampleMeta });
    D.activeSample = null;

    setTimeout(()=>{
      const next = D.done.findIndex(v=>!v);
      D.probeDockedPoint = null;
      if(next === -1){
        D.state = "allDone";
        D.maxState = "prompt";
        render();
        requestAnimationFrame(()=>resetProbePosition(true));
        emit("antarakala:all-done", {});
      } else {
        D.cursor = next;
        D.state = "idle";
        autoSwitchViewForCursor();
        render();
        requestAnimationFrame(()=>resetProbePosition(true));
      }
    }, 1100);
  }

  function startMaxMeasurement(){
    if(D.maxState !== "prompt") return;
    D.maxState = "measuring";
    render();
    emit("antarakala:max-start", {});
    clearTimeout(D.maxTimer);
    D.maxTimer = setTimeout(finishMaxMeasurement, 3200);
  }

  function finishMaxMeasurement(){
    const r = Math.random();
    D.spo2 = r < 0.10 ? Math.floor(88 + Math.random()*2) : r < 0.35 ? Math.floor(90 + Math.random()*5) : Math.floor(95 + Math.random()*5);
    D.hr = Math.floor(92 + Math.random()*40);
    D.maxState = "done";
    render();
    emit("antarakala:max-result", { spo2:D.spo2, hr:D.hr });
  }

  /* ---------- posisi stetoskop (docking ke port alat) + gambar kabel + drag-drop ---------- */
  function scaleFactor(){ return window.__antarakalaScale || 1; }
  function stageEl(){ return $(".device-diagram-row"); }
  function probeEl(){ return $("#deviceProbe"); }
  function portEl(){ return $("#devicePort"); }

  /* ---------- diagram markers (sinkron dengan D.cursor/D.done), lintas 2 layer (depan/belakang) ---------- */
  function syncPointVisuals(){
    const pts = document.querySelectorAll(".diagram-stage .point");
    if(!pts.length) return;
    pts.forEach((p)=>{
      const i = parseInt(p.dataset.idx, 10);
      const done = D.done[i];
      const isActive = i === D.cursor && D.state !== "allDone";
      p.classList.toggle("done", !!done);
      p.classList.toggle("selected", isActive && (D.state === "idle" || D.state === "recording" || D.state === "badsignal") && !done);
      p.classList.toggle("docked", D.probeDockedPoint === i && !done);
    });
  }

  function autoSwitchViewForCursor(){
    // dipanggil HANYA saat titik aktif benar2 berganti (bukan tiap render),
    // supaya tidak "menarik balik" toggle manual milik pengguna
    if(window.__setAntarakalaView) window.__setAntarakalaView(D.cursor >= 4);
  }

  function pointElsList(){
    // hanya titik yang SEDANG TAMPIL (layer depan/belakang yang aktif), bukan yang di-hidden
    return Array.from(document.querySelectorAll(".diagram-stage .point")).filter(p => p.offsetParent !== null);
  }

  function updateCable(){
    const stage = stageEl(), probe = probeEl(), port = portEl();
    const path = $("#cablePath");
    if(!stage || !probe || !port || !path) return;
    const s = scaleFactor();
    const stageR = stage.getBoundingClientRect();
    const portR = port.getBoundingClientRect();
    const probeR = probe.getBoundingClientRect();
    const x0 = (portR.left - stageR.left)/s + portR.width/s/2;
    const y0 = (portR.top - stageR.top)/s + portR.height/s/2;
    const x1 = (probeR.left - stageR.left)/s + probeR.width/s/2;
    const y1 = (probeR.top - stageR.top)/s + probeR.height/s*0.28;
    const dist = Math.hypot(x1-x0, y1-y0);
    const sag = Math.min(70, dist*0.22);
    const mx = (x0+x1)/2, my = (y0+y1)/2 + sag;
    path.setAttribute("d", `M ${x0} ${y0} Q ${mx} ${my} ${x1} ${y1}`);
  }

  const PROBE_CONTACT_X = 0.50;
  const PROBE_CONTACT_Y = 0.63;

  function resetProbePosition(animate){
    const stage = stageEl(), probe = probeEl(), port = portEl();
    if(!stage || !probe || !port) return;
    D.probeDockedPoint = null;
    syncPointVisuals();
    const s = scaleFactor();
    const stageR = stage.getBoundingClientRect();
    const portR = port.getBoundingClientRect();
    const x = (portR.left - stageR.left)/s - 4;
    const y = (portR.top - stageR.top)/s - 25;
    if(animate){
      probe.classList.add("snap-transition");
      probe.style.left = x + "px";
      probe.style.top = y + "px";
      animateCableFor(400);
      setTimeout(()=>probe.classList.remove("snap-transition"), 400);
    } else {
      probe.style.left = x + "px";
      probe.style.top = y + "px";
      requestAnimationFrame(updateCable);
    }
  }

  function probeContactClient(){
    const probe = probeEl();
    if(!probe) return null;
    const r = probe.getBoundingClientRect();
    return {
      x: r.left + r.width * PROBE_CONTACT_X,
      y: r.top + r.height * PROBE_CONTACT_Y,
    };
  }

  function snapProbeToPoint(point, animate){
    const stage = stageEl(), probe = probeEl();
    if(!stage || !probe || !point) return;
    const s = scaleFactor();
    const stageR = stage.getBoundingClientRect();
    const pointR = point.getBoundingClientRect();
    const targetX = (pointR.left - stageR.left)/s + pointR.width/(2*s);
    const targetY = (pointR.top - stageR.top)/s + pointR.height/(2*s);
    const x = targetX - probe.offsetWidth * PROBE_CONTACT_X;
    const y = targetY - probe.offsetHeight * PROBE_CONTACT_Y;
    if(animate) probe.classList.add("snap-transition");
    probe.style.left = x + "px";
    probe.style.top = y + "px";
    if(animate){
      animateCableFor(360);
      setTimeout(()=>probe.classList.remove("snap-transition"), 360);
    } else {
      requestAnimationFrame(updateCable);
    }
  }

  function animateCableFor(ms){
    const start = performance.now();
    function step(t){
      updateCable();
      if(t - start < ms) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function findNearestPoint(){
    const contact = probeContactClient();
    if(!contact) return null;
    let best = null, bestDist = Infinity;
    pointElsList().forEach(p=>{
      const idx = parseInt(p.dataset.idx, 10);
      if(p.classList.contains("done")) return;
      const r = p.getBoundingClientRect();
      const cx = r.left + r.width/2, cy = r.top + r.height/2;
      const d = Math.hypot(contact.x-cx, contact.y-cy);

      // Hit-test benar-benar mengikuti lingkaran marker.
      // Toleransi hanya sedikit di LUAR tepi lingkaran (±2.5 px layar),
      // jadi kepala stetoskop harus tampak tepat menempel pada titik.
      const pointRadius = Math.min(r.width, r.height) / 2;
      const threshold = pointRadius + 2.5;
      if(d <= threshold && d < bestDist){ bestDist = d; best = p; }
    });
    return best;
  }
  function clearDragoverHighlights(){
    pointElsList().forEach(p=>p.classList.remove("dragover"));
  }

  function dropOnPoint(point){
    if(!D.phoneReady || D.state !== "idle" || !point) return;
    const idx = parseInt(point.dataset.idx, 10);
    if(!Number.isInteger(idx) || D.done[idx]) return;

    // Probe boleh ditempel ke titik mana pun yang belum selesai.
    // Posisi fisik probe menjadi pilihan titik pada alat, bukan sebaliknya.
    D.cursor = idx;
    D.probeDockedPoint = idx;
    autoSwitchViewForCursor();
    snapProbeToPoint(point, true);
    render();
    emit("antarakala:probe-docked", { index:idx, name:POINT_NAMES[idx] });
  }

  let dragging = false, dragOffX = 0, dragOffY = 0;

  function bindProbeDrag(){
    const probe = probeEl();
    if(!probe) return;
    probe.addEventListener("pointerdown", (e)=>{
      if(!D.phoneReady || D.state !== "idle"){ flashDenied(); return; }
      dragging = true;
      D.probeDockedPoint = null;
      syncPointVisuals();
      render();
      probe.classList.remove("snap-transition");
      probe.setPointerCapture(e.pointerId);
      const s = scaleFactor();
      const r = probe.getBoundingClientRect();
      dragOffX = (e.clientX - r.left)/s; dragOffY = (e.clientY - r.top)/s;
    });
    probe.addEventListener("pointermove", (e)=>{
      if(!dragging) return;
      const s = scaleFactor();
      const stageR = stageEl().getBoundingClientRect();
      probe.style.left = ((e.clientX - stageR.left)/s - dragOffX) + "px";
      probe.style.top = ((e.clientY - stageR.top)/s - dragOffY) + "px";
      updateCable();
      clearDragoverHighlights();
      const p = findNearestPoint();
      if(p) p.classList.add("dragover");
    });
    probe.addEventListener("pointerup", (e)=>{
      if(!dragging) return;
      dragging = false;
      const target = findNearestPoint();
      clearDragoverHighlights();
      if(target){
        dropOnPoint(target);
      } else {
        resetProbePosition(true);
        render();
      }
    });
  }

  window.addEventListener("resize", ()=>requestAnimationFrame(()=>{
    if(D.probeDockedPoint !== null){
      const p = pointElsList().find(el=>parseInt(el.dataset.idx,10)===D.probeDockedPoint);
      if(p) snapProbeToPoint(p, false);
      else resetProbePosition(false);
    } else {
      resetProbePosition(false);
    }
  }));

  /* ---------- public snapshot for app.js ---------- */
  window.AntarakalaDevice = {
    getSnapshot(){
      return {
        state: D.state,
        cursor: D.cursor,
        done: D.done.slice(),
        results: D.results.slice(),
        pointNames: POINT_NAMES.slice(),
        maxState: D.maxState,
        spo2: D.spo2,
        hr: D.hr,
      };
    },
    resetExam(){
      clearInterval(D.recTimer);
      D.recTimer = null;
      stopSampleAudio();
      D.state = "idle";
      D.cursor = 0;
      D.done = [false,false,false,false,false,false];
      D.results = [null,null,null,null,null,null];
      D.recElapsed = 0;
      D.probeDockedPoint = null;
      D.activeSample = null;
      D.usedSampleIds = [];
      clearTimeout(D.maxTimer);
      D.maxTimer = null;
      D.maxState = "idle";
      D.spo2 = null;
      D.hr = null;
      autoSwitchViewForCursor();
      render();
      requestAnimationFrame(()=>resetProbePosition(false));
      emit("antarakala:reset", {});
    },
    onViewSwitched(){
      // Pergantian sisi tubuh membatalkan docking agar titik berikutnya harus diposisikan ulang.
      D.probeDockedPoint = null;
      syncPointVisuals();
      requestAnimationFrame(()=>{ resetProbePosition(false); render(); });
    }
  };

  /* ---------- button wiring with press visual feedback ---------- */
  function bind(){
    document.querySelectorAll(".device-btn").forEach(btn=>{
      const fire = ()=>{
        btn.classList.add("pressed");
        setTimeout(()=>btn.classList.remove("pressed"), 160);
        const which = btn.dataset.devbtn;
        if(which==="kiri") pressKiri();
        else if(which==="kanan") pressKanan();
        else if(which==="pilih") pressPilih();
      };
      btn.addEventListener("click", fire);
    });
    bindProbeDrag();
  }

  document.addEventListener("DOMContentLoaded", ()=>{
    bind();
    render();
    // beri jeda sedikit supaya .showcase sudah selesai di-scale oleh script
    // fit() di index.html sebelum menghitung posisi docking probe
    setTimeout(resetProbePosition, 60);
  });

  document.addEventListener("antarakala:phone-nav", (e)=>{
    const ready = e.detail.screen === "proses-auskultasi";
    if(ready !== D.phoneReady){
      D.phoneReady = ready;
      if(!ready && D.state !== "recording") stopSampleAudio();
      if(ready){
        // Saat layar auskultasi dibuka, tampilkan sisi tubuh sesuai titik aktif.
        autoSwitchViewForCursor();
        requestAnimationFrame(()=>resetProbePosition(false));
      }
      render();
    }
  });
})();
