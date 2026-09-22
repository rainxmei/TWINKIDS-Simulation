/* =========================================================
   ANTARAKALA — Demo App Logic
   Model B (LightGBM) menjalankan model hasil training secara nyata
   di browser. Output akustik memakai sampel ICBHI nyata yang telah
   diinferensikan dengan rekonstruksi checkpoint CNN; RR dihitung dari
   anotasi respiratory-cycle dan Grad-CAM berasal dari CNN yang sama.
   ========================================================= */
(function(){
  "use strict";

  /* ---------------- constants ---------------- */
  const POINTS = [
    { id:1, name:"Posterior Atas Kiri"  },
    { id:2, name:"Posterior Atas Kanan" },
    { id:3, name:"Posterior Bawah Kiri" },
    { id:4, name:"Posterior Bawah Kanan"},
    { id:5, name:"Anterior Atas Kiri"   },
    { id:6, name:"Anterior Atas Kanan"  },
  ];

  const DANGER_SIGNS = [
    { key:"minum", title:"Tidak Bisa Minum atau Menyusu", desc:"Kesulitan asupan cairan oral yang mengancam hidrasi.",
      icon:'<path d="M6 3h12l-1.5 15a2 2 0 01-2 1.8h-5a2 2 0 01-2-1.8L6 3z"/><path d="M6 3l-1-2M18 3l1-2"/>' },
    { key:"muntah", title:"Muntah Setiap Kali", desc:"Tidak dapat mempertahankan makanan atau cairan di lambung.",
      icon:'<circle cx="12" cy="12" r="9"/><path d="M9 10c.5 1 1.5 1 2 0M13 10c.5 1 1.5 1 2 0"/><path d="M9 15c1.5 1.5 4.5 1.5 6 0"/>' },
    { key:"kejang", title:"Kejang", desc:"Riwayat atau episode kejang aktif selama sakit saat ini.",
      icon:'<path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z"/>' },
    { key:"letargis", title:"Letargis", desc:"Penurunan kesadaran signifikan, tidak merespons rangsangan wajar.",
      icon:'<rect x="2" y="13" width="16" height="6" rx="1.5"/><path d="M2 13v-2a3 3 0 013-3h6a3 3 0 013 3M20 15v4M2 19v1M2 13V9"/>' },
    { key:"chest", title:"Chest Indrawing", desc:"Tarikan dinding dada bagian bawah ke dalam saat menarik napas.",
      icon:'<path d="M12 2s7 7.58 7 12a7 7 0 11-14 0c0-4.42 7-12 7-12z"/>' },
  ];

  /* ---------------- state ---------------- */
  const state = {
    patient:{ name:"", age:null, gender:"" },
    danger:{},              // key -> 'ada' | 'tidak'
    vitals:{ temp:null, spo2:null, hr:null, grunt:"tidak" },
    points:[],              // hasil CNN disimpan internal, ditampilkan hanya pada hasil akhir
    result:null,
    dataConsent:null,
    lastExamScreen:"beranda",
    meetingNo: 1,
  };

  const RESULT_TEXT = {
    highPneumonia:{
      label:"RISIKO TINGGI — PNEUMONIA BERAT",
      lead:"Balita menunjukkan tanda pneumonia berat. Segera rujuk ke fasilitas kesehatan dengan kapasitas lebih tinggi.",
      actions:[
        "Berikan dosis pertama antibiotik oral sebelum rujukan",
        "Rujuk segera ke rumah sakit/fasilitas rujukan terdekat",
        "Pastikan balita didampingi selama perjalanan rujukan"
      ]
    },
    highDanger:{
      label:"RISIKO TINGGI — TANDA BAHAYA UMUM",
      lead:"Balita menunjukkan tanda bahaya yang memerlukan penanganan segera, tidak spesifik pneumonia. Segera rujuk untuk evaluasi lebih lanjut.",
      actions:[
        "Berikan dosis pertama antibiotik oral sebelum rujukan",
        "Rujuk segera ke rumah sakit/fasilitas rujukan terdekat",
        "Pastikan balita didampingi selama perjalanan rujukan"
      ]
    },
    mid:{
      label:"RISIKO SEDANG — PNEUMONIA",
      lead:"Balita menunjukkan tanda pneumonia tanpa tanda bahaya. Dapat ditangani di puskesmas dengan antibiotik oral.",
      actions:[
        "Berikan amoksisilin oral 40mg/kg per dosis, 2× sehari, selama 3-5 hari",
        "Jadwalkan kunjungan kontrol 2-3 hari kemudian",
        "Edukasi orang tua/wali mengenai tanda bahaya yang perlu diwaspadai"
      ]
    },
    low:{
      label:"RISIKO RENDAH — BUKAN PNEUMONIA",
      lead:"Tidak ditemukan tanda pneumonia maupun tanda bahaya. Balita dapat dirawat di rumah.",
      actions:[
        "Balita dapat dirawat di rumah",
        "Edukasi orang tua/wali mengenai tanda bahaya yang perlu diwaspadai",
        "Instruksikan untuk kembali ke puskesmas jika kondisi memburuk"
      ]
    }
  };

  /* ---------------- history (localStorage) ---------------- */
  const HKEY = "twinkids_history_v2";
  const LEGACY_HKEY = "stethokid_history_v2";
  function loadHistory(){
    try{
      const raw = localStorage.getItem(HKEY) || localStorage.getItem(LEGACY_HKEY);
      if(raw){
        const parsed = JSON.parse(raw);
        if(!localStorage.getItem(HKEY)) localStorage.setItem(HKEY, raw);
        return parsed;
      }
    }catch(e){}
    return [
      { name:"Ishwari Adiningrum", id:"RM-2026-1003", riskType:"mid", tier:"mid", spo2:94, hr:108, when:"6 Sep 2026, 22:10" },
      { name:"Ishwari Gyananda Adiningrum", id:"RM-2026-1002", riskType:"highPneumonia", tier:"high", spo2:89, hr:126, when:"6 Sep 2026, 21:07" },
      { name:"Asep Susanto Rahardja", id:"RM-2026-1001", riskType:"low", tier:"low", spo2:97, hr:102, when:"6 Sep 2026, 21:04" },
      { name:"Nadira Putri", id:"RM-2026-1000", riskType:"highDanger", tier:"high", spo2:96, hr:118, when:"6 Sep 2026, 20:42" }
    ];
  }
  function saveHistory(list){
    try{ localStorage.setItem(HKEY, JSON.stringify(list)); }catch(e){}
  }
  let history = loadHistory();

  /* ---------------- helpers ---------------- */
  const $  = (sel,root)=> (root||document).querySelector(sel);
  const $$ = (sel,root)=> Array.from((root||document).querySelectorAll(sel));
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  function rand(min,max){ return Math.random()*(max-min)+min; }
  function pick(arr){ return arr[Math.floor(Math.random()*arr.length)]; }
  function median(values){
    const a = values.filter(Number.isFinite).slice().sort((x,y)=>x-y);
    if(!a.length) return NaN;
    const m = Math.floor(a.length/2);
    return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
  }

  function isScreenVisible(name){
    const el = $(`.screen[data-screen="${name}"]`);
    return !!(el && el.classList.contains("visible"));
  }

  function showToast(msg){
    const t = $("#toast");
    $("#toastText").textContent = msg;
    t.classList.add("show");
    clearTimeout(showToast._tm);
    showToast._tm = setTimeout(()=>t.classList.remove("show"), 2200);
  }

  /* ---------------- navigation ---------------- */
  const NAV_GROUP = {
    "beranda":"beranda",
    "input-pasien":"pasien",
    "tanda-bahaya":"pasien",
    "riwayat":"pasien",
    "detail-riwayat":"pasien",
    "panduan-auskultasi":"pemeriksaan",
    "proses-auskultasi":"pemeriksaan",
    "input-parameter":"pemeriksaan",
    "proses-ai":"pemeriksaan",
    "hasil-skrining":"pemeriksaan",
    "penjelasan-ai":"analisis",
    "faktor-risiko":"analisis",
    "faktor-kontribusi":"analisis",
    "persetujuan-data":"analisis",
  };

  const NAV_ITEMS = [
    { key:"beranda", label:"Home", target:"beranda",
      icon:'<path d="M3 11l9-8 9 8"/><path d="M5 10v9a1 1 0 001 1h4v-6h4v6h4a1 1 0 001-1v-9"/>' },
    { key:"pasien", label:"History", target:"riwayat",
      icon:'<circle cx="12" cy="8" r="4"/><path d="M4 20c0-3.3 3.6-6 8-6s8 2.7 8 6"/>' },
  ];

  function renderBottomNav(){
    $$("[data-navbar]").forEach(nav=>{
      nav.innerHTML = NAV_ITEMS.map(it=>`
        <button class="nav-item" data-navkey="${it.key}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${it.icon}</svg>
          <span>${it.label}</span>
        </button>`).join("");
    });
  }

  function goTo(screenName){
    if(!screenName) return;
    $$(".screen").forEach(s=>s.classList.remove("visible"));
    const target = $(`.screen[data-screen="${screenName}"]`);
    if(!target){ return; }
    target.classList.add("visible");
    $("#appScroll").scrollTop = 0;
    const grp = NAV_GROUP[screenName];
    $$(".nav-item").forEach(b=>b.classList.toggle("active", b.dataset.navkey===grp));
    if(NAV_GROUP[screenName]==="pemeriksaan") state.lastExamScreen = screenName;
    afterNav(screenName);
    document.dispatchEvent(new CustomEvent("antarakala:phone-nav", { detail: { screen: screenName } }));
  }

  function afterNav(screenName){
    if(screenName==="beranda") renderBeranda();
    if(screenName==="tanda-bahaya"){ renderDangerList(); syncDistressControls(); }
    if(screenName==="proses-auskultasi") syncAuscultationScreen();
    if(screenName==="input-parameter") prefillParameter();
    if(screenName==="hasil-skrining") renderHasil();
    if(screenName==="penjelasan-ai") renderPenjelasan();
    if(screenName==="faktor-risiko") renderFaktorRisiko();
    if(screenName==="faktor-kontribusi") renderFaktorKontribusi();
    if(screenName==="persetujuan-data") syncConsentScreen();
    if(screenName==="riwayat") renderRiwayat();
  }

  /* delegate all nav / action clicks */
  document.addEventListener("click", (e)=>{
    const navBtn = e.target.closest("[data-nav]");
    if(navBtn){ goTo(navBtn.dataset.nav); return; }

    const navItem = e.target.closest(".nav-item");
    if(navItem){
      const item = NAV_ITEMS.find(i=>i.key===navItem.dataset.navkey);
      if(item.target==="__exam__"){
        goTo(state.result ? "hasil-skrining" : (state.patient.name ? state.lastExamScreen : "input-pasien"));
      } else if(item.target==="__analysis__"){
        if(state.result) goTo("hasil-skrining");
        else showToast("Belum ada hasil pemeriksaan pada sesi ini");
      } else {
        goTo(item.target);
      }
      return;
    }

    const actionBtn = e.target.closest("[data-action]");
    if(actionBtn){ handleAction(actionBtn.dataset.action); return; }

    // segmented control selection
    const segBtn = e.target.closest(".seg button");
    if(segBtn){
      const seg = segBtn.closest(".seg");
      $$("button", seg).forEach(b=>b.classList.remove("selected"));
      segBtn.classList.add("selected");
      if(segBtn.dataset.danger==="1") segBtn.classList.add("danger");
      onSegChange(seg.id, segBtn.dataset.val);
      return;
    }
  });

  function handleAction(action){
    if(action==="open-about") $("#aboutModal").classList.add("visible");
    if(action==="close-about") $("#aboutModal").classList.remove("visible");
    if(action==="close-history-detail") $("#historyDetailModal").classList.remove("visible");
    if(action==="download-report") downloadReport();
    if(action==="save-finish") finishAndSave();
    if(action==="export-csv") downloadHistoryCSV();
    if(action==="export-patient") downloadCurrentPatientReport();
  }

  /* ---------------- BERANDA ---------------- */
  function renderBeranda(){
    $("#statPasien").textContent = history.length;
    const list = history.slice(0,3);
    $("#homeHistoryList").innerHTML = list.map(h=>historyItemHTML(h)).join("") ||
      `<p style="font-size:12.5px;color:var(--ink-300);">Belum ada riwayat pemeriksaan.</p>`;
  }

  function getRiskType(h){
    if(h && h.riskType) return h.riskType;
    if(h && h.tier === "high") return "highPneumonia";
    return h && h.tier === "mid" ? "mid" : "low";
  }
  function riskLabelFor(h){
    const type = getRiskType(h);
    if(type === "highPneumonia") return "RISIKO TINGGI - PNEUMONIA BERAT";
    if(type === "highDanger") return "RISIKO TINGGI - TANDA BAHAYA (BUKAN PNEUMONIA)";
    if(type === "mid") return "RISIKO SEDANG - PNEUMONIA";
    return "RISIKO RENDAH - BUKAN PNEUMONIA";
  }
  function historyActionFor(h){
    const type = getRiskType(h);
    if(type === "highPneumonia" || type === "highDanger") return "Rujukan Segera";
    if(type === "mid") return "Pengobatan & Kontrol";
    return "Perawatan Rumah";
  }
  function historyItemHTML(h){
    const type = getRiskType(h);
    const tier = h.tier || (type.startsWith("high") ? "high" : type);
    const pill = tier === "high" ? "pill-red" : tier === "mid" ? "pill-amber" : "pill-green";
    const cls = tier === "high" ? "high" : "";
    return `<div class="history-item home-history-item ${cls}">
      <div class="num">${tier==="high" ? "!" : "✓"}</div>
      <div class="content">
        <h4>${h.name}</h4>
        <p class="home-risk-line">${riskLabelFor(h)}</p>
        <div class="history-footer history-footer-home-only">
          <span class="pill ${pill}">${historyActionFor(h)}</span>
        </div>
      </div>
    </div>`;
  }

  /* ---------------- INPUT PASIEN ---------------- */
  function onSegChange(segId, val){
    if(segId==="pGender") state.patient.gender = val;
    if(segId==="vGrunt") state.vitals.grunt = val;
    if(segId==="dataConsent"){ state.dataConsent = val; syncConsentScreen(); }
    if(segId.startsWith("danger-")) state.danger[segId.replace("danger-","")] = val;
    validatePatientForm();
  }

  function updateAgeScopeNote(totalMonths){
    const note = $("#ageScopeNote");
    if(!note) return;
    const invalid = !Number.isFinite(totalMonths) || totalMonths < 0 || totalMonths > 59;
    note.hidden = !invalid;
    note.textContent = "Twinkids ditujukan untuk anak usia 0–59 bulan (maksimal 4 tahun 11 bulan).";
  }

  function validatePatientForm(){
    const p = state.patient;
    const ageValid = Number.isFinite(p.age) && p.age >= 0 && p.age <= 59;
    const ok = $("#pName").value.trim().length>1 && ageValid && p.gender;
    $("#btnToDanger").disabled = !ok;
  }
  $("#pName") && $("#pName").addEventListener("input", ()=>{ state.patient.name=$("#pName").value; validatePatientForm(); });

  function setAgeFromMonths(totalMonths){
    totalMonths = Math.round(totalMonths);
    state.patient.age = totalMonths;
    $("#pAgeYear").value = totalMonths >= 0 ? Math.floor(totalMonths/12) : "";
    $("#pAgeMonth").value = totalMonths >= 0 ? ((totalMonths%12)+12)%12 : "";
    updateAgeScopeNote(totalMonths);
    validatePatientForm();
  }

  function birthdateFromAgeMonths(totalMonths){
    if(!Number.isFinite(totalMonths) || totalMonths < 0) return "";
    const today = new Date();
    const targetMonth = new Date(today.getFullYear(), today.getMonth() - totalMonths, 1);
    const lastDay = new Date(targetMonth.getFullYear(), targetMonth.getMonth() + 1, 0).getDate();
    targetMonth.setDate(Math.min(today.getDate(), lastDay));
    const y = targetMonth.getFullYear();
    const m = String(targetMonth.getMonth() + 1).padStart(2, "0");
    const d = String(targetMonth.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  $("#pBirthdate") && $("#pBirthdate").addEventListener("input", ()=>{
    const val = $("#pBirthdate").value;
    if(!val){ state.patient.age = null; updateAgeScopeNote(null); validatePatientForm(); return; }
    const [birthYear, birthMonth, birthDay] = val.split("-").map(Number);
    const birth = new Date(birthYear, birthMonth - 1, birthDay);
    const today = new Date();
    let months = (today.getFullYear()-birth.getFullYear())*12 + (today.getMonth()-birth.getMonth());
    if(today.getDate() < birth.getDate()) months -= 1;
    if(months < 0 || isNaN(months)){
      state.patient.age = null;
      updateAgeScopeNote(-1);
      validatePatientForm();
      return;
    }
    setAgeFromMonths(months);
  });

  function onManualAgeInput(){
    const yRaw = $("#pAgeYear").value.trim();
    const mRaw = $("#pAgeMonth").value.trim();
    const y = parseInt(yRaw, 10);
    const m = parseInt(mRaw, 10);
    if(!yRaw && !mRaw){
      state.patient.age = null;
      $("#pBirthdate").value = "";
      updateAgeScopeNote(null);
      validatePatientForm();
      return;
    }
    const totalMonths = (Number.isFinite(y) ? y : 0) * 12 + (Number.isFinite(m) ? m : 0);
    state.patient.age = totalMonths;
    $("#pBirthdate").value = birthdateFromAgeMonths(totalMonths);
    updateAgeScopeNote(totalMonths);
    validatePatientForm();
  }
  $("#pAgeYear") && $("#pAgeYear").addEventListener("input", onManualAgeInput);
  $("#pAgeMonth") && $("#pAgeMonth").addEventListener("input", onManualAgeInput);

  $("#btnToDanger") && $("#btnToDanger").addEventListener("click", ()=>{
    if($("#btnToDanger").disabled) return;
    goTo("tanda-bahaya");
  });

  /* ---------------- TANDA BAHAYA ---------------- */
  function syncDistressControls(){
    if(!state.vitals.grunt) state.vitals.grunt = "tidak";
    const seg = $("#vGrunt");
    if(!seg) return;
    $$("button", seg).forEach(btn=>{
      const selected = btn.dataset.val === state.vitals.grunt;
      btn.classList.toggle("selected", selected);
      btn.classList.toggle("danger", selected && btn.dataset.danger === "1");
    });
  }

  function renderDangerList(){
    $("#dangerList").innerHTML = DANGER_SIGNS.map(d=>{
      if(!state.danger[d.key]) state.danger[d.key] = "tidak";
      const val = state.danger[d.key];
      return `<div class="danger-item">
        <div class="danger-item-top">
          <div><h4>${d.title}</h4><p>${d.desc}</p></div>
        </div>
        <div class="seg seg-2" id="danger-${d.key}">
          <button data-val="tidak" class="${val==="tidak"?"selected":""}">Tidak</button>
          <button data-val="ada" data-danger="1" class="${val==="ada"?"selected danger":""}">Ya</button>
        </div>
      </div>`;
    }).join("");
  }

  /* ---------------- PANDUAN → PROSES AUSKULTASI (disinkronkan dengan perangkat fisik) ---------------- */
  function renderPointList(activeIdx, mode){
    $("#pointList").innerHTML = POINTS.map((p,i)=>{
      const done = state.points[i];
      let cls = "point-row";
      if(i===activeIdx && (mode==="recording"||mode==="badsignal")) cls+=" active"; else if(done) cls+=" done";
      let statusHtml = "";
      if(done){
        statusHtml = `<span class="point-status tag-normal">✓ Selesai</span>`;
      } else if(i===activeIdx && mode==="recording"){
        statusHtml = `<span class="point-status" style="color:var(--green-700)">Merekam…</span>`;
      } else if(i===activeIdx && mode==="badsignal"){
        statusHtml = `<span class="point-status tag-wheeze">⚠ Sinyal lemah</span>`;
      }
      return `<div class="${cls}"><div class="point-num">${done?"✓":p.id}</div><div class="point-name">${p.name}</div>${statusHtml}</div>`;
    }).join("");
    updateLanjutButton();
  }

  function updateLanjutButton(){
    const btn = $("#btnLanjutAuskultasi");
    if(!btn) return;
    const allDone = state.points.length===6 && state.points.every(p=>p && p.result);
    const sensorDone = state.vitals.spo2 !== null && state.vitals.hr !== null && Number.isFinite(Number(state.vitals.spo2)) && Number.isFinite(Number(state.vitals.hr));
    btn.disabled = !(allDone && sensorDone);
  }

  function setTimerDisplay(pct, secLabel){
    $("#timerBarFill").style.width = (pct*100)+"%";
    $("#timerNum").textContent = secLabel;
  }

  function formatMMSS(totalSeconds){
    const safe = Math.max(0, Number(totalSeconds) || 0);
    const mins = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60);
    return `${String(mins).padStart(2,"0")}:${String(secs).padStart(2,"0")}`;
  }

  function syncAuscultationScreen(){
    state.points = new Array(6).fill(null);
    const snap = window.AntarakalaDevice ? window.AntarakalaDevice.getSnapshot() : null;
    if(snap){
      snap.results.forEach((r,i)=>{ if(r) state.points[i] = { id:i+1, name:snap.pointNames[i], ...r }; });
      if(snap.state === "recording"){
        $("#activePointLabel").textContent = `Titik Aktif: ${snap.cursor+1}. ${snap.pointNames[snap.cursor]}`;
        renderPointList(snap.cursor, "recording");
      } else if(snap.state === "badsignal"){
        $("#activePointLabel").textContent = `⚠ Sinyal Lemah, Mengulang Titik ${snap.cursor+1}`;
        renderPointList(snap.cursor, "badsignal");
      } else if(snap.state === "allDone"){
        if(snap.maxState === "done"){
          state.vitals.spo2 = Number(snap.spo2);
          state.vitals.hr = Number(snap.hr);
          $("#activePointLabel").textContent = "Pemeriksaan selesai. Siap memulai analisis";
        } else {
          $("#activePointLabel").textContent = "Tempelkan jari sesuai instruksi di perangkat fisik.";
        }
        renderPointList(-1, "waiting");
      } else {
        $("#activePointLabel").textContent = "Menunggu perangkat mulai merekam…";
        renderPointList(-1, "waiting");
      }
    } else {
      $("#activePointLabel").textContent = "Menunggu perangkat mulai merekam…";
      renderPointList(-1, "waiting");
    }
    setTimerDisplay(0, "00:00 / 00:09");
  }

  let phoneAusTimer = null;

  document.addEventListener("antarakala:point-start", (e)=>{
    const { index, name, duration, realDurationMs } = e.detail;
    if(!isScreenVisible("proses-auskultasi")) return;
    $("#activePointLabel").textContent = `Titik Aktif: ${index+1}. ${name}`;
    renderPointList(index, "recording");
    let elapsed = 0;
    clearInterval(phoneAusTimer);
    phoneAusTimer = setInterval(()=>{
      elapsed += 100;
      const targetMs = Number(realDurationMs) || duration*1000;
      const pct = clamp(elapsed/targetMs, 0, 1);
      const elapsedSec = Math.min(duration, Math.floor(pct*duration));
      setTimerDisplay(pct, `${formatMMSS(elapsedSec)} / ${formatMMSS(duration)}`);
      if(elapsed >= targetMs) clearInterval(phoneAusTimer);
    }, 100);
  });

  document.addEventListener("antarakala:signal-warning", (e)=>{
    if(!isScreenVisible("proses-auskultasi")) return;
    clearInterval(phoneAusTimer);
    $("#activePointLabel").textContent = `⚠ Sinyal Lemah, Mengulang Titik ${e.detail.index+1}`;
    setTimerDisplay(1, "Mengulang…");
    renderPointList(e.detail.index, "badsignal");
  });

  document.addEventListener("antarakala:point-result", (e)=>{
    const { index, name, result, confidence, rr, gradcam, audio, sampleId, sourceRecording, annotationCycle, probabilities } = e.detail;
    state.points[index] = { id:index+1, name, result, confidence, rr, gradcam, audio, sampleId, sourceRecording, annotationCycle, probabilities };
    if(isScreenVisible("proses-auskultasi")){
      renderPointList(index, "waiting");
      setTimerDisplay(0, "00:00 / 00:09");
      const doneCount = state.points.filter(Boolean).length;
      $("#activePointLabel").textContent = doneCount>=6
        ? "Tempelkan jari sesuai instruksi di perangkat fisik."
        : `✓ Titik ${index+1} selesai, bersiap titik berikutnya`;
    }
  });

  document.addEventListener("antarakala:all-done", ()=>{
    if(isScreenVisible("proses-auskultasi")){
      $("#activePointLabel").textContent = "Tempelkan jari sesuai instruksi di perangkat fisik.";
    }
    updateLanjutButton();
  });

  document.addEventListener("antarakala:max-start", ()=>{
    if(isScreenVisible("proses-auskultasi")){
      $("#activePointLabel").textContent = "Tempelkan jari sesuai instruksi di perangkat fisik.";
    }
    updateLanjutButton();
  });

  document.addEventListener("antarakala:max-result", (e)=>{
    state.vitals.spo2 = Number(e.detail.spo2);
    state.vitals.hr = Number(e.detail.hr);
    if(isScreenVisible("proses-auskultasi")){
      $("#activePointLabel").textContent = "Pemeriksaan selesai. Siap memulai analisis";
    }
    updateLanjutButton();
  });


  document.addEventListener("antarakala:reset", ()=>{
    if(isScreenVisible("proses-auskultasi")) syncAuscultationScreen();
  });

  $("#btnLanjutAuskultasi") && $("#btnLanjutAuskultasi").addEventListener("click", ()=>{
    if($("#btnLanjutAuskultasi").disabled) return;
    goTo("proses-ai");
    runAIProcessing();
  });

  /* ---------------- INPUT SUHU SEBELUM AUSKULTASI ---------------- */
  function validateTempForm(){
    const el = $("#vTemp");
    const raw = el ? String(el.value).trim().replace(",", ".") : "";
    const val = raw === "" ? NaN : parseFloat(raw);
    const ok = Number.isFinite(val);
    if($("#btnToPanduan")) $("#btnToPanduan").disabled = !ok;
    return ok;
  }

  function prefillParameter(){
    const el = $("#vTemp");
    if(el){
      el.value = Number.isFinite(state.vitals.temp) ? String(state.vitals.temp).replace(".", ",") : "";
    }
    validateTempForm();
  }

  $("#vTemp") && $("#vTemp").addEventListener("input", ()=>{
    validateTempForm();
  });

  $("#btnToPanduan") && $("#btnToPanduan").addEventListener("click", ()=>{
    if(!validateTempForm()) return;
    const raw = String($("#vTemp").value).trim().replace(",", ".");
    state.vitals.temp = parseFloat(raw);
    if(!state.vitals.grunt) state.vitals.grunt = "tidak";
    goTo("panduan-auskultasi");
  });

  /* ---------------- PROSES AI ---------------- */
  function runAIProcessing(){
    const rows = $$("#aiSteps .step-row");
    rows.forEach(r=>{ r.classList.remove("done","current"); $(".step-tag",r).textContent="ANTRIAN"; });
    let i=0;
    function next(){
      if(!isScreenVisible("proses-ai")) return;
      if(i>0){ rows[i-1].classList.remove("current"); rows[i-1].classList.add("done"); $(".step-tag",rows[i-1]).textContent="SELESAI"; }
      if(i>=rows.length){
        computeResult();
        setTimeout(()=>{ if(isScreenVisible("proses-ai")) goTo("hasil-skrining"); }, 450);
        return;
      }
      rows[i].classList.add("current");
      $(".step-tag",rows[i]).textContent="MEMPROSES";
      i++;
      setTimeout(next, 750);
    }
    next();
  }

  /* ---------------- MODEL B: LIGHTGBM REAL INFERENCE ---------------- */
  function computeResult(){
    const p = state.patient, v = state.vitals;
    const generalDangerKeys = ["minum","muntah","kejang","letargis"];
    const generalDangerAda = generalDangerKeys.some(k=>state.danger[k]==="ada");
    const chestAda = state.danger.chest === "ada";

    const crackleCount = state.points.filter(pt=>pt && pt.result==="crackle").length;
    const wheezeCount  = state.points.filter(pt=>pt && pt.result==="wheeze").length;

    // Hasil akustik berasal dari sampel respiratory-cycle ICBHI yang benar-benar diputar
    // saat perekaman. Kelas titik mengikuti prediksi rekonstruksi CNN yang cocok dengan
    // anotasi sampel terpilih. Untuk Model B, crackle menjadi fitur boolean.
    const cracklePresent = crackleCount >= 1;

    // RR diambil dari anotasi respiratory-cycle rekaman sumber. Karena enam titik bisa
    // memakai rekaman berbeda, median RR dipakai agar lebih stabil terhadap outlier.
    const ageM = Number.isFinite(p.age) ? p.age : 18;
    const rrThreshold = ageM < 2 ? 60 : (ageM < 12 ? 50 : 40);
    const rrCandidates = state.points.map(pt=>pt ? Number(pt.rr) : NaN).filter(Number.isFinite);
    const rrMeasured = median(rrCandidates);
    const rrValue = Number.isFinite(rrMeasured) ? Math.round(rrMeasured) : rrThreshold;

    // Product-level override. Model training hanya memiliki override SpO2<90;
    // tanda bahaya umum dan chest indrawing dipertahankan sebagai rule keselamatan UI.
    const override = generalDangerAda || chestAda || v.spo2 < 90;

    // 7 fitur persis seperti model training:
    // age_months, suhu, spo2, rr, status_pcv, flaring_grunting, crackle
    const modelFeatures = {
      age_months: ageM,
      suhu: Number(v.temp),
      spo2: Number(v.spo2),
      rr: Number(rrValue),
      status_pcv: 0,
      flaring_grunting: v.grunt === "ada" ? 1 : 0,
      crackle: cracklePresent ? 1 : 0,
    };

    let mlPred = null;
    if(!window.ANTARAKALA_LGBM || typeof window.ANTARAKALA_LGBM.predict !== "function" || typeof window.ANTARAKALA_LGBM.explain !== "function") {
      throw new Error("Model LightGBM/TreeSHAP tidak termuat");
    }
    mlPred = window.ANTARAKALA_LGBM.predict(modelFeatures);

    const tierMap = { rendah:"low", sedang:"mid", tinggi:"high" };
    let tier = override ? "high" : tierMap[mlPred.className];

    // Confidence menggunakan probabilitas kelas yang dipilih oleh LightGBM.
    // Pada override klinis, hasil berasal dari rule sehingga confidence ditampilkan 100%.
    const classKey = tier === "high" ? "tinggi" : tier === "mid" ? "sedang" : "rendah";
    const confidence = override ? 100 : (mlPred.probabilities[classKey] * 100);

    // TreeSHAP dijalankan pada kelas hasil yang sedang dijelaskan. Jika rule override
    // memaksa risiko tinggi, TreeSHAP tetap menjelaskan output kelas "tinggi" LightGBM;
    // pemicu override disimpan terpisah agar tidak disalahartikan sebagai nilai SHAP.
    const explainClassIndex = tier === "high" ? 2 : tier === "mid" ? 1 : 0;
    const shap = window.ANTARAKALA_LGBM.explain(modelFeatures, explainClassIndex);

    const fgText = v.grunt === "ada" ? "Ada" : "Tidak";
    const featureLabels = {
      age_months: `Usia (${ageM.toFixed(1).replace(/\.0$/,"")} bulan)`,
      suhu: `Suhu (${Number(v.temp).toFixed(1)}°C)`,
      spo2: `SpO₂ (${Number(v.spo2)}%)`,
      rr: `Laju Napas (${rrValue}/menit)`,
      status_pcv: `Parameter internal`,
      flaring_grunting: `Grunting (${fgText})`,
      crackle: cracklePresent ? `Crackle (${crackleCount} titik)` : "Crackle (Tidak terdeteksi)",
    };

    const absVals = shap.values.map(v=>Math.abs(v));
    const maxAbs = Math.max(...absVals, 1e-12);
    const sumAbs = absVals.reduce((a,b)=>a+b,0) || 1;
    const finalFactors = shap.featureNames.map((name,i)=>({
      feature: name,
      label: featureLabels[name] || name,
      shapValue: shap.values[i],
      positive: shap.values[i] >= 0,
      weight: Math.max(2, Math.abs(shap.values[i]) / maxAbs * 100),
      relPct: Math.abs(shap.values[i]) / sumAbs * 100,
    })).filter(f=>f.feature!=="status_pcv").sort((a,b)=>Math.abs(b.shapValue)-Math.abs(a.shapValue));

    const overrideReasons = [];
    if(generalDangerAda){
      DANGER_SIGNS.filter(d=>d.key!=="chest" && state.danger[d.key]==="ada").forEach(d=>overrideReasons.push(`Tanda Bahaya: ${d.title}`));
    }
    if(chestAda) overrideReasons.push("Tarikan Dinding Dada");
    if(v.spo2 < 90) overrideReasons.push(`SpO₂ ${v.spo2}% (<90%)`);

    const riskType = generalDangerAda ? "highDanger" : (tier === "high" ? "highPneumonia" : tier);

    state.result = {
      riskType, tier, confidence, total:null, override, overrideReasons, crackleCount, wheezeCount, rrValue, rrThreshold,
      rrSourceCount: rrCandidates.length,
      factors: finalFactors,
      modelFeatures,
      modelProbabilities: mlPred.probabilities,
      modelForcedHigh: mlPred.forcedHigh,
      modelHighThreshold: mlPred.highThreshold,
      modelSource: "LightGBM balanced weight, best iteration 87",
      shapClassName: shap.className,
      shapClassIndex: shap.classIndex,
      shapBaseValue: shap.baseValue,
      shapOutputValue: shap.outputValue,
      shapSumCheck: shap.sumCheck,
      shapExact: true,
    };
  }

  /* ---------------- HASIL SKRINING ---------------- */
  function renderHasil(){
    const r = state.result;
    if(!r) return;
    const banner = $("#resultBanner");
    banner.className = "result-banner " + r.tier;
    const info = RESULT_TEXT[r.riskType] || RESULT_TEXT.low;
    $("#resultTitle").textContent = info.label;
    $("#resultAction").innerHTML = `<p>${info.lead}</p>`;

    const tagMap = { crackle:"tag-crackle", wheeze:"tag-wheeze", normal:"tag-normal" };
    const labelMap = { crackle:"Crackle", wheeze:"Wheeze", normal:"Normal" };

    $("#pointResultList").innerHTML = state.points.map((pt,i)=>{
      const p = POINTS[i];
      if(!pt) return `<div class="point-row"><div class="point-num">${p.id}</div><div class="point-name">${p.name}</div></div>`;
      return `<div class="point-row done"><div class="point-num">${p.id}</div><div class="point-name">${p.name}</div><span class="point-status pill-tag ${tagMap[pt.result]}">${labelMap[pt.result]}</span></div>`;
    }).join("");

    const rrEl = $("#rrResultValue");
    const hrEl = $("#hrResultValue");
    const spo2El = $("#spo2ResultValue");
    if(rrEl) rrEl.textContent = `${r.rrValue}`;
    if(hrEl) hrEl.textContent = state.vitals.hr !== null && Number.isFinite(Number(state.vitals.hr)) ? `${Math.round(Number(state.vitals.hr))}` : "—";
    if(spo2El) spo2El.textContent = state.vitals.spo2 !== null && Number.isFinite(Number(state.vitals.spo2)) ? `${Math.round(Number(state.vitals.spo2))}` : "—";

    const actionsWrap = $("#resultActionsWrap");
    if(actionsWrap){
      actionsWrap.innerHTML = `<div class="card"><div class="card-title" style="margin-bottom:10px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6l1 2h3a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V6a1 1 0 011-1h3l1-2z"/><path d="M8 11l2 2 5-5M8 17h8"/></svg>Tindakan yang Disarankan</div><ol class="result-actions-list">${info.actions.map(x=>`<li>${x}</li>`).join("")}</ol></div>`;
    }
  }

  /* ---------------- PENJELASAN AKUSTIK / GRAD-CAM ---------------- */
  function renderPenjelasan(){
    const r = state.result;
    if(!r) return;
    const rank = { crackle:3, wheeze:2, normal:1 };
    const representative = state.points.filter(Boolean).slice().sort((a,b)=>{
      const rd = (rank[b.result]||0) - (rank[a.result]||0);
      if(rd) return rd;
      return (Number(b.confidence)||0) - (Number(a.confidence)||0);
    })[0];

    const img = $("#spectroImage");
    if(img && representative && representative.gradcam){
      img.src = representative.gradcam;
      img.alt = `Grad-CAM mel-spektrogram ${representative.result} dari ${representative.name}`;
    }

    if(!representative){
      $("#aiConclusionText").textContent = "Belum ada sampel auskultasi yang dapat divisualisasikan.";
      return;
    }

    const conf = Number.isFinite(Number(representative.confidence)) ? ` (${Math.round(Number(representative.confidence))}% keyakinan CNN)` : "";
    let finding;
    if(representative.result === "crackle") {
      finding = `2D CNN mengenali crackle pada ${representative.name}${conf}. Area merah–oranye adalah Grad-CAM nyata dari layer konvolusi terakhir dan menandai rentang waktu yang paling berkontribusi terhadap kelas crackle.`;
    } else if(representative.result === "wheeze") {
      finding = `2D CNN mengenali wheeze pada ${representative.name}${conf}. Area merah–oranye adalah Grad-CAM nyata dari layer konvolusi terakhir dan menandai rentang waktu yang paling berkontribusi terhadap kelas wheeze.`;
    } else {
      finding = `2D CNN mengklasifikasikan ${representative.name} sebagai suara paru normal${conf}. Area merah–oranye menunjukkan rentang waktu yang paling berkontribusi terhadap keputusan kelas normal.`;
    }
    $("#aiConclusionText").textContent = finding;
  }

  function drawSpectrogram(tier, crackleCount){
    const canvas = $("#spectroCanvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = "#070908";
    ctx.fillRect(0,0,W,H);

    const cols = 48, rows = 22;
    const cw = W/cols, ch = H/rows;
    const hotChance = tier==="high" ? 0.34 : tier==="mid" ? 0.20 : 0.08;

    for(let x=0;x<cols;x++){
      for(let y=0;y<rows;y++){
        const centerBias = 1 - Math.abs((y/rows)-0.5)*1.3;
        let v = Math.random()*0.5 + Math.random()*centerBias*0.5;
        if(Math.random() < hotChance*centerBias) v = 0.7 + Math.random()*0.3;
        const hue = v>0.62 ? lerpColor([201,54,74], [255,196,0], (v-0.62)/0.38) : lerpColor([12,70,60],[80,190,150], v/0.62);
        ctx.fillStyle = `rgb(${hue[0]},${hue[1]},${hue[2]})`;
        ctx.globalAlpha = 0.55 + v*0.45;
        ctx.fillRect(x*cw, H-(y+1)*ch, cw+0.6, ch+0.6);
      }
    }
    ctx.globalAlpha = 1;
  }
  function lerpColor(a,b,t){
    t = clamp(t,0,1);
    return [ Math.round(a[0]+(b[0]-a[0])*t), Math.round(a[1]+(b[1]-a[1])*t), Math.round(a[2]+(b[2]-a[2])*t) ];
  }

  /* ---------------- FAKTOR RISIKO / KONTRIBUSI ---------------- */
  function factorRowHTML(f, showValue){
    const barWidth = clamp(Number(f.weight)||0, 0, 100);
    const value = Number(f.shapValue)||0;
    const shapText = `${value>=0?"+":"−"}${Math.abs(value).toFixed(3)}`;
    return `<div class="factor-row">
      <div class="factor-top">
        <b>${f.label}</b>
        ${showValue ? `<span class="shap-value ${f.positive?'pos':'neg'}" title="Nilai TreeSHAP pada raw score kelas hasil">${shapText}</span>` : ""}
      </div>
      <div class="factor-track"><div class="factor-fill ${f.positive?'fill-shap-pos':'fill-shap-neg'}" style="width:${barWidth}%"></div></div>
    </div>`;
  }

  function renderFaktorRisiko(){
    const r = state.result; if(!r) return;
    const riskTitle = r.riskType === "highDanger" ? "Risiko Tinggi — Tanda Bahaya Umum" : r.riskType === "highPneumonia" ? "Risiko Tinggi — Pneumonia Berat" : r.tier === "mid" ? "Risiko Sedang — Pneumonia" : "Risiko Rendah — Bukan Pneumonia";
    $("#whyTitle").textContent = `Mengapa ${riskTitle}?`;
    $("#factorBars").innerHTML = r.factors.map(f=>factorRowHTML(f,true)).join("");
    $("#confidenceVal").textContent = r.confidence.toFixed(1)+"%";
    if($("#shapMeta")) {
      const cls = r.shapClassName ? r.shapClassName.toUpperCase() : "HASIL";
      $("#shapMeta").textContent = `TreeSHAP kelas ${cls} • base ${r.shapBaseValue.toFixed(3)} • output ${r.shapOutputValue.toFixed(3)}`;
    }
    if($("#shapOverrideNote")) {
      $("#shapOverrideNote").style.display = r.override ? "block" : "none";
      if(r.override) $("#shapOverrideNote").textContent = `Hasil risiko tinggi dipicu override klinis: ${r.overrideReasons.join(", ")}. Grafik TreeSHAP tetap menjelaskan skor kelas TINGGI dari LightGBM.`;
    }
  }

  function renderFaktorKontribusi(){
    const r = state.result, v = state.vitals, p = state.patient; if(!r) return;
    $("#factorBars2").innerHTML = r.factors.map(f=>factorRowHTML(f,true)).join("");
    $("#vgSpo2").textContent = v.spo2 + "%";
    $("#vgSpo2").className = "val " + (v.spo2<93?"val-danger":"val-ok");
    $("#vgTemp").textContent = v.temp.toFixed(1) + " °C";
    $("#vgTemp").className = "val " + (v.temp>=38?"val-danger":"val-ok");
    $("#vgChest").textContent = state.danger.chest==="ada" ? "Ada" : "Tidak";
    $("#vgChest").className = "val " + (state.danger.chest==="ada"?"val-danger":"val-ok");
    $("#vgGrunt").textContent = v.grunt==="ada" ? "Ada" : "Tidak";
    $("#vgGrunt").className = "val " + (v.grunt==="ada"?"val-danger":"val-ok");
  }

  function syncConsentScreen(){
    const seg = $("#dataConsent");
    if(seg){
      $$("button", seg).forEach(btn=>btn.classList.toggle("selected", btn.dataset.val === state.dataConsent));
    }
    const saveBtn = $("#btnConsentSave");
    if(saveBtn) saveBtn.disabled = !(state.dataConsent === "ya" || state.dataConsent === "tidak");
  }

  /* ---------------- SAVE / FINISH ---------------- */
  function finishAndSave(){
    const r = state.result, p = state.patient;
    if(!r){ showToast("Belum ada hasil untuk disimpan"); return; }
    if(state.dataConsent !== "ya" && state.dataConsent !== "tidak"){ showToast("Pilih persetujuan penggunaan data terlebih dahulu"); return; }
    const entry = {
      name: p.name || "Pasien Tanpa Nama",
      id: "RM-2026-" + String(1000+history.length),
      riskType: r.riskType,
      tier: r.tier,
      spo2: Number(state.vitals.spo2),
      hr: Number(state.vitals.hr),
      dataConsent: state.dataConsent,
      when: fullDateTime(),
      snapshot: {
        patient: { name:p.name, age:p.age, gender:p.gender },
        danger: { ...state.danger },
        vitals: { ...state.vitals },
        points: state.points.map(pt=>pt ? {...pt} : null),
        result: { riskType:r.riskType, tier:r.tier, confidence:r.confidence, rrValue:r.rrValue, rrThreshold:r.rrThreshold, override:r.override, overrideReasons:[...r.overrideReasons] },
        dataConsent: state.dataConsent
      },
    };
    history.unshift(entry);
    saveHistory(history);
    showToast("Hasil pemeriksaan tersimpan ke riwayat");
    state.patient = { name:"", age:null, gender:"" };
    state.danger = {};
    state.vitals = { temp:null, spo2:null, hr:null, grunt:"tidak" };
    state.points = new Array(6).fill(null);
    state.result = null;
    state.dataConsent = null;
    resetPatientForm();
    if(window.AntarakalaDevice && typeof window.AntarakalaDevice.resetExam === "function"){
      window.AntarakalaDevice.resetExam();
    }
    setTimeout(()=> goTo("beranda"), 300);
  }

  function resetPatientForm(){
    if($("#pName")) $("#pName").value = "";
    if($("#pBirthdate")) $("#pBirthdate").value = "";
    if($("#pAgeYear")) $("#pAgeYear").value = "";
    if($("#pAgeMonth")) $("#pAgeMonth").value = "";
    const ageNote = $("#ageScopeNote"); if(ageNote) ageNote.hidden = true;
    state.patient.age = null;
    $$(".seg button").forEach(b=>b.classList.remove("selected","danger"));
    syncDistressControls();
    if($("#vTemp")) $("#vTemp").value = "";
    validatePatientForm();
    syncConsentScreen();
  }

  /* ---------------- RIWAYAT ---------------- */
  let riwayatCurrentList = history;
  function historyPageItemHTML(h, i){
    const type = getRiskType(h);
    const tier = h.tier || (type.startsWith("high") ? "high" : type);
    const pill = tier === "high" ? "pill-red" : tier === "mid" ? "pill-amber" : "pill-green";
    const cls = tier === "high" ? "high" : "";
    return `<div class="history-item home-history-item history-page-item ${cls}" data-histidx="${i}">
      <div class="num">${tier==="high" ? "!" : "✓"}</div>
      <div class="content">
        <h4>${h.name}</h4>
        <p class="home-risk-line">${riskLabelFor(h)}</p>
        <div class="history-footer history-footer-stacked">
          <span class="pill ${pill}">${historyActionFor(h)}</span>
          <span class="history-time">${h.when}</span>
        </div>
      </div>
    </div>`;
  }

  function renderRiwayat(list){
    const data = list || history;
    riwayatCurrentList = data;
    $("#riwayatList").innerHTML = data.map((h,i)=> historyPageItemHTML(h,i)).join("") || `<p style="font-size:12.5px;color:var(--ink-300);">Tidak ada data ditemukan.</p>`;
  }
  $("#searchRiwayat") && $("#searchRiwayat").addEventListener("input",(e)=>{
    const q = e.target.value.toLowerCase();
    renderRiwayat(history.filter(h=> h.name.toLowerCase().includes(q) || ((h.id||"").toLowerCase().includes(q))));
  });
  $("#btnExportCsv") && $("#btnExportCsv").addEventListener("click", downloadHistoryCSV);

  const GENDER_LABEL = { L:"Laki-laki", P:"Perempuan" };
  const RESULT_TAG_LABEL = { crackle:"Crackle", wheeze:"Wheeze", normal:"Normal" };
  const DANGER_LABEL = { ada:"Ya", tidak:"Tidak" };
  let currentHistoryDetail = null;

  function formatAgeDetail(months){
    if(!Number.isFinite(months) || months < 0) return "—";
    const y = Math.floor(months/12);
    const m = months % 12;
    if(y && m) return `${y} tahun ${m} bulan`;
    if(y) return `${y} tahun`;
    return `${m} bulan`;
  }

  function buildHistoryDetailHTML(h){
    const s = h.snapshot || {};
    const result = s.result || {};
    const tier = h.tier || (getRiskType(h).startsWith("high") ? "high" : getRiskType(h));
    const tierInfo = tier === "high" ? {pill:"Rujukan Segera", cls:"high"} : tier === "mid" ? {pill:"Pengobatan & Kontrol", cls:"mid"} : {pill:"Perawatan Rumah", cls:"low"};
    const dangerRows = DANGER_SIGNS.map(d=>`<div class="detail-list-row"><span>${d.title}</span><b>${DANGER_LABEL[(s.danger||{})[d.key]] || "Tidak"}</b></div>`).join("");
    const distressRows = `<div class="detail-list-row"><span>Grunting</span><b>${(s.vitals && s.vitals.grunt==="ada") ? "Ya" : "Tidak"}</b></div>`;
    const pointsHtml = (s.points && s.points.length) ? s.points.map((pt,i)=>{
      const p = POINTS[i];
      if(!pt) return `<div class="point-row"><div class="point-num">${p.id}</div><div class="point-name">${p.name}</div></div>`;
      return `<div class="point-row done"><div class="point-num">${p.id}</div><div class="point-name">${p.name}</div><span class="point-status pill-tag tag-${pt.result}">${RESULT_TAG_LABEL[pt.result]}</span></div>`;
    }).join("") : `<div class="detail-list-row"><span>Data auskultasi</span><b>Tidak tersedia</b></div>`;

    return `<div class="detail-grid">
      <div class="card">
        <div class="detail-section-badge ${tierInfo.cls}">${tierInfo.pill}</div>
        <div class="card-title" style="margin-bottom:10px;">Identitas Pasien</div>
        <div class="detail-meta"><b>Nama:</b> ${h.name || "—"}</div>
        <div class="detail-meta"><b>Waktu Pemeriksaan:</b> ${h.when || "—"}</div>
        <div class="detail-meta"><b>Usia:</b> ${formatAgeDetail(s.patient && s.patient.age)}</div>
        <div class="detail-meta"><b>Jenis Kelamin:</b> ${GENDER_LABEL[s.patient && s.patient.gender] || "—"}</div>
      </div>
      <div class="card">
        <div class="card-title" style="margin-bottom:10px;">Hasil Skrining</div>
        <div class="detail-meta"><b>Status:</b> ${riskLabelFor(h)}</div>
        <div class="detail-meta"><b>Tingkat Kepercayaan:</b> ${Number.isFinite(result.confidence) ? result.confidence.toFixed(1)+"%" : "—"}</div>
        <div class="detail-meta"><b>RR:</b> ${Number.isFinite(result.rrValue) ? result.rrValue+"/menit" : "—"}</div>
        <div class="detail-meta"><b>HR:</b> ${Number.isFinite(Number(h.hr)) ? Math.round(Number(h.hr))+" bpm" : "—"}</div>
        <div class="detail-meta"><b>SpO₂:</b> ${Number.isFinite(Number(h.spo2)) ? Math.round(Number(h.spo2))+"%" : "—"}</div>
        <div class="detail-meta"><b>Suhu:</b> ${(s.vitals && Number.isFinite(s.vitals.temp)) ? s.vitals.temp.toFixed(1)+"°C" : "—"}</div>
        <div class="detail-meta"><b>Persetujuan penggunaan data:</b> ${(s.dataConsent || h.dataConsent)==="ya" ? "Ya" : (s.dataConsent || h.dataConsent)==="tidak" ? "Tidak" : "—"}</div>
      </div>
      <div class="card"><div class="card-title" style="margin-bottom:10px;">Tanda Bahaya</div><div class="detail-list">${dangerRows}</div></div>
      <div class="card"><div class="card-title" style="margin-bottom:10px;">Tanda Distres Pernapasan</div><div class="detail-list">${distressRows}</div></div>
      <div class="card"><div class="card-title" style="margin-bottom:10px;">Hasil Auskultasi 6 Titik</div><div class="point-list point-list-compact">${pointsHtml}</div></div>
    </div>`;
  }

  function openHistoryDetail(idx){
    const h = riwayatCurrentList[idx];
    if(!h) return;
    currentHistoryDetail = h;
    $("#historyDetailPageBody").innerHTML = buildHistoryDetailHTML(h);
    goTo("detail-riwayat");
  }
  document.addEventListener("click",(e)=>{
    const item = e.target.closest("#riwayatList .history-item[data-histidx]");
    if(item) openHistoryDetail(parseInt(item.dataset.histidx,10));
  });

  function htmlEscape(value){
    return String(value == null ? "" : value)
      .replace(/&/g,"&amp;")
      .replace(/</g,"&lt;")
      .replace(/>/g,"&gt;")
      .replace(/"/g,"&quot;")
      .replace(/'/g,"&#039;");
  }

  function downloadCurrentPatientReport(){
    const h = currentHistoryDetail;
    if(!h){ showToast("Pilih data pasien terlebih dahulu"); return; }
    const s = h.snapshot || {};
    const p = s.patient || {};
    const v = s.vitals || {};
    const r = s.result || {};
    const dangerRows = DANGER_SIGNS.map(d=>`<tr><td>${htmlEscape(d.title)}</td><td>${(s.danger && s.danger[d.key]==="ada") ? "Ya" : "Tidak"}</td></tr>`).join("");
    const pointRows = POINTS.map((pt,i)=>{
      const result = s.points && s.points[i];
      return `<tr><td>${pt.id}. ${htmlEscape(pt.name)}</td><td>${result ? htmlEscape(RESULT_TAG_LABEL[result.result] || result.result) : "—"}</td></tr>`;
    }).join("");
    const html = `<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Laporan Twinkids - ${htmlEscape(h.name||"Pasien")}</title><style>
      body{font-family:Arial,sans-serif;max-width:760px;margin:32px auto;padding:0 22px;color:#151A18;line-height:1.45}h1{color:#008C9E;margin-bottom:4px}h2{font-size:17px;color:#00626E;margin:24px 0 8px}p.meta{color:#6B746F;margin-top:0}table{width:100%;border-collapse:collapse;margin:8px 0 18px}td,th{border-bottom:1px solid #e5e9e6;padding:8px 6px;text-align:left;font-size:13px}td:first-child{width:52%;color:#3A423F}.risk{font-weight:700;font-size:18px;margin:8px 0 14px}.note{font-size:11px;color:#9AA39D;margin-top:28px}@media print{body{margin:0;max-width:none}.note{page-break-inside:avoid}}
    </style></head><body>
      <h1>Laporan Pemeriksaan Twinkids</h1><p class="meta">Diperiksa: ${htmlEscape(h.when||"—")}</p>
      <div class="risk">${htmlEscape(riskLabelFor(h))}</div>
      <h2>Identitas Pasien</h2><table>
        <tr><td>Nama</td><td>${htmlEscape(h.name||"—")}</td></tr>
        <tr><td>Usia</td><td>${htmlEscape(formatAgeDetail(p.age))}</td></tr>
        <tr><td>Jenis Kelamin</td><td>${htmlEscape(GENDER_LABEL[p.gender]||"—")}</td></tr>
      </table>
      <h2>Hasil Skrining</h2><table>
        <tr><td>SpO₂</td><td>${typeof h.spo2!=="undefined" ? htmlEscape(h.spo2)+"%" : "—"}</td></tr>
        <tr><td>HR</td><td>${typeof h.hr!=="undefined" ? htmlEscape(h.hr)+" bpm" : "—"}</td></tr>
        <tr><td>Suhu</td><td>${Number.isFinite(v.temp) ? htmlEscape(v.temp.toFixed(1))+" °C" : "—"}</td></tr>
        <tr><td>Laju Napas</td><td>${Number.isFinite(r.rrValue) ? htmlEscape(r.rrValue)+"/menit" : "—"}</td></tr>
        <tr><td>Grunting</td><td>${v.grunt==="ada" ? "Ya" : "Tidak"}</td></tr>
      </table>
      <h2>Tanda Bahaya</h2><table>${dangerRows}</table>
      <h2>Hasil Auskultasi 6 Titik</h2><table>${pointRows}</table>
      <p class="note">Laporan ini berasal dari purwarupa Twinkids dan merupakan alat bantu skrining, bukan diagnosis medis. File dapat dibuka di browser lalu dicetak atau disimpan sebagai PDF.</p>
    </body></html>`;
    const blob = new Blob([html], {type:"text/html;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Laporan_Twinkids_${String(h.name||"pasien").replace(/[^a-z0-9_-]+/gi,"_")}.html`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function csvEscape(val){
    const s = String(val == null ? "" : val);
    return '"' + s.replace(/"/g, '""') + '"';
  }

  function downloadHistoryCSV(){
    if(!history.length){ showToast("Belum ada data riwayat untuk diexport"); return; }
    const rows = [[
      "nama","waktu","risiko","usia_bulan","jenis_kelamin","spo2","hr","suhu","laju_napas","grunting","tanda_bahaya","persetujuan_penggunaan_data"
    ]];
    history.forEach(h=>{
      const s = h.snapshot || {};
      const dangerSelected = DANGER_SIGNS.filter(d => s.danger && s.danger[d.key] === "ada").map(d => d.title).join("; ");
      rows.push([
        h.name || "",
        h.when || "",
        riskLabelFor(h),
        s.patient && Number.isFinite(s.patient.age) ? s.patient.age : "",
        GENDER_LABEL[s.patient && s.patient.gender] || "",
        typeof h.spo2 !== "undefined" ? h.spo2 : "",
        typeof h.hr !== "undefined" ? h.hr : "",
        s.vitals && Number.isFinite(s.vitals.temp) ? s.vitals.temp.toFixed(1) : "",
        s.result && Number.isFinite(s.result.rrValue) ? s.result.rrValue : "",
        s.vitals && s.vitals.grunt === "ada" ? "Ya" : "Tidak",
        dangerSelected,
        (s.dataConsent || h.dataConsent || "")
      ]);
    });
    const csv = rows.map(row => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], {type:"text/csv;charset=utf-8;"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Riwayat_Twinkids_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Riwayat berhasil diexport ke CSV");
  }

  /* ---------------- REPORT DOWNLOAD ---------------- */
  function downloadReport(){
    const r = state.result;
    if(!r){ showToast("Belum ada hasil untuk diunduh"); return; }
    const p = state.patient, v = state.vitals;
    const info = RESULT_TEXT[r.riskType] || RESULT_TEXT.low;
    const actions = `<ol>${info.actions.map(x=>`<li>${x}</li>`).join("")}</ol>`;
    const html = `<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
    <title>Laporan Skrining Twinkids, ${p.name||"Pasien"}</title>
    <style>
      body{font-family:Arial,sans-serif; max-width:680px; margin:40px auto; color:#151A18; line-height:1.45;}
      h1{color:#00A3AE; font-size:22px; margin-bottom:2px;}
      .tag{display:inline-block; padding:6px 14px; border-radius:20px; font-weight:700; color:#fff; margin:10px 0 12px; background:${r.tier==='high'?'#D9364A':r.tier==='mid'?'#C98A00':'#00A3AE'};}
      .lead{margin:0 0 12px} table{width:100%; border-collapse:collapse; margin-bottom:18px;}
      td{padding:7px 4px; border-bottom:1px solid #eee; font-size:13.5px;} td:first-child{color:#6B746F; width:45%;}
      h3{font-size:14px; color:#00A3AE; margin:18px 0 8px;} .factor{display:flex; justify-content:space-between; font-size:13px; padding:5px 0; border-bottom:1px dashed #eee;}
      footer{margin-top:26px; font-size:11px; color:#9AA39D; line-height:1.6;}
    </style></head><body>
      <h1>Laporan Hasil Skrining, Twinkids</h1>
      <div class="tag">${info.label}</div>
      <p class="lead">${info.lead}</p>
      <h3>Tindakan yang disarankan</h3>${actions}
      <table>
        <tr><td>Nama Pasien</td><td>${p.name||"—"}</td></tr>
        <tr><td>Usia</td><td>${Number.isFinite(p.age)?p.age:"—"} bulan</td></tr>
        <tr><td>Jenis Kelamin</td><td>${p.gender==="L"?"Laki-laki":p.gender==="P"?"Perempuan":"—"}</td></tr>
        <tr><td>SpO₂</td><td>${Number.isFinite(Number(v.spo2))?Math.round(Number(v.spo2))+"%":"—"}</td></tr>
        <tr><td>HR</td><td>${Number.isFinite(Number(v.hr))?Math.round(Number(v.hr))+" bpm":"—"}</td></tr>
        <tr><td>Suhu Tubuh</td><td>${Number.isFinite(v.temp)?v.temp.toFixed(1)+" °C":"—"}</td></tr>
        <tr><td>RR</td><td>${r.rrValue}/menit</td></tr>
        <tr><td>Tingkat Kepercayaan AI</td><td>${r.confidence.toFixed(1)}%</td></tr>
        <tr><td>Persetujuan penggunaan data</td><td>${state.dataConsent==="ya"?"Ya":state.dataConsent==="tidak"?"Tidak":"Belum dipilih"}</td></tr>
      </table>
      <h3>Faktor Kontribusi Utama</h3>
      ${r.factors.map(f=>`<div class="factor"><span>${f.label}</span><span>${f.shapValue>=0?"+":"−"}${Math.abs(f.shapValue).toFixed(3)}</span></div>`).join("")}
      <footer>Dokumen ini dihasilkan oleh purwarupa Twinkids untuk demonstrasi sistem skrining. Hasil ini merupakan alat bantu skrining dan bukan diagnosis medis. Dibuat: ${new Date().toLocaleString("id-ID")}</footer>
    </body></html>`;
    const blob = new Blob([html], {type:"text/html"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Laporan_Twinkids_${(p.name||"pasien").replace(/\s+/g,"_")}.html`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Laporan berhasil diunduh");
  }

  /* ---------------- clock ---------------- */
  function nowHHMM(){
    const d = new Date();
    return String(d.getHours()).padStart(2,"0")+":"+String(d.getMinutes()).padStart(2,"0");
  }
  function tickClock(){ const el = $("#clock"); if(el) el.textContent = nowHHMM(); }
  function fullDateTime(){
    const d = new Date();
    const tgl = d.toLocaleDateString("id-ID", { day:"numeric", month:"short", year:"numeric" });
    return `${tgl}, ${nowHHMM()}`;
  }

  /* ---------------- init ---------------- */
  function init(){
    renderBottomNav();
    goTo("beranda");
    tickClock();
    setInterval(tickClock, 15000);
    // re-bind nav-item active state after nav render
    const grp = NAV_GROUP["beranda"];
    $$(".nav-item").forEach(b=>b.classList.toggle("active", b.dataset.navkey===grp));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
