const API_BASE = window.location.origin;

const state = {
  conditions: new Set(),
  restrictions: new Set(),
};

const $ = (id) => document.getElementById(id);

function setupChipGroup(containerId, targetSet) {
  const container = $(containerId);
  container.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const value = chip.dataset.value;
      if (targetSet.has(value)) {
        targetSet.delete(value);
        chip.classList.remove("active");
      } else {
        targetSet.add(value);
        chip.classList.add("active");
      }
    });
  });
}
setupChipGroup("conditionChips", state.conditions);
setupChipGroup("restrictionChips", state.restrictions);

// ---------- Image capture / OCR ----------
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const preview = $("preview");
const ocrStatus = $("ocrStatus");
const ingredientText = $("ingredientText");

dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file) handleImageFile(file);
});
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) handleImageFile(file);
});

async function handleImageFile(file) {
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropzone.querySelector(".dropzone-inner").hidden = true;

  showStatus(ocrStatus, "Reading label…", false);
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/api/ocr`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `OCR failed (${res.status})`);
    }
    const data = await res.json();
    ingredientText.value = data.extracted_text;
    showStatus(
      ocrStatus,
      `Extracted ${data.extracted_text.split(/\s+/).length} words` +
        (data.confidence ? ` · ${(data.confidence * 100).toFixed(0)}% confidence — review before analyzing` : ""),
      false
    );
  } catch (err) {
    showStatus(ocrStatus, err.message, true);
  }
}

function showStatus(el, message, isError) {
  el.textContent = message;
  el.hidden = false;
  el.classList.toggle("error", isError);
}

// ---------- Analyze ----------
$("analyzeBtn").addEventListener("click", runAnalysis);

async function runAnalysis() {
  const text = ingredientText.value.trim();
  const analyzeStatus = $("analyzeStatus");
  if (text.length < 2) {
    showStatus(analyzeStatus, "Add an ingredient list first.", true);
    return;
  }

  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.textContent = "Reading…";
  analyzeStatus.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ingredient_text: text,
        profile: {
          conditions: Array.from(state.conditions),
          restrictions: Array.from(state.restrictions),
        },
        use_llm: true,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Analysis failed (${res.status})`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showStatus(analyzeStatus, err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Read the label";
  }
}

function renderResults(data) {
  $("results").hidden = false;
  $("summaryLine").textContent = data.overall_summary;

  const badge = $("novaBadge");
  badge.textContent = data.nova.classification.replace("nova_", "NOVA ").toUpperCase();
  badge.className = `nova-badge ${data.nova.classification}`;
  $("novaExplanation").textContent = data.nova.explanation;

  const additivesList = $("additivesList");
  additivesList.innerHTML = "";
  if (data.additives.length === 0) {
    additivesList.innerHTML = `<li class="empty-note">No E-numbers detected in the text.</li>`;
  } else {
    for (const a of data.additives) {
      const li = document.createElement("li");
      li.className = `finding-item rating-${a.rating}`;
      const warnings = Object.values(a.condition_warnings || {}).join(" ");
      li.innerHTML = `
        <div class="fi-title">${a.code} — ${a.name}</div>
        <div class="fi-body">${a.summary}${warnings ? " ⚠ " + warnings : ""}</div>
      `;
      additivesList.appendChild(li);
    }
  }

  const allergensList = $("allergensList");
  allergensList.innerHTML = "";
  if (data.allergens.length === 0) {
    allergensList.innerHTML = `<li class="empty-note">No conflicts with your selected profile.</li>`;
  } else {
    for (const a of data.allergens) {
      const li = document.createElement("li");
      li.className = "finding-item rating-amber";
      li.innerHTML = `
        <div class="fi-title">${a.restriction.replace("_", " ")} — "${a.matched_ingredient}"</div>
        <div class="fi-body">${a.note}</div>
      `;
      allergensList.appendChild(li);
    }
  }

  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}
