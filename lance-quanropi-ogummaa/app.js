/**
 * app.js — Gummaa-Atlas frontend
 *
 * Responsibilities:
 *   - Google Maps init + click-to-coordinate capture
 *   - Chat message send / receive with coordinate injection
 *   - GeoJSON overlay rendering (markers + info windows)
 *   - Memory panel: fetch stats, display records, prune
 *   - Status dot management
 *   - Panel navigation (Chat ↔ Memory)
 */

(() => {
  "use strict";

  // ── State ────────────────────────────────────────────────────────────────
  let sessionId   = crypto.randomUUID();
  let selectedLat = null;
  let selectedLng = null;
  let mapInstance = null;
  let markers     = [];
  let infoWindows = [];

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const messagesEl  = document.getElementById("messages");
  const userInput   = document.getElementById("userInput");
  const sendBtn     = document.getElementById("sendBtn");
  const coordBadge  = document.getElementById("coordBadge");
  const coordText   = document.getElementById("coordText");
  const coordClear  = document.getElementById("coordClear");
  const statusDot   = document.getElementById("statusDot");
  const statusLabel = document.getElementById("statusLabel");
  const mapLegend   = document.getElementById("mapLegend");
  const navBtns     = document.querySelectorAll(".nav-btn");
  const panelChat   = document.getElementById("panelChat");
  const panelMemory = document.getElementById("panelMemory");

  // ── Status helpers ───────────────────────────────────────────────────────
  function setStatus(state, label) {
    statusDot.className = "status-dot " + state;
    statusLabel.textContent = label;
  }

  // ── Map initialisation (called by Google Maps script) ───────────────────
  window.initMap = function () {
    mapInstance = new google.maps.Map(document.getElementById("map"), {
      center:    { lat: 5.0, lng: 20.0 },   // Centre of Africa
      zoom:      4,
      mapTypeId: "terrain",
      styles:    DARK_MAP_STYLE,
      disableDefaultUI:          false,
      zoomControl:               true,
      mapTypeControl:            false,
      streetViewControl:         false,
      fullscreenControl:         true,
      gestureHandling:           "greedy",
    });

    // Click-to-select coordinate
    mapInstance.addListener("click", (e) => {
      selectedLat = e.latLng.lat();
      selectedLng = e.latLng.lng();
      coordText.textContent =
        `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`;
      coordBadge.hidden = false;

      // Drop a temporary pin at click site
      placeClickMarker(selectedLat, selectedLng);
    });

    setStatus("ok", "Map ready");
    document.querySelector(".brand-glyph").title = "Map loaded";
  };

  // ── Panel navigation ─────────────────────────────────────────────────────
  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      navBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const target = btn.dataset.panel;
      if (target === "memory") {
        panelMemory.hidden = false;
        fetchMemoryStats();
      } else {
        panelMemory.hidden = true;
      }
    });
  });

  // ── Clear coordinate ─────────────────────────────────────────────────────
  coordClear.addEventListener("click", () => {
    selectedLat = null;
    selectedLng = null;
    coordBadge.hidden = true;
  });

  // ── Map controls ─────────────────────────────────────────────────────────
  document.getElementById("btnClearMarkers").addEventListener("click", clearMarkers);
  document.getElementById("btnCenterAfrica").addEventListener("click", () => {
    if (mapInstance) mapInstance.panTo({ lat: 5.0, lng: 20.0 });
    if (mapInstance) mapInstance.setZoom(4);
  });

  // ── Send message ─────────────────────────────────────────────────────────
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = "";
    userInput.style.height = "auto";
    sendBtn.disabled = true;
    setStatus("busy", "Reasoning…");

    appendUserMessage(text);
    const typingEl = appendTypingIndicator();

    const payload = {
      session_id: sessionId,
      message:    text,
    };

    if (selectedLat !== null && selectedLng !== null) {
      payload.coordinates = { lat: selectedLat, lng: selectedLng };
    }

    try {
      const resp = await fetch("/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Server error");
      }

      const data = await resp.json();

      typingEl.remove();
      appendAIMessage(data.response);

      if (data.geojson) {
        renderGeoJSON(data.geojson, data.map_points || []);
      }

      setStatus("ok", "Ready");
    } catch (err) {
      typingEl.remove();
      appendAIMessage(`⚠ Error: ${err.message}`);
      setStatus("err", "Error");
    } finally {
      sendBtn.disabled = false;
    }
  }

  // ── Message renderers ─────────────────────────────────────────────────────
  function appendUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-user";
    wrap.innerHTML = `
      <div class="msg-avatar">U</div>
      <div class="msg-body"><p>${escapeHtml(text)}</p></div>
    `;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendAIMessage(md) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-ai";
    wrap.innerHTML = `
      <div class="msg-avatar">G</div>
      <div class="msg-body">${renderMarkdown(md)}</div>
    `;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  function appendTypingIndicator() {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-ai msg-typing";
    wrap.innerHTML = `
      <div class="msg-avatar">G</div>
      <div class="msg-body" style="padding:14px 18px;">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap;
  }

  // ── Tiny markdown renderer ────────────────────────────────────────────────
  function renderMarkdown(text) {
    return text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g,   "<em>$1</em>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n/g, "<br>")
      .replace(/^/, "<p>").replace(/$/, "</p>");
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ── Map overlays ─────────────────────────────────────────────────────────
  const SCOPE_COLORS = {
    historical:  "#d4a843",
    geospatial:  "#4caf82",
    political:   "#b85c38",
    cultural:    "#8b55b0",
    session:     "#5b8fc0",
  };

  function placeClickMarker(lat, lng) {
    if (!mapInstance) return;
    const marker = new google.maps.Marker({
      position: { lat, lng },
      map: mapInstance,
      icon: {
        path:        google.maps.SymbolPath.CIRCLE,
        scale:       7,
        fillColor:   "#7ab8e8",
        fillOpacity: 0.9,
        strokeColor: "#fff",
        strokeWeight: 1.5,
      },
      title: "Selected location",
      zIndex: 999,
    });
    markers.push(marker);
  }

  function renderGeoJSON(geojson, rawPoints) {
    if (!mapInstance) return;

    // Clear previous feature-set markers (keep click marker)
    clearFeatureMarkers();

    const bounds = new google.maps.LatLngBounds();
    const legendItems = [];

    (geojson.features || []).forEach((feat) => {
      const coords   = feat.geometry.coordinates;  // [lng, lat]
      const props    = feat.properties;
      const scope    = props.scope || "historical";
      const color    = SCOPE_COLORS[scope] || "#d4a843";

      const pos = { lat: coords[1], lng: coords[0] };
      bounds.extend(pos);

      const marker = new google.maps.Marker({
        position: pos,
        map: mapInstance,
        icon: {
          path:        google.maps.SymbolPath.CIRCLE,
          scale:       Math.max(8, (props.importance || 0.5) * 14),
          fillColor:   color,
          fillOpacity: 0.85,
          strokeColor: "#fff",
          strokeWeight: 1.5,
        },
        title: props.title,
        zIndex: 100,
      });

      const infoContent = `
        <div class="map-info-title">${props.title || "Location"}</div>
        <div class="map-info-body">${props.description || ""}</div>
      `;
      const iw = new google.maps.InfoWindow({ content: infoContent });

      marker.addListener("click", () => {
        infoWindows.forEach((w) => w.close());
        iw.open(mapInstance, marker);
      });

      markers.push(marker);
      infoWindows.push(iw);

      if (!legendItems.find((l) => l.scope === scope)) {
        legendItems.push({ scope, color });
      }
    });

    // Update legend
    if (legendItems.length) {
      mapLegend.innerHTML = legendItems.map((l) =>
        `<div class="legend-item">
          <div class="legend-dot" style="background:${l.color}"></div>
          <span>${capitalize(l.scope)}</span>
        </div>`
      ).join("");
    }

    // Fit map to markers
    if (!bounds.isEmpty()) {
      mapInstance.fitBounds(bounds, { padding: 60 });
    }
  }

  function clearFeatureMarkers() {
    // Remove all markers except those with title "Selected location"
    markers = markers.filter((m) => {
      if (m.title !== "Selected location") {
        m.setMap(null);
        return false;
      }
      return true;
    });
    infoWindows.forEach((w) => w.close());
    infoWindows = [];
  }

  function clearMarkers() {
    markers.forEach((m) => m.setMap(null));
    markers = [];
    infoWindows.forEach((w) => w.close());
    infoWindows = [];
    mapLegend.innerHTML = "";
    selectedLat = null;
    selectedLng = null;
    coordBadge.hidden = true;
  }

  // ── Memory panel ─────────────────────────────────────────────────────────
  document.getElementById("btnRefreshMem").addEventListener("click", fetchMemoryStats);
  document.getElementById("btnPruneMem").addEventListener("click", async () => {
    const r = await fetch("/api/memory?ttl_days=30", { method: "DELETE" });
    const d = await r.json();
    alert(`Pruned ${d.pruned} expired memory records.`);
    fetchMemoryStats();
  });

  async function fetchMemoryStats() {
    try {
      const r = await fetch(`/api/memory?session_id=${sessionId}`);
      const d = await r.json();

      document.getElementById("memTable").textContent = d.table  || "—";
      document.getElementById("memRows").textContent  = d.rows   ?? "—";
      document.getElementById("memDim").textContent   = d.emb_dim || "—";

      const recordsEl = document.getElementById("memoryRecords");
      recordsEl.innerHTML = "";

      (d.session_records || []).forEach((rec) => {
        const el = document.createElement("div");
        el.className = "memory-record";
        el.innerHTML = `
          <div class="memory-record-scope">${rec.scope}</div>
          <div class="memory-record-content">${escapeHtml(rec.content)}</div>
          <div class="memory-record-importance">
            <div class="memory-record-importance-fill"
                 style="width:${Math.round(rec.importance * 100)}%"></div>
          </div>
        `;
        recordsEl.appendChild(el);
      });

      if (!d.session_records || !d.session_records.length) {
        recordsEl.innerHTML =
          `<p style="color:var(--text-dim);font-size:0.8rem;text-align:center;padding:24px 0;">
            No session memories yet. Start a conversation.
          </p>`;
      }
    } catch (e) {
      console.warn("Memory fetch failed:", e);
    }
  }

  // ── Health check on load ─────────────────────────────────────────────────
  (async () => {
    try {
      const r = await fetch("/health");
      if (r.ok) setStatus("ok", "Connected");
      else      setStatus("err", "Degraded");
    } catch {
      setStatus("err", "Offline");
    }
  })();

  // ── Utilities ─────────────────────────────────────────────────────────────
  function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : ""; }

  // ── Dark Google Maps style ────────────────────────────────────────────────
  const DARK_MAP_STYLE = [
    { elementType: "geometry",       stylers: [{ color: "#1a1410" }] },
    { elementType: "labels.text.stroke", stylers: [{ color: "#0d0b0a" }] },
    { elementType: "labels.text.fill",   stylers: [{ color: "#6a5a4a" }] },
    { featureType: "administrative",
      elementType: "geometry.stroke",  stylers: [{ color: "#332820" }] },
    { featureType: "administrative.land_parcel",
      elementType: "labels.text.fill", stylers: [{ color: "#4a3a2a" }] },
    { featureType: "landscape.natural",
      elementType: "geometry",          stylers: [{ color: "#16120e" }] },
    { featureType: "poi",
      elementType: "geometry",          stylers: [{ color: "#1e1812" }] },
    { featureType: "poi",
      elementType: "labels.text.fill",  stylers: [{ color: "#5a4a38" }] },
    { featureType: "poi.park",
      elementType: "geometry.fill",     stylers: [{ color: "#1a1e14" }] },
    { featureType: "road",
      elementType: "geometry",          stylers: [{ color: "#2a2018" }] },
    { featureType: "road",
      elementType: "labels.text.fill",  stylers: [{ color: "#50402e" }] },
    { featureType: "road.highway",
      elementType: "geometry",          stylers: [{ color: "#382a1c" }] },
    { featureType: "road.highway",
      elementType: "geometry.stroke",   stylers: [{ color: "#1e1610" }] },
    { featureType: "transit",
      elementType: "geometry",          stylers: [{ color: "#1a1410" }] },
    { featureType: "water",
      elementType: "geometry.fill",     stylers: [{ color: "#0c1018" }] },
    { featureType: "water",
      elementType: "labels.text.fill",  stylers: [{ color: "#2a3848" }] },
  ];

})();
