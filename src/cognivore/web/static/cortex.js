// Cognivore knowledge map ("cortex") -- vanilla canvas 2D, no libraries.
//
// Draws the server's /api/insights/map payload: every knowledge-base chunk
// as a point (PCA coordinates in [-1, 1]), each topic as a soft nebula with
// its c-TF-IDF keywords, and each point linked to its nearest neighbour in
// the same topic (the "constellation" lines). When a question is answered,
// the question itself is projected into the same space and animated beams
// run from it to the passages that were retrieved.

const Cortex = (() => {
  const PAD = 30;
  // Topic names drawn on the map itself; the list under the map has them all.
  const MAX_TOPIC_LABELS = 6;
  const reducedMotion =
    typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  let canvas = null;
  let ctx = null;
  let tooltipEl = null;
  let emptyEl = null;
  let dpr = 1;
  let data = { points: [], clusters: [] };
  let byId = new Map();
  let links = []; // [pointIndexA, pointIndexB]
  let spreads = new Map(); // cluster id -> radius in map units
  let colors = { clusters: [] };
  let hoverIndex = -1;
  let focusCluster = null;
  let query = null; // { x, y, hits: [{ id, rank }], t0 }
  let pulseId = null;
  let pulseT0 = 0;
  let rafId = 0;
  let onPointClick = null;
  let labelQuery = "";

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    colors = {
      clusters: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => v(`--c${i}`)),
      text: v("--text"),
      muted: v("--muted"),
      query: v("--query"),
      border: v("--border"),
      surface: v("--surface-2"),
    };
  }

  function clusterColor(id) {
    return colors.clusters[((id % 8) + 8) % 8] || "#5eead4";
  }

  function withAlpha(hex, alpha) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }

  function scale() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    return { w, h, s: Math.max(10, Math.min(w, h) / 2 - PAD) };
  }

  function toPx(x, y) {
    const { w, h, s } = scale();
    return [w / 2 + x * s, h / 2 - y * s];
  }

  function resize() {
    if (!canvas) return;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    requestDraw();
  }

  function precompute() {
    byId = new Map(data.points.map((p, i) => [p.id, i]));
    links = [];
    spreads = new Map();
    const pts = data.points;
    const seen = new Set();
    for (let i = 0; i < pts.length; i++) {
      let best = -1;
      let bestD = Infinity;
      for (let j = 0; j < pts.length; j++) {
        if (i === j || pts[j].cluster !== pts[i].cluster) continue;
        const d = (pts[i].x - pts[j].x) ** 2 + (pts[i].y - pts[j].y) ** 2;
        if (d < bestD) {
          bestD = d;
          best = j;
        }
      }
      if (best < 0) continue;
      const key = i < best ? `${i}-${best}` : `${best}-${i}`;
      if (!seen.has(key)) {
        seen.add(key);
        links.push([i, best]);
      }
    }
    for (const c of data.clusters) {
      const members = pts.filter((p) => p.cluster === c.id);
      const r2 =
        members.reduce((acc, p) => acc + (p.x - c.x) ** 2 + (p.y - c.y) ** 2, 0) /
        Math.max(1, members.length);
      spreads.set(c.id, Math.max(0.12, Math.sqrt(r2) * 1.6));
    }
  }

  function hitRank(id) {
    if (!query) return 0;
    const hit = query.hits.find((h) => h.id === id);
    return hit ? hit.rank : 0;
  }

  let lastFrame = 0;

  function draw(now) {
    rafId = 0;
    if (!ctx) return;
    // The ambient twinkle doesn't need 60 fps; ~30 keeps it cheap on laptops.
    if (!reducedMotion && now - lastFrame < 30 && data.points.length) {
      rafId = requestAnimationFrame(draw);
      return;
    }
    lastFrame = now;
    const { w, h, s } = scale();
    ctx.clearRect(0, 0, w, h);
    const t = reducedMotion ? 0 : now / 1000;

    // Faint polar grid -- gives the map a "scope" feel and a sense of scale.
    ctx.strokeStyle = withAlpha(colors.muted, 0.12);
    ctx.lineWidth = 1;
    for (const r of [0.33, 0.66, 1]) {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, r * s, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Topic nebulae.
    for (const c of data.clusters) {
      const [cx, cy] = toPx(c.x, c.y);
      const r = spreads.get(c.id) * s;
      const dim = focusCluster !== null && focusCluster !== c.id;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, withAlpha(clusterColor(c.id), dim ? 0.05 : 0.2));
      g.addColorStop(1, withAlpha(clusterColor(c.id), 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Constellation lines.
    ctx.lineWidth = 1;
    for (const [a, b] of links) {
      const pa = data.points[a];
      const pb = data.points[b];
      const dim = focusCluster !== null && focusCluster !== pa.cluster;
      ctx.strokeStyle = withAlpha(clusterColor(pa.cluster), dim ? 0.06 : 0.28);
      const [x1, y1] = toPx(pa.x, pa.y);
      const [x2, y2] = toPx(pb.x, pb.y);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Beams from the question to the retrieved passages.
    let animating = false;
    if (query) {
      const [qx, qy] = toPx(query.x, query.y);
      const progress = reducedMotion ? 1 : Math.min(1, (now - query.t0) / 900);
      if (progress < 1) animating = true;
      for (const hit of query.hits) {
        const idx = byId.get(hit.id);
        if (idx === undefined) continue;
        const p = data.points[idx];
        const [px, py] = toPx(p.x, p.y);
        const mx = (qx + px) / 2 + (py - qy) * 0.18;
        const my = (qy + py) / 2 - (px - qx) * 0.18;
        const strength = 1 - (hit.rank - 1) * 0.16;
        ctx.strokeStyle = withAlpha(colors.query, 0.25 + 0.5 * strength);
        ctx.lineWidth = 1 + 1.4 * strength;
        ctx.setLineDash([]);
        ctx.beginPath();
        const steps = 24;
        for (let i = 0; i <= steps * progress; i++) {
          const u = i / steps;
          const bx = (1 - u) ** 2 * qx + 2 * (1 - u) * u * mx + u * u * px;
          const by = (1 - u) ** 2 * qy + 2 * (1 - u) * u * my + u * u * py;
          if (i === 0) ctx.moveTo(bx, by);
          else ctx.lineTo(bx, by);
        }
        ctx.stroke();
        if (!reducedMotion && progress >= 1) {
          // A particle travelling along each beam: "signal flowing".
          const u = (t * 0.45 + hit.rank * 0.21) % 1;
          const bx = (1 - u) ** 2 * qx + 2 * (1 - u) * u * mx + u * u * px;
          const by = (1 - u) ** 2 * qy + 2 * (1 - u) * u * my + u * u * py;
          ctx.fillStyle = colors.query;
          ctx.beginPath();
          ctx.arc(bx, by, 1.8 + strength, 0, Math.PI * 2);
          ctx.fill();
          animating = true;
        }
      }
    }

    // Points.
    data.points.forEach((p, i) => {
      const [x, y] = toPx(p.x, p.y);
      const rank = hitRank(p.id);
      const dim = focusCluster !== null && focusCluster !== p.cluster;
      const twinkle = reducedMotion ? 0 : Math.sin(t * 1.7 + i * 1.3) * 0.5;
      let r = 3 + twinkle * 0.6;
      if (i === hoverIndex) r = 5.5;
      if (rank) r = 6.5 - rank * 0.4;
      ctx.fillStyle = withAlpha(clusterColor(p.cluster), dim ? 0.25 : 0.95);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1.5, r), 0, Math.PI * 2);
      ctx.fill();
      if (rank) {
        ctx.strokeStyle = colors.query;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = colors.text;
        ctx.font = "600 10px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(String(rank), x + r + 5, y - r - 2);
      }
      if (pulseId === p.id && !reducedMotion) {
        const k = ((now - pulseT0) / 1200) % 1;
        ctx.strokeStyle = withAlpha(colors.query, 1 - k);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 6 + k * 18, 0, Math.PI * 2);
        ctx.stroke();
        animating = true;
      }
    });

    // Topic labels (on top of points so they stay readable), largest topics
    // first. Each label tries a few slots alternately above and below its
    // topic and takes the first one that stays inside the canvas and clear
    // of labels already placed; with no free slot it is left out -- the
    // topic list under the map always names every topic.
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const placed = [];
    const queryLabel = query && labelQuery ? queryLabelBox(w, h) : null;
    if (queryLabel) placed.push(queryLabel);
    const ordered = [...data.clusters].sort((a, b) => b.size - a.size);
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    for (const c of ordered) {
      if (!c.keywords.length) continue;
      if (placed.length >= MAX_TOPIC_LABELS + (queryLabel ? 1 : 0)) break;
      const label = c.keywords.slice(0, 2).join(" · ");
      const tw = ctx.measureText(label).width + 12;
      const [cx0, cy] = toPx(c.x, c.y);
      const cx = Math.max(tw / 2 + 4, Math.min(w - tw / 2 - 4, cx0));
      const base = cy - spreads.get(c.id) * s * 0.55;
      let y = null;
      for (const offset of [0, -22, 22, -44, 44, -66, 66]) {
        const candidate = base + offset;
        if (candidate < 13 || candidate > h - 13) continue;
        const clash = placed.some(
          (r) => Math.abs(r.x - cx) < (r.w + tw) / 2 + 4 && Math.abs(r.y - candidate) < 20
        );
        if (!clash) {
          y = candidate;
          break;
        }
      }
      if (y === null) continue;
      placed.push({ x: cx, y, w: tw });
      const dim = focusCluster !== null && focusCluster !== c.id;
      ctx.globalAlpha = dim ? 0.3 : 0.92;
      ctx.fillStyle = colors.surface;
      roundRect(cx - tw / 2, y - 9, tw, 18, 9);
      ctx.fill();
      ctx.strokeStyle = withAlpha(clusterColor(c.id), 0.7);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.fillText(label, cx, y + 0.5);
      ctx.globalAlpha = 1;
    }

    // The question itself.
    if (query) {
      const [qx, qy] = toPx(query.x, query.y);
      const k = reducedMotion ? 0 : (t % 1.6) / 1.6;
      ctx.strokeStyle = withAlpha(colors.query, 0.9 * (1 - k));
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(qx, qy, 7 + k * 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = colors.query;
      ctx.beginPath();
      ctx.moveTo(qx, qy - 7);
      ctx.lineTo(qx + 7, qy);
      ctx.lineTo(qx, qy + 7);
      ctx.lineTo(qx - 7, qy);
      ctx.closePath();
      ctx.fill();
      if (labelQuery) {
        const box = queryLabelBox(w, h);
        ctx.fillStyle = colors.query;
        roundRect(box.x - box.w / 2, box.y - 9, box.w, 18, 9);
        ctx.fill();
        ctx.fillStyle = "#231500";
        ctx.textAlign = "center";
        ctx.fillText(labelQuery, box.x, box.y + 0.5);
      }
    }

    const idle = reducedMotion || document.hidden;
    if (!idle && (animating || data.points.length)) {
      rafId = requestAnimationFrame(draw);
    }
  }

  // Where the "your question" label goes; topic labels are laid out around
  // it so the two never overlap.
  function queryLabelBox(w, h) {
    const [qx, qy] = toPx(query.x, query.y);
    ctx.font = "700 11px Inter, system-ui, sans-serif";
    const tw = ctx.measureText(labelQuery).width + 12;
    return {
      x: Math.max(tw / 2 + 4, Math.min(w - tw / 2 - 4, qx)),
      y: qy + 22 > h - 12 ? qy - 22 : qy + 22,
      w: tw,
    };
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function requestDraw() {
    if (!rafId && ctx) rafId = requestAnimationFrame(draw);
  }

  function nearestPoint(mx, my) {
    let best = -1;
    let bestD = 12 * 12;
    data.points.forEach((p, i) => {
      const [x, y] = toPx(p.x, p.y);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  function onMove(event) {
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const idx = nearestPoint(mx, my);
    if (idx !== hoverIndex) {
      hoverIndex = idx;
      requestDraw();
    }
    if (idx < 0) {
      tooltipEl.hidden = true;
      canvas.style.cursor = "crosshair";
      return;
    }
    canvas.style.cursor = "pointer";
    const p = data.points[idx];
    tooltipEl.replaceChildren();
    const src = document.createElement("div");
    src.className = "tt-src";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = clusterColor(p.cluster);
    src.append(dot, document.createTextNode(p.source));
    const text = document.createElement("div");
    text.textContent = p.preview;
    tooltipEl.append(src, text);
    tooltipEl.hidden = false;
    const wrap = canvas.parentElement.getBoundingClientRect();
    const left = Math.min(mx + 14, wrap.width - tooltipEl.offsetWidth - 8);
    const top = my + 14 + tooltipEl.offsetHeight > wrap.height ? my - tooltipEl.offsetHeight - 10 : my + 14;
    tooltipEl.style.left = `${Math.max(8, left)}px`;
    tooltipEl.style.top = `${Math.max(8, top)}px`;
  }

  function onLeave() {
    hoverIndex = -1;
    tooltipEl.hidden = true;
    requestDraw();
  }

  function onClick(event) {
    const rect = canvas.getBoundingClientRect();
    const idx = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
    if (idx >= 0 && onPointClick) onPointClick(data.points[idx].id);
  }

  return {
    init(options) {
      canvas = options.canvas;
      ctx = canvas.getContext("2d");
      tooltipEl = options.tooltip;
      emptyEl = options.empty;
      onPointClick = options.onPointClick;
      readColors();
      new ResizeObserver(resize).observe(canvas);
      canvas.addEventListener("mousemove", onMove);
      canvas.addEventListener("mouseleave", onLeave);
      canvas.addEventListener("click", onClick);
      document.addEventListener("visibilitychange", requestDraw);
      resize();
    },
    setData(map) {
      data = { points: map.points || [], clusters: map.clusters || [] };
      if (query) query.hits = query.hits.filter((h) => byId.has(h.id));
      precompute();
      emptyEl.hidden = data.points.length > 0;
      requestDraw();
    },
    showQuery(point, hits, label) {
      labelQuery = label || "";
      if (!point) {
        query = null;
      } else {
        const clamp = (v) => Math.max(-1.12, Math.min(1.12, v));
        query = {
          x: clamp(point[0]),
          y: clamp(point[1]),
          hits: hits.map((h) => ({ id: h.id, rank: h.rank })),
          t0: performance.now(),
        };
      }
      requestDraw();
    },
    setQueryLabel(label) {
      labelQuery = label;
      requestDraw();
    },
    focusCluster(id) {
      focusCluster = id;
      requestDraw();
    },
    pulse(id) {
      pulseId = id;
      pulseT0 = performance.now();
      requestDraw();
      setTimeout(() => {
        if (pulseId === id) {
          pulseId = null;
          requestDraw();
        }
      }, 2400);
    },
    colorFor: clusterColor,
    refreshTheme() {
      readColors();
      requestDraw();
    },
  };
})();
