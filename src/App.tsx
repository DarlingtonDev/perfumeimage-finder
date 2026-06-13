import { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";
import { createWorker, type Worker } from "tesseract.js";
import appSource from "./App.tsx?raw";
import mainSource from "./main.tsx?raw";
import cssSource from "./index.css?raw";
import cnSource from "./utils/cn.ts?raw";
import indexHtmlSource from "../index.html?raw";
import packageJsonSource from "../package.json?raw";
import tsconfigSource from "../tsconfig.json?raw";
import viteConfigSource from "../vite.config.ts?raw";
import workerSource from "../serp_proxy_worker.js?raw";

type SearchImageHit = {
  position: number;
  title?: string;
  source?: string;
  link?: string;
  original?: string;
  thumbnail?: string;
  imageWidth?: number;
  imageHeight?: number;
};

type CandidateOption = {
  id: string;
  imageUrl: string;
  title: string;
  domain: string;
  score: number;
  width?: number;
  height?: number;
  usesThumbnail: boolean;
};

type ProcessStatus =
  | "pending"
  | "searching"
  | "verifying"
  | "awaiting_pick"
  | "processing"
  | "done"
  | "failed";

type PerfumeItem = {
  id: string;
  rawLine: string;
  cleanName: string;
  status: ProcessStatus;
  note: string;
  candidates?: CandidateOption[];
  selectedCandidateId?: string;
  previewUrl?: string;
  sourceText?: string;
  sourceDomain?: string;
  score?: number;
};

const STOP_WORDS = new Set([
  "perfume",
  "oil",
  "eau",
  "de",
  "ml",
  "for",
  "men",
  "women",
  "unisex",
  "by",
  "spray",
  "parfum",
  "edp",
  "edt",
]);

const DEFAULT_PRIMARY_SITE = "https://fragrances.com.ng/";

const WORKER_SCRIPT = `export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET,OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    const q = url.searchParams.get("q") || "";
    const engine = url.searchParams.get("engine") || "google_images";
    const ijn = url.searchParams.get("ijn") || "0";
    const num = url.searchParams.get("num") || "20";

    if (!q.trim()) {
      return new Response(JSON.stringify({ error: "Missing q parameter" }), {
        status: 400,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    const apiKey = env.SERPAPI_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "Missing SERPAPI_KEY secret" }), {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    const serpUrl = new URL("https://serpapi.com/search.json");
    serpUrl.searchParams.set("engine", engine);
    serpUrl.searchParams.set("q", q);
    serpUrl.searchParams.set("ijn", ijn);
    serpUrl.searchParams.set("num", num);
    serpUrl.searchParams.set("api_key", apiKey);

    try {
      const serpResponse = await fetch(serpUrl.toString());
      const text = await serpResponse.text();

      return new Response(text, {
        status: serpResponse.status,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error) {
      return new Response(
        JSON.stringify({ error: error instanceof Error ? error.message : "Proxy failed" }),
        {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        }
      );
    }
  },
};`;

const normalize = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const sanitizePerfumeLine = (line: string) =>
  line
    .replace(/\s*[-–—]\s*[\d.,]+\s*[a-zA-Z]{0,6}\s*$/g, "")
    .replace(/\s+/g, " ")
    .trim();

const tokeniseForMatching = (name: string) =>
  normalize(name)
    .split(" ")
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token) && !/^\d+$/.test(token));

const extractHost = (value?: string) => {
  if (!value) return "";
  try {
    return new URL(value).hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return normalize(value);
  }
};

const inferImageExtension = (mimeType: string) => {
  if (mimeType.includes("png")) return "png";
  if (mimeType.includes("webp")) return "webp";
  return "jpg";
};

const OUTPUT_SIZE = 500;
const MAX_WORK_DIM = 1600;

const loadImageFromBlob = (blob: Blob) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not load image"));
    };
    image.src = objectUrl;
  });

const colorDistance = (
  r1: number,
  g1: number,
  b1: number,
  r2: number,
  g2: number,
  b2: number
) => {
  const dr = r1 - r2;
  const dg = g1 - g2;
  const db = b1 - b2;
  return Math.sqrt(dr * dr + dg * dg + db * db);
};

// Algorithmic background removal: flood-fill from the image borders, clearing
// every pixel that is connected to an edge and close in color to the border color.
// This isolates the product without any AI model. Works best on solid/soft backgrounds.
const removeBackgroundByFloodFill = (
  data: Uint8ClampedArray,
  width: number,
  height: number,
  tolerance: number
) => {
  // Estimate the background color from the four corners.
  const corners = [
    0,
    (width - 1) * 4,
    (height - 1) * width * 4,
    ((height - 1) * width + (width - 1)) * 4,
  ];
  let br = 0;
  let bg = 0;
  let bb = 0;
  for (const c of corners) {
    br += data[c];
    bg += data[c + 1];
    bb += data[c + 2];
  }
  br /= corners.length;
  bg /= corners.length;
  bb /= corners.length;

  const visited = new Uint8Array(width * height);
  const stack: number[] = [];

  const pushIfEdge = (x: number, y: number) => {
    stack.push(y * width + x);
  };

  for (let x = 0; x < width; x++) {
    pushIfEdge(x, 0);
    pushIfEdge(x, height - 1);
  }
  for (let y = 0; y < height; y++) {
    pushIfEdge(0, y);
    pushIfEdge(width - 1, y);
  }

  while (stack.length) {
    const idx = stack.pop()!;
    if (visited[idx]) continue;
    visited[idx] = 1;

    const p = idx * 4;
    const dist = colorDistance(data[p], data[p + 1], data[p + 2], br, bg, bb);
    if (dist > tolerance) continue;

    // Mark this background pixel transparent.
    data[p + 3] = 0;

    const x = idx % width;
    const y = (idx - x) / width;
    if (x > 0) stack.push(idx - 1);
    if (x < width - 1) stack.push(idx + 1);
    if (y > 0) stack.push(idx - width);
    if (y < height - 1) stack.push(idx + width);
  }
};

const countTransparentEdgePixels = (data: Uint8ClampedArray, width: number, height: number) => {
  let transparent = 0;
  let total = 0;

  for (let x = 0; x < width; x++) {
    total += 2;
    if (data[(x * 4) + 3] < 8) transparent++;
    if (data[((height - 1) * width + x) * 4 + 3] < 8) transparent++;
  }
  for (let y = 1; y < height - 1; y++) {
    total += 2;
    if (data[(y * width) * 4 + 3] < 8) transparent++;
    if (data[(y * width + (width - 1)) * 4 + 3] < 8) transparent++;
  }

  return total ? transparent / total : 0;
};

// Keep only the largest opaque connected component to remove background labels/noise.
const keepLargestOpaqueComponent = (data: Uint8ClampedArray, width: number, height: number) => {
  const visited = new Uint8Array(width * height);
  let bestPixels: number[] = [];

  for (let i = 0; i < width * height; i++) {
    if (visited[i]) continue;
    if (data[i * 4 + 3] <= 10) {
      visited[i] = 1;
      continue;
    }

    const stack = [i];
    const component: number[] = [];
    visited[i] = 1;

    while (stack.length) {
      const idx = stack.pop()!;
      component.push(idx);

      const x = idx % width;
      const y = (idx - x) / width;
      const neighbors = [
        x > 0 ? idx - 1 : -1,
        x < width - 1 ? idx + 1 : -1,
        y > 0 ? idx - width : -1,
        y < height - 1 ? idx + width : -1,
      ];

      for (const n of neighbors) {
        if (n < 0 || visited[n]) continue;
        visited[n] = 1;
        if (data[n * 4 + 3] > 10) stack.push(n);
      }
    }

    if (component.length > bestPixels.length) {
      bestPixels = component;
    }
  }

  if (!bestPixels.length) return null;
  const keep = new Uint8Array(width * height);
  for (const idx of bestPixels) keep[idx] = 1;

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;

  for (let i = 0; i < width * height; i++) {
    if (!keep[i]) {
      data[i * 4 + 3] = 0;
      continue;
    }

    const x = i % width;
    const y = (i - x) / width;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  return { minX, minY, maxX, maxY };
};

// Full pipeline: download -> optional flood-fill bg removal -> crop subject ->
// center on a white 500x500 canvas -> export JPEG.
const processToWhite500 = async (sourceBlob: Blob, removeBg: boolean) => {
  const image = await loadImageFromBlob(sourceBlob);

  // Work canvas (scaled down for performance on phones).
  const scaleToWork = Math.min(1, MAX_WORK_DIM / Math.max(image.width, image.height));
  const workW = Math.max(1, Math.round(image.width * scaleToWork));
  const workH = Math.max(1, Math.round(image.height * scaleToWork));

  const work = document.createElement("canvas");
  work.width = workW;
  work.height = workH;
  const workCtx = work.getContext("2d", { willReadFrequently: true });
  if (!workCtx) throw new Error("Canvas not supported");
  workCtx.imageSmoothingEnabled = true;
  workCtx.imageSmoothingQuality = "high";
  workCtx.drawImage(image, 0, 0, workW, workH);

  let bounds = { minX: 0, minY: 0, maxX: workW - 1, maxY: workH - 1 };

  if (removeBg) {
    const imageData = workCtx.getImageData(0, 0, workW, workH);
    const original = new Uint8ClampedArray(imageData.data);

    // Adaptive pass: increase tolerance until we clear most border pixels.
    let chosenData = imageData.data;
    for (const tolerance of [34, 42, 52, 64]) {
      const trial = new Uint8ClampedArray(original);
      removeBackgroundByFloodFill(trial, workW, workH, tolerance);
      if (countTransparentEdgePixels(trial, workW, workH) > 0.7) {
        chosenData = trial;
        break;
      }
      chosenData = trial;
    }

    imageData.data.set(chosenData);
    const subject = keepLargestOpaqueComponent(imageData.data, workW, workH);
    workCtx.putImageData(imageData, 0, 0);
    if (subject) bounds = subject;
  }

  const subjectW = bounds.maxX - bounds.minX + 1;
  const subjectH = bounds.maxY - bounds.minY + 1;

  // Output canvas: white background, subject centered with padding.
  const out = document.createElement("canvas");
  out.width = OUTPUT_SIZE;
  out.height = OUTPUT_SIZE;
  const outCtx = out.getContext("2d");
  if (!outCtx) throw new Error("Canvas not supported");
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = "high";
  outCtx.fillStyle = "#ffffff";
  outCtx.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

  const padding = 36;
  const maxBox = OUTPUT_SIZE - padding * 2;
  const scale = Math.min(maxBox / subjectW, maxBox / subjectH);
  const drawW = subjectW * scale;
  const drawH = subjectH * scale;
  const dx = (OUTPUT_SIZE - drawW) / 2;
  const dy = (OUTPUT_SIZE - drawH) / 2;

  outCtx.drawImage(work, bounds.minX, bounds.minY, subjectW, subjectH, dx, dy, drawW, drawH);

  return new Promise<Blob>((resolve, reject) => {
    out.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not export image"));
    }, "image/png");
  });
};

const PROXY_BUILDERS = [
  (url: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  (url: string) => `https://corsproxy.io/?${encodeURIComponent(url)}`,
];

const fetchWithTimeout = async (url: string, timeoutMs = 20000) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
};

const fetchWithProxyFallback = async (url: string) => {
  const errors: string[] = [];

  try {
    const directResponse = await fetchWithTimeout(url);
    if (directResponse.ok) return directResponse;
    errors.push(`direct:${directResponse.status}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : "direct failed";
    errors.push(`direct:${message}`);
  }

  for (const buildProxyUrl of PROXY_BUILDERS) {
    try {
      const proxyResponse = await fetchWithTimeout(buildProxyUrl(url));
      if (proxyResponse.ok) return proxyResponse;
      errors.push(`proxy:${proxyResponse.status}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "proxy failed";
      errors.push(`proxy:${message}`);
    }
  }

  throw new Error(`Load failed (${errors.join(" | ")})`);
};

const scoreCandidate = (hit: SearchImageHit, tokens: string[], preferredSite: string) => {
  const host = extractHost(hit.link || hit.source);
  const joinedText = normalize(`${hit.title ?? ""} ${hit.source ?? ""} ${hit.link ?? ""}`);
  const matchedTokens = tokens.filter((token) => joinedText.includes(token)).length;
  const tokenScore = tokens.length ? (matchedTokens / tokens.length) * 65 : 0;
  const perfumeBonus = joinedText.includes("perfume") || joinedText.includes("fragrance") ? 15 : 0;
  const bottleBonus = joinedText.includes("bottle") ? 8 : 0;
  const brandedBonus = joinedText.includes("ml") ? 4 : 0;
  const preferredDomain = extractHost(preferredSite);
  const preferredSiteBonus = preferredDomain && host.includes(preferredDomain) ? 26 : 0;
  const originalImageBonus = hit.original ? 10 : -8;
  const resolutionBonus =
    (hit.imageWidth ?? 0) >= 700 && (hit.imageHeight ?? 0) >= 700
      ? 6
      : (hit.imageWidth ?? 0) >= 400
        ? 3
        : 0;
  return (
    tokenScore +
    perfumeBonus +
    bottleBonus +
    brandedBonus +
    preferredSiteBonus +
    originalImageBonus +
    resolutionBonus
  );
};

export default function App() {
  const [serpApiKey, setSerpApiKey] = useState("");
  const [proxyUrl, setProxyUrl] = useState("");
  const [rawList, setRawList] = useState(
    "Ahmed Al Maghribi Rose Noir 65k"
  );
  const [items, setItems] = useState<PerfumeItem[]>([]);
  const [ocrEnabled, setOcrEnabled] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [runLog, setRunLog] = useState<string[]>([]);
  const [zipBusy, setZipBusy] = useState(false);
  const [sourceZipBusy, setSourceZipBusy] = useState(false);
  const [copiedWorker, setCopiedWorker] = useState(false);
  const [removeBg, setRemoveBg] = useState(true);
  const [primarySite, setPrimarySite] = useState(DEFAULT_PRIMARY_SITE);

  const workerRef = useRef<Worker | null>(null);
  const processedBlobsRef = useRef<Map<string, Blob>>(new Map());

  useEffect(() => {
    return () => {
      if (workerRef.current) {
        void workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  const successfulItems = useMemo(
    () => items.filter((item) => item.status === "done" && item.previewUrl),
    [items]
  );

  const itemsWithSelection = useMemo(
    () => items.filter((item) => item.selectedCandidateId),
    [items]
  );

  const upsertItem = (id: string, patch: Partial<PerfumeItem>) => {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  };

  const appendLog = (message: string) => {
    setRunLog((current) => [message, ...current].slice(0, 120));
  };

  const selectCandidate = (itemId: string, candidateId: string) => {
    setItems((current) =>
      current.map((item) => {
        if (item.id !== itemId) return item;
        const nextId = item.selectedCandidateId === candidateId ? undefined : candidateId;
        return {
          ...item,
          selectedCandidateId: nextId,
          note: nextId ? "Image selected. Ready to process." : "No image selected yet.",
        };
      })
    );
  };

  const getWorker = async () => {
    if (!workerRef.current) {
      workerRef.current = await createWorker("eng");
    }
    return workerRef.current;
  };

  const runOcrMatchScore = async (imageUrl: string, targetName: string) => {
    const worker = await getWorker();
    const response = await fetchWithProxyFallback(imageUrl);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    let result;

    try {
      result = await worker.recognize(objectUrl);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }

    const foundText = normalize(result.data.text);
    const keyTokens = tokeniseForMatching(targetName).slice(0, 4);
    const hits = keyTokens.filter((token) => foundText.includes(token)).length;
    return {
      hits,
      tokenCount: keyTokens.length,
      score: keyTokens.length ? (hits / keyTokens.length) * 35 : 0,
    };
  };

  const fetchInternetCandidates = async (query: string) => {
    const apiParam = serpApiKey.trim() ? `&api_key=${encodeURIComponent(serpApiKey)}` : "";
    const url = proxyUrl.trim()
      ? `${proxyUrl.trim()}?engine=google_images&q=${encodeURIComponent(query)}&ijn=0&num=30${apiParam}`
      : `https://serpapi.com/search.json?engine=google_images&q=${encodeURIComponent(
          query
        )}&ijn=0&num=30${apiParam}`;
    const response = await fetchWithProxyFallback(url);

    if (!response.ok) {
      throw new Error(`Internet search request failed (${response.status})`);
    }

    let data: { images_results?: SearchImageHit[]; error?: string };

    try {
      data = (await response.json()) as { images_results?: SearchImageHit[]; error?: string };
    } catch {
      throw new Error("Search response could not be parsed. Try another network or disable VPN.");
    }

    if (data.error) {
      throw new Error(data.error);
    }

    const results = data.images_results ?? [];

    if (!results.length && !proxyUrl.trim()) {
      throw new Error(
        "No results returned. SerpAPI browser calls can fail due to CORS. Set a proxy endpoint URL."
      );
    }

    return results.filter((hit) => hit.original || hit.thumbnail);
  };

  const processList = async () => {
    const parsed = rawList
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((rawLine, index) => ({
        id: `${Date.now()}-${index}`,
        rawLine,
        cleanName: sanitizePerfumeLine(rawLine),
      }));

    if (!proxyUrl.trim() && !serpApiKey.trim()) {
      appendLog("Missing SerpAPI key. If using proxy with server secret, just set proxy URL.");
      return;
    }

    if (!parsed.length) {
      appendLog("List is empty. Add at least one perfume line.");
      return;
    }

    setRunLog([]);
    processedBlobsRef.current.clear();
    setItems(
      parsed.map((entry) => ({
        ...entry,
        status: "pending",
        note: "Waiting",
      }))
    );

    setIsProcessing(true);

    for (const item of parsed) {
      try {
        upsertItem(item.id, { status: "searching", note: "Searching image candidates" });
        appendLog(`Searching: ${item.cleanName}`);

        const genericHits = await fetchInternetCandidates(`${item.cleanName} perfume bottle`);
        const preferredHost = extractHost(primarySite);
        const priorityQuery = preferredHost ? `${item.cleanName} perfume bottle site:${preferredHost}` : "";
        const priorityHits = priorityQuery ? await fetchInternetCandidates(priorityQuery) : [];

        const mergedByUrl = new Map<string, SearchImageHit>();
        for (const hit of [...priorityHits, ...genericHits]) {
          const key = hit.original ?? hit.thumbnail;
          if (!key) continue;
          if (!mergedByUrl.has(key)) mergedByUrl.set(key, hit);
        }

        const hits = [...mergedByUrl.values()];

        if (!hits.length) {
          upsertItem(item.id, { status: "failed", note: "No image found" });
          appendLog(`No candidates: ${item.cleanName}`);
          continue;
        }

        const tokens = tokeniseForMatching(item.cleanName);
        const preferredCount = preferredHost
          ? hits.filter((hit) => extractHost(hit.link || hit.source).includes(preferredHost)).length
          : 0;
        appendLog(`Candidates: ${item.cleanName} -> ${preferredCount} preferred-site / ${hits.length} total`);

        const ranked = hits
          .map((hit) => ({ hit, score: scoreCandidate(hit, tokens, primarySite) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 8);

        if (ocrEnabled) {
          upsertItem(item.id, { status: "verifying", note: "Running OCR verification" });

          for (const candidate of ranked) {
            const candidateUrl = candidate.hit.original ?? candidate.hit.thumbnail;
            if (!candidateUrl) continue;

            try {
              const ocr = await runOcrMatchScore(candidateUrl, item.cleanName);
              candidate.score += ocr.score;
            } catch (error) {
              const message = error instanceof Error ? error.message : "OCR failed";
              appendLog(`OCR skipped: ${item.cleanName} (${message})`);
              candidate.score -= 8;
            }
          }
        }

        ranked.sort((a, b) => b.score - a.score);

        const shortlist = ranked
          .slice(0, 6)
          .map((entry, index): CandidateOption | null => {
            const usesThumbnail = !entry.hit.original && Boolean(entry.hit.thumbnail);
            const imageUrl = entry.hit.original ?? entry.hit.thumbnail;
            if (!imageUrl) return null;

            // Skip very small thumbnail-only images that will look blurry after 500x500 processing.
            if (usesThumbnail && (entry.hit.imageWidth ?? 0) > 0 && (entry.hit.imageWidth ?? 0) < 420) {
              return null;
            }

            return {
              id: `${item.id}-opt-${index}`,
              imageUrl,
              title: entry.hit.title ?? "Untitled",
              domain: extractHost(entry.hit.link || entry.hit.source) || "unknown",
              score: Math.max(0, Math.min(100, Math.round(entry.score))),
              width: entry.hit.imageWidth,
              height: entry.hit.imageHeight,
              usesThumbnail,
            };
          })
          .filter((entry): entry is CandidateOption => Boolean(entry));

        if (!shortlist.length) {
          upsertItem(item.id, { status: "failed", note: "No usable image candidate" });
          appendLog(`No usable candidates: ${item.cleanName}`);
          continue;
        }

        upsertItem(item.id, {
          status: "awaiting_pick",
          note: "Pick the best image below",
          candidates: shortlist,
          selectedCandidateId: undefined,
          previewUrl: undefined,
          sourceText: undefined,
          sourceDomain: undefined,
          score: undefined,
        });
        appendLog(`Ready for selection: ${item.cleanName} (${shortlist.length} options)`);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        upsertItem(item.id, { status: "failed", note: message });
        appendLog(`Failed: ${item.cleanName} -> ${message}`);
      }
    }

    setIsProcessing(false);
  };

  const processSelectedImages = async () => {
    const selected = items.filter((item) => item.selectedCandidateId);

    if (!selected.length) {
      appendLog("No selected images yet. Pick candidates first.");
      return;
    }

    setIsProcessing(true);

    for (const item of selected) {
      const candidate = item.candidates?.find((opt) => opt.id === item.selectedCandidateId);

      if (!candidate) {
        upsertItem(item.id, { status: "failed", note: "Selected image not found" });
        continue;
      }

      upsertItem(item.id, {
        status: "processing",
        note: removeBg ? "Removing background + resizing 500x500" : "Resizing 500x500",
      });
      appendLog(`Processing selected: ${item.cleanName}`);

      try {
        const imageResponse = await fetchWithProxyFallback(candidate.imageUrl);
        const rawBlob = await imageResponse.blob();
        const processedBlob = await processToWhite500(rawBlob, removeBg);

        processedBlobsRef.current.set(item.id, processedBlob);
        const localPreview = URL.createObjectURL(processedBlob);

        upsertItem(item.id, {
          status: "done",
          note: "Ready for ZIP",
          previewUrl: localPreview,
          sourceText: candidate.title,
          sourceDomain: candidate.domain,
          score: candidate.score,
        });
        appendLog(`Ready: ${item.cleanName}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Processing failed";
        upsertItem(item.id, { status: "failed", note: `Image processing failed: ${message}` });
        appendLog(`Processing failed: ${item.cleanName} -> ${message}`);
      }
    }

    setIsProcessing(false);
  };

  const downloadZip = async () => {
    if (!successfulItems.length) {
      appendLog("No successful images to zip.");
      return;
    }

    setZipBusy(true);

    try {
      const zip = new JSZip();

      for (const item of successfulItems) {
        const blob = processedBlobsRef.current.get(item.id);
        if (!blob) continue;
        const ext = inferImageExtension(blob.type);
        zip.file(`${item.cleanName}.${ext}`, blob);
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "perfume_images.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      appendLog("ZIP download started.");
    } catch {
      appendLog("Failed to build ZIP file.");
    } finally {
      setZipBusy(false);
    }
  };

  const downloadSourceZip = async () => {
    setSourceZipBusy(true);

    try {
      const zip = new JSZip();

      const sourceFiles: Record<string, string> = {
        "index.html": indexHtmlSource,
        "package.json": packageJsonSource,
        "tsconfig.json": tsconfigSource,
        "vite.config.ts": viteConfigSource,
        "serp_proxy_worker.js": workerSource,
        "src/App.tsx": appSource,
        "src/main.tsx": mainSource,
        "src/index.css": cssSource,
        "src/utils/cn.ts": cnSource,
      };

      for (const [filePath, content] of Object.entries(sourceFiles)) {
        zip.file(filePath, content);
      }

      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "perfumeimage-finder-source.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      appendLog("Source ZIP download started.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown export error";
      appendLog(`Source ZIP failed: ${message}`);
    } finally {
      setSourceZipBusy(false);
    }
  };

  const copyWorkerScript = async () => {
    try {
      await navigator.clipboard.writeText(WORKER_SCRIPT);
      setCopiedWorker(true);
      appendLog("Worker script copied. Paste it in Cloudflare Worker editor.");
      setTimeout(() => setCopiedWorker(false), 1800);
    } catch {
      appendLog("Could not copy script automatically. Copy it manually from the box.");
    }
  };

  return (
    <main className="min-h-screen bg-zinc-100 text-zinc-900">
      <section className="mx-auto w-full max-w-6xl p-4 sm:p-6 lg:p-8">
        <h1 className="text-2xl font-semibold tracking-tight">Perfume Image Verifier + Downloader</h1>
        <p className="mt-2 max-w-3xl text-sm text-zinc-600">
          Paste your perfume list, remove the price automatically, search the open internet with Google Images
          results through SerpAPI, verify candidates with algorithmic scoring (plus OCR if enabled), then
          download a ZIP where each file name equals the perfume name.
        </p>

        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <label className="text-sm font-medium text-zinc-700">
            SerpAPI Key (Google Images)
            <input
              value={serpApiKey}
              onChange={(event) => setSerpApiKey(event.target.value)}
              placeholder="Paste your SerpAPI key"
              className="mt-2 w-full border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900"
            />
          </label>

          <label className="text-sm font-medium text-zinc-700">
            Proxy Endpoint URL (recommended)
            <input
              value={proxyUrl}
              onChange={(event) => setProxyUrl(event.target.value)}
              placeholder="https://your-proxy-domain/search"
              className="mt-2 w-full border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900"
            />
          </label>

          <label className="text-sm font-medium text-zinc-700">
            Preferred Site (checked first)
            <input
              value={primarySite}
              onChange={(event) => setPrimarySite(event.target.value)}
              placeholder="example: fragrantica.com"
              className="mt-2 w-full border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900"
            />
            <div className="mt-1 text-xs font-normal text-zinc-600">
              Optional. We run a site-restricted search first, then merge with normal SerpAPI image results.
            </div>
          </label>

          <label className="text-sm font-medium text-zinc-700">
            Verification Mode
            <div className="mt-2 flex items-center gap-3 border border-zinc-300 bg-white px-3 py-2 text-sm">
              <input
                id="ocr-toggle"
                type="checkbox"
                checked={ocrEnabled}
                onChange={(event) => setOcrEnabled(event.target.checked)}
                className="h-4 w-4"
              />
              <span>
                OCR enabled (slower and can fail on some phones; keep off first, then enable if needed)
              </span>
            </div>
          </label>

          <label className="text-sm font-medium text-zinc-700">
            Output Image Style
            <div className="mt-2 flex items-center gap-3 border border-zinc-300 bg-white px-3 py-2 text-sm">
              <input
                id="bg-toggle"
                type="checkbox"
                checked={removeBg}
                onChange={(event) => setRemoveBg(event.target.checked)}
                className="h-4 w-4"
              />
              <span>
                Remove background + white 500x500 (fast algorithm, best on plain/soft backgrounds)
              </span>
            </div>
          </label>
        </div>

        <p className="mt-2 text-xs text-zinc-600">
          SerpAPI does not reliably support direct browser calls from mobile. If you still see Load failed,
          use a small backend proxy URL and keep your SerpAPI key server-side.
        </p>

        <section className="mt-4 border border-zinc-300 bg-white p-3">
          <h2 className="text-sm font-semibold">No File Explorer? Copy Worker Script Here</h2>
          <p className="mt-1 text-xs text-zinc-600">
            Use this exact script in Cloudflare Workers. Then put your worker URL in Proxy Endpoint URL.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={copyWorkerScript}
              className="border border-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-900"
            >
              {copiedWorker ? "Copied" : "Copy Worker Script"}
            </button>
            <a
              href="https://dash.cloudflare.com/"
              target="_blank"
              rel="noreferrer"
              className="border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700"
            >
              Open Cloudflare Dashboard
            </a>
          </div>
          <textarea
            readOnly
            value={WORKER_SCRIPT}
            rows={16}
            className="mt-3 w-full border border-zinc-300 bg-zinc-50 px-3 py-2 font-mono text-xs outline-none"
          />
        </section>

        <label className="mt-4 block text-sm font-medium text-zinc-700">
          Perfume List (one line per perfume)
          <textarea
            value={rawList}
            onChange={(event) => setRawList(event.target.value)}
            rows={9}
            className="mt-2 w-full border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900"
          />
        </label>

        <div className="mt-4 flex flex-wrap gap-3">
          <button
            onClick={processList}
            disabled={isProcessing}
            className="bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-500"
          >
            {isProcessing ? "Processing..." : "Search Candidates"}
          </button>
          <button
            onClick={processSelectedImages}
            disabled={isProcessing || !itemsWithSelection.length}
            className="border border-zinc-900 px-4 py-2 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:border-zinc-400 disabled:text-zinc-400"
          >
            Process Selected ({itemsWithSelection.length})
          </button>
          <button
            onClick={downloadZip}
            disabled={zipBusy || !successfulItems.length}
            className="border border-zinc-900 px-4 py-2 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:border-zinc-400 disabled:text-zinc-400"
          >
            {zipBusy ? "Preparing ZIP..." : "Download ZIP"}
          </button>
          <button
            onClick={downloadSourceZip}
            disabled={sourceZipBusy}
            className="border border-zinc-900 px-4 py-2 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:border-zinc-400 disabled:text-zinc-400"
          >
            {sourceZipBusy ? "Preparing Source..." : "Download Source ZIP"}
          </button>
        </div>

        <p className="mt-2 text-xs text-zinc-600">
          Flow: search candidates, pick one image per perfume, process selected, then download ZIP.
        </p>

        <section className="mt-8 space-y-4">
          {items.map((item) => (
            <article key={item.id} className="border border-zinc-300 bg-white p-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold">{item.cleanName}</h3>
                  <p className="text-xs text-zinc-500">{item.note}</p>
                </div>
                <div className="text-xs text-zinc-600">
                  {item.score ? `${item.score}%` : "-"} {item.sourceDomain ? `| ${item.sourceDomain}` : ""}
                </div>
              </div>

              {item.candidates?.length ? (
                <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                  {item.candidates.map((candidate) => {
                    const checked = item.selectedCandidateId === candidate.id;
                    return (
                      <button
                        type="button"
                        key={candidate.id}
                        onClick={() => selectCandidate(item.id, candidate.id)}
                        className={`border p-1 text-left ${checked ? "border-zinc-900" : "border-zinc-300"}`}
                      >
                        <img
                          src={candidate.imageUrl}
                          alt={candidate.title}
                          className="h-28 w-full bg-zinc-100 object-cover"
                        />
                        <div className="mt-1 text-[11px] font-medium text-zinc-800">
                          {checked ? "Selected" : "Tap to select"} - {candidate.score}%
                        </div>
                        <div className="line-clamp-2 text-[11px] text-zinc-500">
                          {candidate.domain} {candidate.width ? `| ${candidate.width}x${candidate.height ?? "?"}` : ""}
                        </div>
                        {candidate.usesThumbnail ? (
                          <div className="text-[10px] text-amber-700">thumbnail source (lower quality)</div>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              ) : null}

              {item.previewUrl ? (
                <div className="mt-3">
                  <div className="mb-1 text-xs font-medium text-zinc-700">Processed 500x500 output</div>
                  <img src={item.previewUrl} alt={item.cleanName} className="h-36 w-36 border border-zinc-300 bg-white object-contain" />
                </div>
              ) : null}
            </article>
          ))}
        </section>

        <section className="mt-4 border border-zinc-300 bg-white p-3">
          <h2 className="text-sm font-semibold">Run Log</h2>
          <div className="mt-2 max-h-48 overflow-y-auto text-xs text-zinc-700">
            {runLog.length ? (
              runLog.map((line, index) => (
                <div key={`${line}-${index}`} className="border-b border-zinc-100 py-1">
                  {line}
                </div>
              ))
            ) : (
              <div className="text-zinc-500">No run yet.</div>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}
