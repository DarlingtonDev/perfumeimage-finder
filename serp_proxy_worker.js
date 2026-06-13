// Cloudflare Worker proxy for SerpAPI image search.
// Deploy this script on Cloudflare Workers, then use your worker URL in the app.

export default {
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
      return new Response(JSON.stringify({ error: error instanceof Error ? error.message : "Proxy failed" }), {
        status: 502,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }
  },
};