import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  AlertTriangle,
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  ExternalLink,
  Filter,
  Loader2,
  ScrollText,
  Search,
  Timer,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { statusClasses } from "@/lib/status";

const PIPELINE_NODES = [
  "ingest_rss",
  "ingest_search",
  "embed",
  "dedupe",
  "select",
  "fetch_articles",
  "draft",
  "critique",
  "editorial",
  "puzzles",
  "finalize",
  "render",
  "notify",
];

const STAGE_LABELS = {
  dropped_url: "Cross-issue (URL)",
  dropped_title: "Cross-issue (Title)",
  dropped_similarity: "Cross-issue (Similarity)",
  select_llm: "Quality filter",
  select_cap: "Per-batch cap",
  intra_dedup_llm: "Intra-issue dedup (LLM)",
  intra_dedup_emb: "Intra-issue dedup (Embedding)",
  select_topic_cap: "Per-topic cap",
};

const STAGE_COLORS = {
  dropped_url: "bg-amber-50 text-amber-800 border-amber-200",
  dropped_title: "bg-amber-50 text-amber-800 border-amber-200",
  dropped_similarity: "bg-amber-50 text-amber-800 border-amber-200",
  select_llm: "bg-rose-50 text-rose-800 border-rose-200",
  select_cap: "bg-neutral-100 text-neutral-800 border-neutral-200",
  intra_dedup_llm: "bg-indigo-50 text-indigo-800 border-indigo-200",
  intra_dedup_emb: "bg-sky-50 text-sky-800 border-sky-200",
  select_topic_cap: "bg-neutral-100 text-neutral-800 border-neutral-200",
};

const TABS = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "pipeline", label: "Pipeline", icon: Database },
  { id: "drops", label: "Drops", icon: Filter },
  { id: "timings", label: "Timings", icon: Timer },
  { id: "logs", label: "Logs", icon: ScrollText },
  { id: "run", label: "Run info", icon: Clock },
];

function Card({ title, children, className = "" }) {
  return (
    <div className={`rounded-lg border border-neutral-200 bg-white p-5 ${className}`}>
      {title && (
        <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500 mb-3">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-neutral-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-neutral-900">
        {value ?? "-"}
      </div>
      {sub && <div className="mt-0.5 text-xs text-neutral-500">{sub}</div>}
    </div>
  );
}

function EmptyData({ children }) {
  return (
    <p className="text-sm text-neutral-500 italic">{children || "Not available for this run."}</p>
  );
}

function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "-";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.round((ms % 60000) / 1000);
  return `${m}m ${s}s`;
}

function fmtIso(iso) {
  if (!iso) return "-";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function hasAny(obj) {
  return obj && typeof obj === "object" && Object.keys(obj).length > 0;
}

function FunnelView({ funnel }) {
  if (!hasAny(funnel)) return <EmptyData />;
  const steps = [
    { key: "ingested_new", label: "Newly ingested" },
    { key: "embedded_candidates", label: "Embedded" },
    { key: "after_cross_dedup", label: "After cross-issue dedup" },
    { key: "after_select_and_intra_dedup", label: "After select + intra-dedup" },
    ...(funnel.articles_fetched != null
      ? [{ key: "articles_fetched", label: "Full articles fetched" }]
      : []),
    { key: "drafted", label: "Drafted" },
  ];
  const max = Math.max(1, ...steps.map((s) => funnel[s.key] ?? 0));
  return (
    <div className="space-y-2">
      {steps.map((s) => {
        const v = funnel[s.key] ?? 0;
        const pct = Math.round((v / max) * 100);
        return (
          <div key={s.key} className="flex items-center gap-3">
            <div className="w-52 text-sm text-neutral-700">{s.label}</div>
            <div className="flex-1 relative h-6 rounded bg-neutral-100 overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-sky-500/80"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="w-14 text-right text-sm font-mono tabular-nums">{v}</div>
          </div>
        );
      })}
      <div className="flex items-center gap-3 pt-2 text-xs text-neutral-500">
        <div className="w-52">Editorial</div>
        <div className="flex-1">{funnel.editorial_ok ? "Generated" : "Not generated"}</div>
      </div>
      <div className="flex items-center gap-3 text-xs text-neutral-500">
        <div className="w-52">Puzzles</div>
        <div className="flex-1">{funnel.puzzles_status ?? "-"}</div>
      </div>
    </div>
  );
}

function TopicBreakdown({ topics }) {
  if (!topics || topics.length === 0) return <EmptyData />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
          <tr>
            <th className="py-2 pr-4">Topic</th>
            <th className="py-2 pr-4 text-right">Candidates</th>
            <th className="py-2 pr-4 text-right">Batches</th>
            <th className="py-2 pr-4 text-right">Kept</th>
            <th className="py-2 pr-4 text-right">Collapsed</th>
            <th className="py-2 pr-4">Lead?</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-neutral-100">
          {topics.map((t) => (
            <tr key={t.topic_id}>
              <td className="py-2 pr-4 text-neutral-900">{t.topic_name}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{t.candidates}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{t.batches}</td>
              <td className="py-2 pr-4 text-right tabular-nums font-medium">{t.kept}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{t.collapsed_total}</td>
              <td className="py-2 pr-4 text-xs">
                {t.has_lead ? (
                  <span className="text-emerald-700">✓</span>
                ) : (
                  <span className="text-neutral-400">-</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TimingsView({ timings }) {
  if (!hasAny(timings)) return <EmptyData />;
  const entries = Object.entries(timings);
  const total = entries.reduce((s, [, v]) => s + (v || 0), 0);
  const max = Math.max(1, ...entries.map(([, v]) => v || 0));
  return (
    <div className="space-y-2">
      <div className="text-xs text-neutral-500">
        Total agent wall-clock: <span className="font-semibold">{fmtMs(total)}</span>
      </div>
      {entries.map(([node, ms]) => (
        <div key={node} className="flex items-center gap-3">
          <div className="w-36 text-sm font-mono text-neutral-700">{node}</div>
          <div className="flex-1 relative h-5 rounded bg-neutral-100 overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 bg-indigo-500/70"
              style={{ width: `${Math.round(((ms || 0) / max) * 100)}%` }}
            />
          </div>
          <div className="w-20 text-right text-sm tabular-nums">{fmtMs(ms)}</div>
        </div>
      ))}
    </div>
  );
}

function LlmCallsView({ calls }) {
  if (!hasAny(calls)) return <EmptyData />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-neutral-500">
          <tr>
            <th className="py-2 pr-4">Stage</th>
            <th className="py-2 pr-4 text-right">Calls</th>
            <th className="py-2 pr-4 text-right">Total</th>
            <th className="py-2 pr-4 text-right">Retries</th>
            <th className="py-2 pr-4 text-right">Timeouts</th>
            <th className="py-2 pr-4 text-right">Parse err</th>
            <th className="py-2 pr-4 text-right">Exhausted</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-neutral-100">
          {Object.entries(calls).map(([stage, c]) => (
            <tr key={stage}>
              <td className="py-2 pr-4 text-neutral-900">{stage}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{c.count ?? 0}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{fmtMs(c.total_duration_ms)}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{c.retries ?? 0}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{c.timeouts ?? 0}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{c.parse_errors ?? 0}</td>
              <td className="py-2 pr-4 text-right tabular-nums">{c.exhausted ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DropsView({ drops, topics }) {
  const [stageFilter, setStageFilter] = useState("");
  const [topicFilter, setTopicFilter] = useState("");
  const [search, setSearch] = useState("");

  const stages = useMemo(() => {
    const s = new Set();
    for (const d of drops || []) if (d.stage) s.add(d.stage);
    return Array.from(s);
  }, [drops]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (drops || []).filter((d) => {
      if (stageFilter && d.stage !== stageFilter) return false;
      if (topicFilter && (d.topic_id ?? "") !== topicFilter) return false;
      if (q) {
        const hay = [d.snapshot?.title, d.snapshot?.source_domain, d.snapshot?.summary, d.reason]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [drops, stageFilter, topicFilter, search]);

  if (!drops || drops.length === 0) {
    return (
      <EmptyData>
        No drop trace recorded for this run. Either nothing was dropped or this issue was generated
        before drop tracking was added.
      </EmptyData>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <select
          className="h-9 rounded-md border border-neutral-300 bg-white px-3"
          value={stageFilter}
          onChange={(e) => setStageFilter(e.target.value)}
        >
          <option value="">All stages</option>
          {stages.map((s) => (
            <option key={s} value={s}>
              {STAGE_LABELS[s] ?? s}
            </option>
          ))}
        </select>
        <select
          className="h-9 rounded-md border border-neutral-300 bg-white px-3"
          value={topicFilter}
          onChange={(e) => setTopicFilter(e.target.value)}
        >
          <option value="">All topics</option>
          {topics.map((t) => (
            <option key={t.topic_id} value={t.topic_id}>
              {t.topic_name}
            </option>
          ))}
        </select>
        <div className="relative flex-1 min-w-64">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-400" />
          <Input
            className="pl-8"
            placeholder="Search title, source, reason…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="text-xs text-neutral-500">
          {filtered.length} of {drops.length}
        </div>
      </div>

      <div className="divide-y divide-neutral-100 rounded-lg border border-neutral-200 bg-white">
        {filtered.map((d, i) => (
          <div key={`${d.news_item_id}-${d.stage}-${i}`} className="p-4 space-y-1.5">
            <div className="flex flex-wrap items-start gap-2">
              <span
                className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${
                  STAGE_COLORS[d.stage] ?? "bg-neutral-100 text-neutral-700 border-neutral-200"
                }`}
              >
                {STAGE_LABELS[d.stage] ?? d.stage}
              </span>
              {d.topic_name && (
                <span className="inline-flex items-center rounded-md bg-neutral-100 border border-neutral-200 px-2 py-0.5 text-xs text-neutral-700">
                  {d.topic_name}
                </span>
              )}
            </div>
            <div className="text-sm font-medium text-neutral-900">
              {d.snapshot?.title || <em className="text-neutral-500">no title</em>}
            </div>
            <div className="text-sm text-neutral-600 italic">{d.reason || "-"}</div>
            {d.snapshot?.summary && (
              <div className="text-xs text-neutral-500 line-clamp-3">{d.snapshot.summary}</div>
            )}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-500">
              {d.snapshot?.source_domain && <span>{d.snapshot.source_domain}</span>}
              {d.snapshot?.published_at && (
                <span>Published {formatDropDate(d.snapshot.published_at)}</span>
              )}
              {d.snapshot?.url && (
                <a
                  href={d.snapshot.url}
                  target="_blank"
                  rel="noopener"
                  className="inline-flex items-center gap-1 text-neutral-700 hover:underline"
                >
                  <ExternalLink className="h-3 w-3" />
                  source
                </a>
              )}
              <span className="font-mono text-[11px]">{d.news_item_id}</span>
            </div>
            {renderDropOrigin(d.snapshot)}
          </div>
        ))}
      </div>
    </div>
  );
}

function formatDropDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

function renderDropOrigin(snapshot) {
  if (!snapshot) return null;
  if (snapshot.ingested_via === "rss" && snapshot.source_feed_url) {
    return (
      <div className="text-xs text-neutral-500 truncate" title={snapshot.source_feed_url}>
        via RSS: <span className="font-mono">{snapshot.source_feed_url}</span>
      </div>
    );
  }
  if (snapshot.ingested_via === "searxng" && snapshot.source_query) {
    return (
      <div className="text-xs text-neutral-500 truncate" title={snapshot.source_query}>
        via SearXNG: <span className="italic">{snapshot.source_query}</span>
      </div>
    );
  }
  return null;
}

function useNow(enabled) {
  // Ticks every second so elapsed-time counters repaint even between
  // dashboard refetches. Cheap; only enabled while a run is active.
  // Lazy initializer keeps Date.now() out of the render body so
  // react-hooks/purity stays happy.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [enabled]);
  return now;
}

export default function IssueDetails() {
  const { id } = useParams();
  const [tab, setTab] = useState("overview");

  const q = useQuery({
    queryKey: ["issue-details", id],
    queryFn: () => api.getIssueDetails(id),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "running" || s === "pending" ? 2000 : false;
    },
  });

  // Hooks must run unconditionally on every render. Compute these BEFORE
  // the early returns. `useNow` internally calls useState+useEffect.
  const d = q.data ?? {};
  const topics = d.select?.by_topic ?? [];
  const isActive = d.status === "running" || d.status === "pending";
  const now = useNow(isActive);

  if (q.isLoading) {
    return <p className="text-sm text-neutral-500">Loading…</p>;
  }
  if (q.isError) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <div className="font-medium flex items-center gap-2">
          <AlertTriangle className="h-4 w-4" />
          Couldn't load issue details
        </div>
        <div className="mt-1 text-red-600">{q.error.message}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm text-neutral-600 hover:underline"
        >
          <ArrowLeft className="h-4 w-4" /> Back to newspapers
        </Link>
      </div>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">{d.config_name ?? "-"}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-neutral-600">
            <span>{d.issue_date ?? "-"}</span>
            <span>·</span>
            <span className="font-mono text-xs">{d.slug}</span>
            <span>·</span>
            <span
              className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${statusClasses(
                d.status,
              )}`}
            >
              {d.status}
            </span>
          </div>
        </div>
        <div className="flex gap-2">
          <Link to={`/issues/${id}`}>
            <Button variant="outline" size="sm">
              Open preview
            </Button>
          </Link>
        </div>
      </div>

      {isActive && <RunningBanner details={d} now={now} />}

      {d.status === "failed" && <FailedBanner details={d} />}

      {(d.warnings ?? []).length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <div className="font-medium mb-1 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Warnings ({d.warnings.length})
          </div>
          <ul className="list-disc pl-5 space-y-0.5 text-xs">
            {d.warnings.slice(0, 20).map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {isActive && <PipelineProgress details={d} now={now} />}

      <div className="flex flex-wrap gap-1 border-b border-neutral-200">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-sm -mb-px border-b-2 transition-colors ${
                tab === t.id
                  ? "border-neutral-900 text-neutral-900 font-medium"
                  : "border-transparent text-neutral-500 hover:text-neutral-700"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === "overview" && (
        <div className="space-y-6">
          <Card title="Summary">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
              <Stat
                label="Drafted"
                value={d.drafts_count ?? 0}
                sub={d.has_editorial ? "with editorial" : "no editorial"}
              />
              <Stat
                label="Kept in select"
                value={d.select?.total_kept ?? "-"}
                sub={`${d.select?.topics_with_keeps ?? 0} topics`}
              />
              <Stat
                label="After cross-dedup"
                value={d.dedup_breakdown?.kept ?? "-"}
                sub={`of ${d.dedup_breakdown?.total ?? "-"} candidates`}
              />
              <Stat label="Total drops" value={(d.drops ?? []).length} sub="this run, all stages" />
            </div>
          </Card>

          <Card title="Funnel">
            <FunnelView funnel={d.funnel} />
          </Card>

          <Card title="By topic">
            <TopicBreakdown topics={topics} />
          </Card>
        </div>
      )}

      {tab === "pipeline" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card title="Ingest RSS">
              {hasAny(d.ingest_rss) ? (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Feeds polled" value={d.ingest_rss.feeds_polled} />
                  <Stat label="New items" value={d.ingest_rss.items_new} />
                  <Stat label="Failures" value={d.ingest_rss.failures} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Ingest search">
              {hasAny(d.ingest_search) ? (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Topics queried" value={d.ingest_search.topics_queried} />
                  <Stat label="Items added" value={d.ingest_search.items_added} />
                  <Stat label="Failures" value={d.ingest_search.failures} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Embedding">
              {hasAny(d.embedding) ? (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Total" value={d.embedding.total_candidates} />
                  <Stat label="New" value={d.embedding.newly_embedded} />
                  <Stat label="Missing" value={d.embedding.unembedded} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Cross-issue dedup">
              {hasAny(d.dedup_breakdown) ? (
                <div className="grid grid-cols-5 gap-2">
                  <Stat label="Total" value={d.dedup_breakdown.total} />
                  <Stat label="Kept" value={d.dedup_breakdown.kept} />
                  <Stat label="URL" value={d.dedup_breakdown.dropped_url} />
                  <Stat label="Title" value={d.dedup_breakdown.dropped_title} />
                  <Stat label="Similar" value={d.dedup_breakdown.dropped_similarity} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Draft">
              {hasAny(d.draft) ? (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Attempted" value={d.draft.attempted} />
                  <Stat label="Succeeded" value={d.draft.succeeded} />
                  <Stat label="Failures" value={d.draft.failures} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Critique">
              {hasAny(d.critique) ? (
                <div className="grid grid-cols-3 gap-4">
                  <Stat label="Checked" value={d.critique.checked} />
                  <Stat label="Flagged" value={d.critique.flagged} />
                  <Stat label="Rewrites" value={d.critique.rewrites} />
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Editorial">
              {hasAny(d.editorial) ? (
                <div className="space-y-1 text-sm">
                  <div>
                    <span className="text-neutral-500">Status:</span>{" "}
                    <span className="font-medium">{d.editorial.status ?? "-"}</span>
                  </div>
                  {d.editorial.word_count != null && (
                    <div>
                      <span className="text-neutral-500">Words:</span>{" "}
                      <span className="font-medium">{d.editorial.word_count}</span>
                    </div>
                  )}
                  {d.editorial.error && (
                    <div className="text-xs text-red-700">{d.editorial.error}</div>
                  )}
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
            <Card title="Puzzles">
              {hasAny(d.puzzles) ? (
                <div className="space-y-1 text-sm">
                  <div>
                    <span className="text-neutral-500">Status:</span>{" "}
                    <span className="font-medium">{d.puzzles.status ?? "-"}</span>
                  </div>
                  {d.puzzles.theme_pool_size != null && (
                    <div>
                      <span className="text-neutral-500">Theme pool:</span>{" "}
                      <span className="font-medium">{d.puzzles.theme_pool_size}</span>
                    </div>
                  )}
                  {d.puzzles.crypto_phrase_candidates != null && (
                    <div>
                      <span className="text-neutral-500">Crypto candidates:</span>{" "}
                      <span className="font-medium">{d.puzzles.crypto_phrase_candidates}</span>
                    </div>
                  )}
                </div>
              ) : (
                <EmptyData />
              )}
            </Card>
          </div>
        </div>
      )}

      {tab === "drops" && <DropsView drops={d.drops ?? []} topics={topics} />}

      {tab === "timings" && (
        <div className="space-y-6">
          <Card title="Per-node wall-clock">
            <TimingsView timings={d.timings} />
          </Card>
          <Card title="LLM calls by stage">
            <LlmCallsView calls={d.llm_calls} />
          </Card>
        </div>
      )}

      {tab === "logs" && <LogsView issueId={id} active={isActive} />}

      {tab === "run" && (
        <div className="space-y-6">
          <Card title="Run identity">
            {hasAny(d.run) ? (
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
                {Object.entries(d.run).map(([k, v]) => (
                  <div key={k} className="flex gap-3">
                    <dt className="w-56 text-neutral-500">{k}</dt>
                    <dd className="font-mono text-neutral-900 break-all">
                      {typeof v === "string" && /_at$/.test(k) ? fmtIso(v) : String(v)}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : (
              <EmptyData />
            )}
          </Card>
          <Card title="Lookback window">
            <div className="text-sm text-neutral-700">
              {d.lookback_start ?? "-"} → {d.lookback_end ?? "-"}
            </div>
          </Card>
          <Card title="Delivery">
            <DeliveryBlock notify={d.notify} />
            {d.status === "ready" && (
              <div className="mt-4 pt-4 border-t border-neutral-200">
                <SendIssueForm issueId={id} />
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}

function DeliveryBlock({ notify }) {
  if (!notify || !notify.status) {
    return <EmptyData>No automatic delivery recorded for this run.</EmptyData>;
  }
  const { status, sent, failed, per_recipient, reason, error } = notify;
  let pill = "bg-neutral-100 text-neutral-700 border-neutral-200";
  let label = status;
  if (status === "ok") {
    pill = "bg-emerald-50 text-emerald-800 border-emerald-200";
    label = `Sent to ${sent ?? 0} recipient${sent === 1 ? "" : "s"}`;
  } else if (status === "partial") {
    pill = "bg-amber-50 text-amber-800 border-amber-200";
    label = `Sent ${sent ?? 0}, failed ${failed ?? 0}`;
  } else if (status === "failed") {
    pill = "bg-red-50 text-red-800 border-red-200";
    label = "Failed";
  } else if (status === "misconfigured") {
    pill = "bg-red-50 text-red-800 border-red-200";
    label = "SMTP not configured";
  } else if (status === "skipped") {
    pill = "bg-neutral-100 text-neutral-700 border-neutral-200";
    label = `Skipped: ${reason ?? "unknown"}`;
  }
  return (
    <div className="space-y-2 text-sm">
      <span
        className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${pill}`}
      >
        {label}
      </span>
      {error && <div className="text-xs text-red-700 font-mono">{error}</div>}
      {per_recipient && per_recipient.length > 0 && (
        <ul className="text-xs text-neutral-700 space-y-0.5">
          {per_recipient.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span
                className={`font-mono ${r.status === "ok" ? "text-emerald-700" : "text-red-700"}`}
              >
                {r.status === "ok" ? "✓" : "✗"}
              </span>
              <span>{r.email}</span>
              {r.error && <span className="text-red-700 italic">{r.error}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SendIssueForm({ issueId }) {
  const [emails, setEmails] = useState("");
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);

  async function onSend() {
    setErr(null);
    setResult(null);
    const list = emails
      .split(",")
      .map((e) => e.trim())
      .filter(Boolean);
    if (list.length === 0) {
      setErr("Enter at least one email.");
      return;
    }
    setSending(true);
    try {
      const data = await api.sendIssue(issueId, list);
      setResult(data);
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-wide text-neutral-500">Send this issue to</div>
      <div className="flex gap-2">
        <Input
          value={emails}
          onChange={(e) => setEmails(e.target.value)}
          placeholder="one@example.com, two@example.com"
          className="flex-1"
          disabled={sending}
        />
        <Button onClick={onSend} disabled={sending} size="sm">
          {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Send
        </Button>
      </div>
      {err && <div className="text-xs text-red-700">{err}</div>}
      {result && (
        <div className="text-xs text-neutral-700 space-y-1">
          <div>
            Sent {result.sent}, failed {result.failed}.
          </div>
          <ul className="space-y-0.5">
            {result.per_recipient?.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span
                  className={`font-mono ${r.status === "ok" ? "text-emerald-700" : "text-red-700"}`}
                >
                  {r.status === "ok" ? "✓" : "✗"}
                </span>
                <span>{r.email}</span>
                {r.error && <span className="text-red-700 italic">{r.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function RunningBanner({ details, now }) {
  const node = details.current_node;
  const startedAt = node && details.node_started_at?.[node];
  const startedMs = startedAt ? Date.parse(startedAt) : null;
  const elapsed = startedMs ? Math.max(0, now - startedMs) : null;
  const completed = details.completed_nodes ?? [];
  return (
    <div className="rounded-lg border border-sky-200 bg-sky-50 p-4 text-sm text-sky-900">
      <div className="flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="font-medium">
          Generation in progress: {completed.length}/{PIPELINE_NODES.length} steps complete
        </span>
      </div>
      <div className="mt-1 text-xs">
        {node ? (
          <>
            Currently running: <span className="font-mono">{node}</span>
            {elapsed != null && <> · {fmtMs(elapsed)} on this step</>}
          </>
        ) : (
          <>Waiting to start…</>
        )}
      </div>
    </div>
  );
}

function FailedBanner({ details }) {
  const failedAt = details.failed_at_node || details.current_node;
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900">
      <div className="flex items-center gap-2 font-medium">
        <XCircle className="h-4 w-4" />
        Generation failed
      </div>
      {failedAt && (
        <div className="mt-1 text-xs">
          Failed at node: <span className="font-mono">{failedAt}</span>
        </div>
      )}
      {details.error && (
        <pre className="mt-2 text-xs whitespace-pre-wrap font-mono text-red-800 bg-red-100/50 rounded p-2 max-h-48 overflow-auto">
          {details.error}
        </pre>
      )}
    </div>
  );
}

const NODES_WITH_SUB_DETAIL = new Set(["select", "fetch_articles", "draft", "critique", "editorial"]);

function PipelineProgress({ details, now }) {
  const completed = new Set(details.completed_nodes ?? []);
  const current = details.current_node;
  const hasSubDetail = NODES_WITH_SUB_DETAIL.has(current) && details[current]?.progress != null;
  // Default expanded while active; user can collapse per session.
  const [expanded, setExpanded] = useState(true);
  return (
    <Card title="Pipeline">
      <ol className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2 text-sm">
        {PIPELINE_NODES.map((n) => {
          const done = completed.has(n);
          const running = n === current && !done;
          const pending = !done && !running;
          const ms = details.timings?.[n];
          const startedAt = running && details.node_started_at?.[n];
          const elapsed = startedAt ? Math.max(0, now - Date.parse(startedAt)) : null;
          const canExpand = running && NODES_WITH_SUB_DETAIL.has(n) && hasSubDetail;
          return (
            <li
              key={n}
              onClick={canExpand ? () => setExpanded((v) => !v) : undefined}
              className={`rounded-md border px-3 py-2 flex items-center justify-between gap-2 ${
                done
                  ? "border-emerald-200 bg-emerald-50/50"
                  : running
                    ? "border-sky-300 bg-sky-50"
                    : "border-neutral-200 bg-white"
              } ${canExpand ? "cursor-pointer hover:bg-sky-100" : ""}`}
            >
              <span
                className={`font-mono text-xs flex items-center gap-1 ${pending ? "text-neutral-500" : "text-neutral-900"}`}
              >
                {canExpand &&
                  (expanded ? (
                    <ChevronDown className="h-3 w-3" />
                  ) : (
                    <ChevronRight className="h-3 w-3" />
                  ))}
                {n}
              </span>
              <span className="text-xs tabular-nums">
                {done && ms != null && fmtMs(ms)}
                {running && (
                  <span className="inline-flex items-center gap-1 text-sky-700">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {elapsed != null ? fmtMs(elapsed) : "…"}
                  </span>
                )}
                {pending && <span className="text-neutral-400">-</span>}
              </span>
            </li>
          );
        })}
      </ol>
      {hasSubDetail && expanded && (
        <div className="mt-3 rounded-md border border-sky-200 bg-sky-50/40 p-3">
          <NodeSubDetail node={current} progress={details[current].progress} now={now} />
        </div>
      )}
    </Card>
  );
}

function NodeSubDetail({ node, progress, now }) {
  if (node === "select") return <SelectProgressDetail progress={progress} now={now} />;
  if (node === "fetch_articles") return <FetchArticlesProgressDetail progress={progress} now={now} />;
  if (node === "draft") return <DraftProgressDetail progress={progress} now={now} />;
  if (node === "critique") return <CritiqueProgressDetail progress={progress} />;
  if (node === "editorial") return <EditorialProgressDetail progress={progress} now={now} />;
  return null;
}

function FetchArticlesProgressDetail({ progress, now }) {
  const total = progress.stories_total ?? 0;
  const done = progress.stories_done ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  return (
    <div className="space-y-1.5 text-xs">
      <div className="flex items-center justify-between text-neutral-700">
        <span>
          Fetching full text: {done}/{total} stories ({pct}%)
        </span>
        {progress.current_title && (
          <span
            className="truncate max-w-[260px] text-neutral-500 font-mono text-[11px]"
            title={progress.current_title}
          >
            {progress.current_title}
          </span>
        )}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-sky-200">
        <div
          className="h-full bg-sky-600 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function SelectProgressDetail({ progress, now }) {
  const perTopic = progress.per_topic ?? [];
  const phaseStartedAt = progress.phase_started_at ? Date.parse(progress.phase_started_at) : null;
  const phaseElapsed = phaseStartedAt ? Math.max(0, now - phaseStartedAt) : null;
  const phaseLabel = (() => {
    const p = progress.current_phase;
    if (p === "batch") return `batch ${progress.current_batch}/${progress.current_batch_total}`;
    if (p === "intra_dedup_llm") return "intra-dedup (llm)";
    if (p === "intra_dedup_emb") return "intra-dedup (emb)";
    return null;
  })();
  return (
    <div className="space-y-1.5 text-xs">
      <div className="flex items-center justify-between text-neutral-700">
        <span>
          Topics: {progress.topics_done}/{progress.topics_total}
        </span>
      </div>
      <ul className="space-y-1">
        {perTopic.map((t) => {
          const isRunning = t.status === "running";
          const isDone = t.status === "done";
          return (
            <li
              key={t.topic_id}
              className="flex items-center gap-2 py-0.5 border-b border-sky-100 last:border-b-0"
            >
              <span className="w-4 shrink-0">
                {isDone && <Check className="h-3.5 w-3.5 text-emerald-600" />}
                {isRunning && <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-600" />}
                {!isDone && !isRunning && <span className="text-neutral-400">-</span>}
              </span>
              <span className="flex-1 font-medium text-neutral-900">{t.topic_name}</span>
              {isDone && (
                <span className="text-neutral-600 tabular-nums">
                  kept {t.kept}/{t.candidates} · {fmtMs(t.elapsed_ms)}
                </span>
              )}
              {isRunning && (
                <span className="text-sky-700 tabular-nums">
                  {phaseLabel}
                  {phaseElapsed != null && <> · {fmtMs(phaseElapsed)}</>}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function DraftProgressDetail({ progress, now }) {
  const total = progress.stories_total ?? 0;
  const done = progress.stories_done ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const storyStartedAt = progress.story_started_at ? Date.parse(progress.story_started_at) : null;
  const storyElapsed = storyStartedAt ? Math.max(0, now - storyStartedAt) : null;
  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="text-neutral-700">
          Stories: {done}/{total}
        </span>
        <span className="tabular-nums text-neutral-500">{pct}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-sky-100 overflow-hidden">
        <div
          className="h-full bg-sky-500 transition-[width] duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      {progress.current_story_index ? (
        <div className="flex items-start gap-2 pt-1">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-600 mt-0.5 shrink-0" />
          <div className="flex-1">
            <div className="text-neutral-900">
              Story {progress.current_story_index}/{total}
              {progress.current_topic_name && (
                <span className="text-neutral-500"> · {progress.current_topic_name}</span>
              )}
            </div>
            {progress.current_title && (
              <div className="text-neutral-600 truncate" title={progress.current_title}>
                {progress.current_title}
              </div>
            )}
            {storyElapsed != null && (
              <div className="text-sky-700 tabular-nums mt-0.5">{fmtMs(storyElapsed)}</div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function CritiqueProgressDetail({ progress }) {
  const total = progress.drafts_total ?? 0;
  const checked = progress.drafts_checked ?? 0;
  const rewrites = progress.rewrites_done ?? 0;
  return (
    <div className="space-y-1.5 text-xs">
      <div className="flex items-center justify-between text-neutral-700">
        <span>
          Checked {checked}/{total} · Rewrites {rewrites}
        </span>
        {progress.current_phase && (
          <span className="rounded-md border border-sky-300 bg-sky-100 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-sky-800">
            {progress.current_phase}
          </span>
        )}
      </div>
      {progress.current_draft_index ? (
        <div className="text-neutral-600">
          <Loader2 className="inline h-3 w-3 animate-spin mr-1 text-sky-600" />
          Draft {progress.current_draft_index}/{total}
        </div>
      ) : null}
    </div>
  );
}

function EditorialProgressDetail({ progress, now }) {
  const startedAt = progress.started_at ? Date.parse(progress.started_at) : null;
  const elapsed = startedAt ? Math.max(0, now - startedAt) : null;
  return (
    <div className="space-y-1 text-xs">
      <div className="flex items-center gap-2 text-neutral-700">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-600" />
        <span>
          Writing editorial
          {progress.topics_count ? <> for {progress.topics_count} topics</> : null}
        </span>
      </div>
      {elapsed != null && <div className="text-sky-700 tabular-nums pl-5">{fmtMs(elapsed)}</div>}
    </div>
  );
}

function LogsView({ issueId, active }) {
  const [since, setSince] = useState(null);
  const [entries, setEntries] = useState([]);
  const [err, setErr] = useState(null);
  const scrollerRef = useRef(null);
  const bottomRef = useRef(null);
  const [autoscroll, setAutoscroll] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const data = await api.getIssueLogs(issueId, since);
        if (cancelled) return;
        if (data.entries?.length) {
          setEntries((prev) => [...prev, ...data.entries].slice(-1000));
          setSince(data.entries[data.entries.length - 1].ts);
        }
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      }
    }
    poll();
    if (!active) return;
    const t = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [issueId, since, active]);

  useEffect(() => {
    if (!autoscroll) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries, autoscroll]);

  if (!entries.length) {
    return (
      <Card title="Live logs">
        {err && <div className="text-xs text-red-600 mb-2">{err}</div>}
        <EmptyData>
          {active
            ? "Waiting for the first log entry…"
            : "No logs recorded for this run. Either the buffer aged out (10 min after completion) or the run predates log capture."}
        </EmptyData>
      </Card>
    );
  }

  return (
    <Card title="Live logs">
      {err && <div className="text-xs text-red-600 mb-2">{err}</div>}
      <div className="flex items-center justify-between gap-3 mb-2 text-xs text-neutral-500">
        <span>{entries.length} entries · newest at bottom</span>
        <label className="inline-flex items-center gap-1">
          <input
            type="checkbox"
            checked={autoscroll}
            onChange={(e) => setAutoscroll(e.target.checked)}
          />
          auto-scroll
        </label>
      </div>
      <div
        ref={scrollerRef}
        className="max-h-[65vh] overflow-auto rounded-md border border-neutral-200 bg-neutral-950 p-3 font-mono text-xs leading-relaxed"
      >
        {entries.map((e, i) => (
          <div key={`${e.ts}-${i}`} className="whitespace-pre-wrap">
            <span className="text-neutral-500">{e.ts.slice(11, 19)}</span>{" "}
            <span className={levelClass(e.level)}>{e.level.padEnd(7)}</span>{" "}
            <span className="text-neutral-400">{e.logger}</span>{" "}
            <span className="text-neutral-200">{e.msg}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </Card>
  );
}

function levelClass(level) {
  switch (level) {
    case "ERROR":
    case "CRITICAL":
      return "text-red-400";
    case "WARNING":
      return "text-amber-300";
    case "INFO":
      return "text-sky-300";
    case "DEBUG":
      return "text-neutral-500";
    default:
      return "text-neutral-300";
  }
}
