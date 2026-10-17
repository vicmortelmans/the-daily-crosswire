import { useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Trash2, Plus, ArrowLeft, Save, Play, ExternalLink } from "lucide-react";
import { statusClasses, isActiveStatus } from "@/lib/status";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ScheduleFields } from "@/components/ScheduleFields";

const DEFAULT_SCHEDULE = {
  schedule_kind: "interval",
  frequency_days: 1,
  weekdays: null,
  run_hour: 6,
};

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "de", label: "German" },
  { code: "nl", label: "Dutch" },
];

function pickSchedulePayload(s) {
  if (s.schedule_kind === "interval") {
    return {
      schedule_kind: "interval",
      frequency_days: s.frequency_days ?? 1,
      weekdays: null,
      run_hour: s.run_hour ?? 6,
    };
  }
  return {
    schedule_kind: "weekdays",
    frequency_days: null,
    weekdays: s.weekdays ?? [],
    run_hour: s.run_hour ?? 6,
  };
}

export default function ConfigEdit() {
  const { id } = useParams();
  const isNew = !id || id === "new";
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: config, isLoading } = useQuery({
    queryKey: ["config", id],
    queryFn: () => api.getConfig(id),
    enabled: !isNew,
  });

  if (isNew) return <NewConfigForm onCreated={(c) => navigate(`/configs/${c.id}`)} />;
  if (isLoading) return <p className="text-sm text-neutral-500">Loading…</p>;
  if (!config) return <p className="text-sm text-red-600">Config not found.</p>;

  return (
    // key={config.id} forces React to remount the form when the user
    // navigates to a different config. That resets all useState
    // initializers with the new config's values, replacing what used
    // to be a prop-resync useEffect (which ESLint's react-hooks/purity
    // rule flags because it triggers cascading renders).
    <ExistingConfigForm
      key={config.id}
      config={config}
      onSaved={() => qc.invalidateQueries({ queryKey: ["config", id] })}
      onDeleted={() => navigate("/configs")}
    />
  );
}

function NewConfigForm({ onCreated }) {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [schedule, setSchedule] = useState(DEFAULT_SCHEDULE);
  const [includePuzzles, setIncludePuzzles] = useState(true);
  const [language, setLanguage] = useState("en");
  const [error, setError] = useState(null);
  const qc = useQueryClient();

  const create = useMutation({
    mutationFn: (data) => api.createConfig(data),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["configs"] });
      onCreated(c);
    },
    onError: (e) => setError(e.message),
  });

  function submit() {
    setError(null);
    create.mutate({
      name: name.trim(),
      include_puzzles: includePuzzles,
      language,
      ...pickSchedulePayload(schedule),
    });
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <Button variant="ghost" size="sm" onClick={() => navigate("/configs")}>
          <ArrowLeft className="h-3.5 w-3.5" /> Back
        </Button>
        <h1 className="text-2xl font-semibold text-neutral-900 mt-3">New config</h1>
      </div>

      <Card>
        <CardContent>
          <div className="space-y-1.5">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              placeholder="daily-crosswire"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <ScheduleFields value={schedule} onChange={setSchedule} />

          <div className="space-y-1.5">
            <Label htmlFor="language-new">Output language</Label>
            <select
              id="language-new"
              className="h-9 w-full max-w-48 rounded-md border border-neutral-300 bg-white px-3 text-sm"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Drafts + editorial will be written in this language, translated from multilingual
              sources as needed.
            </p>
          </div>

          <label className="flex items-center gap-2 text-sm text-neutral-800">
            <Checkbox
              checked={includePuzzles}
              onChange={(e) => setIncludePuzzles(e.target.checked)}
            />
            Include puzzles page
          </label>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex justify-end">
            <Button disabled={!name.trim() || create.isPending} onClick={submit}>
              <Save className="h-4 w-4" />
              Create
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ExistingConfigForm({ config, onSaved, onDeleted }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [name, setName] = useState(config.name);
  const [schedule, setSchedule] = useState(scheduleFromConfig(config));
  const [includePuzzles, setIncludePuzzles] = useState(config.include_puzzles);
  const [active, setActive] = useState(config.active);
  const [language, setLanguage] = useState(config.language ?? "en");
  const [newsletterEmails, setNewsletterEmails] = useState(
    (config.newsletter_emails ?? []).join(", "),
  );
  const [error, setError] = useState(null);

  // State initializers above pull from `config` on mount. The parent
  // passes `key={config.id}`, so navigating to a different config
  // remounts this component and re-runs initializers, replacing the
  // prop-resync useEffect that react-hooks/purity flags.

  const save = useMutation({
    mutationFn: (data) => api.updateConfig(config.id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["configs"] });
      onSaved();
    },
    onError: (e) => setError(e.message),
  });

  const remove = useMutation({
    mutationFn: () => api.deleteConfig(config.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["configs"] });
      onDeleted();
    },
  });

  function submit() {
    setError(null);
    const parsedEmails = newsletterEmails
      .split(",")
      .map((e) => e.trim())
      .filter(Boolean);
    save.mutate({
      name: name.trim(),
      include_puzzles: includePuzzles,
      active,
      language,
      newsletter_emails: parsedEmails,
      ...pickSchedulePayload(schedule),
    });
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <Button variant="ghost" size="sm" onClick={() => navigate("/configs")}>
            <ArrowLeft className="h-3.5 w-3.5" /> Back
          </Button>
          <h1 className="text-2xl font-semibold text-neutral-900 mt-3">{config.name}</h1>
        </div>
        <Button
          variant="destructive"
          size="sm"
          onClick={() => {
            if (confirm(`Delete config "${config.name}"? This removes its topics and feeds.`))
              remove.mutate();
          }}
        >
          <Trash2 className="h-3.5 w-3.5" />
          Delete config
        </Button>
      </div>

      <GenerateNowSection config={config} />

      <Card>
        <CardHeader>
          <CardTitle>Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-1.5">
            <Label htmlFor="name">Name</Label>
            <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <ScheduleFields value={schedule} onChange={setSchedule} />

          <div className="space-y-1.5">
            <Label htmlFor="language-edit">Output language</Label>
            <select
              id="language-edit"
              className="h-9 w-full max-w-48 rounded-md border border-neutral-300 bg-white px-3 text-sm"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Drafts + editorial will be written in this language, translated from multilingual
              sources as needed.
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="newsletter-emails">Newsletter emails</Label>
            <Input
              id="newsletter-emails"
              value={newsletterEmails}
              onChange={(e) => setNewsletterEmails(e.target.value)}
              placeholder="alice@example.com, bob@example.com"
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated. After each successful run, a PDF of the issue is emailed to this
              list. Leave empty to disable delivery for this config.
            </p>
          </div>

          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 text-sm text-neutral-800">
              <Checkbox
                checked={includePuzzles}
                onChange={(e) => setIncludePuzzles(e.target.checked)}
              />
              Include puzzles page
            </label>
            <label className="flex items-center gap-2 text-sm text-neutral-800">
              <Checkbox checked={active} onChange={(e) => setActive(e.target.checked)} />
              Active (scheduler runs it)
            </label>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex justify-end">
            <Button disabled={save.isPending} onClick={submit}>
              <Save className="h-4 w-4" />
              Save
            </Button>
          </div>
        </CardContent>
      </Card>

      <TopicsSection config={config} />
    </div>
  );
}

function GenerateNowSection({ config }) {
  const qc = useQueryClient();
  const [pollingIssueId, setPollingIssueId] = useState(null);

  // Poll the most recent issue for this config while a run is in flight.
  const issueQ = useQuery({
    queryKey: ["issue", pollingIssueId],
    queryFn: () => api.getIssue(pollingIssueId),
    enabled: Boolean(pollingIssueId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "running" || s === "pending" ? 30000 : false;
    },
  });

  // Look up the most recent issue for this config on mount (so a fresh reload
  // after triggering still shows the tail of the last run).
  const latestQ = useQuery({
    queryKey: ["issues", "latest", config.id],
    queryFn: () => api.listIssues({ config_id: config.id, limit: 1 }),
    // Only poll while the most recent issue is in-flight.
    refetchInterval: (query) => (isActiveStatus(query.state.data?.[0]?.status) ? 30000 : false),
  });
  const latest = latestQ.data && latestQ.data[0];

  const trigger = useMutation({
    mutationFn: () => api.triggerGenerate(config.id),
    onSuccess: (data) => {
      // Backend creates the Issue synchronously and returns its id, so we
      // can poll that exact row; no timer race.
      if (data?.issue_id) {
        setPollingIssueId(data.issue_id);
      }
      qc.invalidateQueries({ queryKey: ["issues"] });
      latestQ.refetch();
    },
  });

  const shown = issueQ.data ?? latest;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Generate now</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            onClick={() => trigger.mutate()}
            disabled={trigger.isPending || config.topics.length === 0}
          >
            <Play className="h-4 w-4" />
            {trigger.isPending ? "Enqueueing…" : "Generate now"}
          </Button>
          {config.topics.length === 0 && (
            <p className="text-xs text-neutral-500">Add at least one topic before running.</p>
          )}
          {trigger.error && <p className="text-xs text-red-600">{trigger.error.message}</p>}
        </div>

        {shown && (
          <div className="rounded-md border border-neutral-200 bg-neutral-50/50 p-3 text-sm space-y-1">
            <div className="flex items-center justify-between">
              <span>
                Last run: <span className="font-mono text-xs">{shown.slug}</span>
              </span>
              <span
                className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${statusClasses(shown.status)}`}
              >
                {shown.status}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-neutral-600">
              <span>Issue date: {shown.issue_date}</span>
              <Link to={`/issues/${shown.id}`} className="underline">
                Open preview
              </Link>
              {shown.status === "ready" && shown.html_path && (
                <a
                  href={`/${shown.html_path}`}
                  target="_blank"
                  rel="noopener"
                  className="inline-flex items-center gap-1 underline"
                >
                  <ExternalLink className="h-3 w-3" /> Raw
                </a>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function scheduleFromConfig(config) {
  return {
    schedule_kind: config.schedule_kind ?? "interval",
    frequency_days: config.frequency_days,
    weekdays: config.weekdays,
    run_hour: config.run_hour ?? 6,
  };
}

function TopicsSection({ config }) {
  const qc = useQueryClient();
  const refetch = () => qc.invalidateQueries({ queryKey: ["config", config.id] });

  const createTopic = useMutation({
    mutationFn: (data) => api.createTopic(config.id, data),
    onSuccess: refetch,
  });

  const [newName, setNewName] = useState("");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Topics ({config.topics.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input
            placeholder="New topic (e.g. AI/LLMs)"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && newName.trim()) {
                createTopic.mutate({ name: newName.trim(), preferred_sources: [] });
                setNewName("");
              }
            }}
          />
          <Button
            disabled={!newName.trim() || createTopic.isPending}
            onClick={() => {
              createTopic.mutate({ name: newName.trim(), preferred_sources: [] });
              setNewName("");
            }}
          >
            <Plus className="h-4 w-4" />
            Add
          </Button>
        </div>

        {config.topics.length === 0 ? (
          <p className="text-sm text-neutral-500">
            No topics yet. Add one to start configuring sources.
          </p>
        ) : (
          <div className="space-y-3">
            {config.topics.map((t) => (
              <TopicRow key={t.id} topic={t} onChange={refetch} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TopicRow({ topic, onChange }) {
  const [name, setName] = useState(topic.name);
  const [sourcesText, setSourcesText] = useState(topic.preferred_sources.join(", "));
  const [hints, setHints] = useState(topic.search_hints ?? "");
  const [feedUrl, setFeedUrl] = useState("");
  const [dirty, setDirty] = useState(false);

  // Parent renders `<TopicRow key={t.id} ...>` so a new topic id remounts
  // this row and the initializers above pull fresh values. No prop-resync
  // useEffect needed; keeping one made ESLint's react-hooks/purity
  // unhappy and overwrote user edits after auto-save refetches.

  const save = useMutation({
    mutationFn: (data) => api.updateTopic(topic.id, data),
    onSuccess: () => {
      setDirty(false);
      onChange();
    },
  });

  function persist() {
    if (!dirty) return;
    save.mutate({
      name: name.trim(),
      preferred_sources: sourcesText
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      search_hints: hints || null,
    });
  }

  function markDirty(setter) {
    return (e) => {
      setter(e.target.value);
      setDirty(true);
    };
  }

  const remove = useMutation({
    mutationFn: () => api.deleteTopic(topic.id),
    onSuccess: onChange,
  });
  const addFeed = useMutation({
    mutationFn: (url) => api.createFeed(topic.id, { url }),
    onSuccess: () => {
      setFeedUrl("");
      onChange();
    },
  });
  const removeFeed = useMutation({
    mutationFn: (fid) => api.deleteFeed(fid),
    onSuccess: onChange,
  });

  return (
    <div className="rounded-md border border-neutral-200 bg-neutral-50/50 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Input
          value={name}
          onChange={markDirty(setName)}
          onBlur={persist}
          className="font-medium"
        />
        {dirty && (
          <span className="text-xs text-amber-600 font-medium whitespace-nowrap">unsaved</span>
        )}
        {save.isPending && (
          <span className="text-xs text-neutral-500 whitespace-nowrap">saving…</span>
        )}
        <Button
          size="icon"
          variant="ghost"
          onClick={() => {
            if (confirm(`Remove topic "${topic.name}"?`)) remove.mutate();
          }}
        >
          <Trash2 className="h-3.5 w-3.5 text-red-600" />
        </Button>
      </div>

      <div className="space-y-1.5">
        <Label>Preferred sources (comma-separated domains)</Label>
        <Input
          placeholder="arxiv.org, theverge.com"
          value={sourcesText}
          onChange={markDirty(setSourcesText)}
          onBlur={persist}
        />
      </div>

      <div className="space-y-1.5">
        <Label>Search hints (free text, guides the agent)</Label>
        <Textarea
          placeholder="e.g. prefer academic or industry coverage over speculation"
          value={hints}
          onChange={markDirty(setHints)}
          onBlur={persist}
        />
      </div>

      <div className="space-y-1.5">
        <Label>RSS feeds ({topic.feeds.length})</Label>
        <div className="flex gap-2">
          <Input
            placeholder="https://example.com/feed.xml"
            value={feedUrl}
            onChange={(e) => setFeedUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && feedUrl.trim()) addFeed.mutate(feedUrl.trim());
            }}
          />
          <Button
            size="sm"
            variant="outline"
            disabled={!feedUrl.trim() || addFeed.isPending}
            onClick={() => addFeed.mutate(feedUrl.trim())}
          >
            Add
          </Button>
        </div>
        {topic.feeds.length > 0 && (
          <ul className="space-y-1">
            {topic.feeds.map((f) => (
              <li
                key={f.id}
                className="flex items-center justify-between rounded border border-neutral-200 bg-white px-3 py-1.5 text-xs"
              >
                <span className="truncate text-neutral-700">{f.url}</span>
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => {
                    if (confirm(`Remove feed?\n${f.url}`)) removeFeed.mutate(f.id);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5 text-red-600" />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
