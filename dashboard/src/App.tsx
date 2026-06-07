import { useState, useEffect } from "react";
import { apiFetch } from "./api";
import { EventFeed } from "./components/EventFeed";
import { SessionTimeline } from "./components/SessionTimeline";
import { AlertBadge } from "./components/AlertBadge";

type Tab = "feed" | "timeline" | "alerts";

interface AlertEvent {
  event_id: string;
  session_id: string;
  agent_id: string;
  timestamp: string;
  event_type: string;
  severity: "info" | "warning" | "critical";
  flags: string[];
  payload: Record<string, unknown>;
}

const SEVERITY_BG: Record<string, string> = {
  warning: "bg-amber-950 border-l-2 border-amber-500",
  critical: "bg-red-950 border-l-2 border-red-500",
};

const SEVERITY_TEXT: Record<string, string> = {
  warning: "text-amber-400",
  critical: "text-red-400",
};

function formatTs(ts: string): string {
  return new Date(ts).toLocaleString("en-US", { hour12: false });
}

function AlertsPanel() {
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await apiFetch("/events/alerts");
        if (res.ok && !cancelled) setAlerts(await res.json());
      } catch (_) {}
    };
    load();
    const id = setInterval(load, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (alerts.length === 0) {
    return (
      <div className="p-8 text-center text-gray-600 text-sm">
        No alerts. All clear.
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      {alerts.map((a) => (
        <div
          key={a.event_id}
          className={`rounded p-4 ${SEVERITY_BG[a.severity] ?? "bg-gray-800"}`}
        >
          <div className="flex items-center justify-between mb-1">
            <span className={`text-sm font-bold ${SEVERITY_TEXT[a.severity]}`}>
              {a.event_type}
            </span>
            <span className="text-xs text-gray-500">{formatTs(a.timestamp)}</span>
          </div>
          <div className="text-xs text-sky-400 mb-2">
            agent: {a.agent_id} &nbsp;·&nbsp; session: {a.session_id.slice(0, 16)}…
          </div>
          {a.flags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {a.flags.map((f) => (
                <span
                  key={f}
                  className="text-xs px-1.5 py-0.5 bg-purple-900 text-purple-300 rounded"
                >
                  {f}
                </span>
              ))}
            </div>
          )}
          {Object.keys(a.payload).length > 0 && (
            <pre className="text-xs text-gray-400 bg-gray-900 rounded p-2 overflow-auto max-h-40">
              {JSON.stringify(a.payload, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("feed");
  const [alertCount, setAlertCount] = useState(0);
  const [eventCount, setEventCount] = useState(0);

  useEffect(() => {
    const load = async () => {
      try {
        const [healthRes, alertRes] = await Promise.all([
          apiFetch("/health"),
          apiFetch("/events/alerts?limit=200"),
        ]);
        if (healthRes.ok) {
          const h = await healthRes.json();
          setEventCount(h.event_count ?? 0);
        }
        if (alertRes.ok) {
          const a = await alertRes.json();
          setAlertCount(Array.isArray(a) ? a.length : 0);
        }
      } catch (_) {}
    };
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  const tabs: { id: Tab; label: string }[] = [
    { id: "feed", label: "Live Feed" },
    { id: "timeline", label: "Session Timeline" },
    { id: "alerts", label: "Alerts" },
  ];

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">
            AgentGuard
          </h1>
          <p className="text-xs text-gray-500">Security Observability for AI Agents</p>
        </div>
        <div className="text-xs text-gray-600">
          {eventCount} total events
        </div>
      </header>

      {/* Nav */}
      <nav className="border-b border-gray-800 px-6">
        <div className="flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium transition-colors flex items-center ${
                tab === t.id
                  ? "text-white border-b-2 border-sky-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {t.label}
              {t.id === "alerts" && <AlertBadge count={alertCount} />}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="p-0">
        {tab === "feed" && <EventFeed />}
        {tab === "timeline" && <SessionTimeline />}
        {tab === "alerts" && <AlertsPanel />}
      </main>
    </div>
  );
}
