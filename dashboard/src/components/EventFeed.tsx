import { useState, useEffect } from "react";
import { apiFetch } from "../api";

interface SecurityEvent {
  event_id: string;
  session_id: string;
  agent_id: string;
  timestamp: string;
  source: string;
  event_type: string;
  severity: "info" | "warning" | "critical";
  payload: Record<string, unknown>;
  flags: string[];
  parent_event_id: string | null;
}

const SEVERITY_CLASSES: Record<string, string> = {
  info: "text-gray-400",
  warning: "text-amber-400",
  critical: "text-red-400",
};

const SEVERITY_BG: Record<string, string> = {
  info: "bg-gray-800",
  warning: "bg-amber-950 border-l-2 border-amber-500",
  critical: "bg-red-950 border-l-2 border-red-500",
};

function formatTs(ts: string): string {
  return new Date(ts).toLocaleTimeString("en-US", { hour12: false });
}

function truncate(s: string, n = 12): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

interface RowProps {
  event: SecurityEvent;
}

function EventRow({ event }: RowProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr
        className={`cursor-pointer hover:brightness-125 transition-all ${SEVERITY_BG[event.severity]}`}
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-3 py-2 text-gray-500 text-xs">{formatTs(event.timestamp)}</td>
        <td className="px-3 py-2 text-xs text-gray-400 font-mono">
          {truncate(event.session_id, 14)}
        </td>
        <td className="px-3 py-2 text-xs text-sky-400">{event.agent_id}</td>
        <td className="px-3 py-2 text-xs">{event.event_type}</td>
        <td className={`px-3 py-2 text-xs font-bold uppercase ${SEVERITY_CLASSES[event.severity]}`}>
          {event.severity}
        </td>
        <td className="px-3 py-2 text-xs text-purple-400">
          {event.flags.map((f) => (
            <span key={f} className="mr-1 px-1 bg-purple-900 rounded text-purple-300">
              {f}
            </span>
          ))}
        </td>
      </tr>
      {expanded && (
        <tr className={SEVERITY_BG[event.severity]}>
          <td colSpan={6} className="px-4 pb-3">
            <pre className="text-xs text-gray-300 bg-gray-900 rounded p-3 overflow-auto max-h-64">
              {JSON.stringify(event.payload, null, 2)}
            </pre>
          </td>
        </tr>
      )}
    </>
  );
}

export function EventFeed() {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchEvents() {
      try {
        const res = await apiFetch("/events?limit=100");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setEvents(data);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    }

    fetchEvents();
    const id = setInterval(fetchEvents, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (error) {
    return (
      <div className="p-6 text-red-400 text-sm">
        Failed to load events: {error}
        <br />
        <span className="text-gray-500">Is the API running on localhost:8000?</span>
      </div>
    );
  }

  return (
    <div className="overflow-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="text-gray-500 text-xs uppercase tracking-wider border-b border-gray-800">
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Session</th>
            <th className="px-3 py-2">Agent</th>
            <th className="px-3 py-2">Event Type</th>
            <th className="px-3 py-2">Severity</th>
            <th className="px-3 py-2">Flags</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {events.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-3 py-8 text-center text-gray-600 text-sm">
                No events yet. Run an example to generate traffic.
              </td>
            </tr>
          ) : (
            events.map((e) => <EventRow key={e.event_id} event={e} />)
          )}
        </tbody>
      </table>
    </div>
  );
}
