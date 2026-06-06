import { useState, useEffect } from "react";

interface Session {
  session_id: string;
  total_events: number;
  critical_count: number;
  warning_count: number;
  started_at: string | null;
  last_seen: string | null;
}

interface SecurityEvent {
  event_id: string;
  session_id: string;
  agent_id: string;
  timestamp: string;
  event_type: string;
  severity: "info" | "warning" | "critical";
  payload: Record<string, unknown>;
  flags: string[];
}

const BORDER_CLASSES: Record<string, string> = {
  info: "border-gray-700",
  warning: "border-amber-500",
  critical: "border-red-500",
};

const DOT_CLASSES: Record<string, string> = {
  info: "bg-gray-600",
  warning: "bg-amber-500",
  critical: "bg-red-500",
};

function formatTs(ts: string): string {
  return new Date(ts).toLocaleTimeString("en-US", { hour12: false });
}

export function SessionTimeline() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/sessions")
      .then((r) => r.json())
      .then((data) => {
        setSessions(data);
        if (data.length > 0 && !selectedId) {
          setSelectedId(data[0].session_id);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    fetch(`/sessions/${selectedId}`)
      .then((r) => r.json())
      .then((data) => {
        setEvents(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [selectedId]);

  return (
    <div className="p-4">
      <div className="mb-6">
        <label className="text-xs text-gray-500 uppercase tracking-wider block mb-2">
          Session
        </label>
        <select
          className="bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 w-full max-w-md border border-gray-700 focus:outline-none focus:border-sky-500"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
        >
          {sessions.length === 0 && (
            <option value="">No sessions yet</option>
          )}
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id.slice(0, 20)}… — {s.total_events} events
              {s.critical_count > 0 ? ` ⚠ ${s.critical_count} critical` : ""}
            </option>
          ))}
        </select>
      </div>

      {loading && <p className="text-gray-500 text-sm">Loading events…</p>}

      {!loading && events.length === 0 && selectedId && (
        <p className="text-gray-600 text-sm">No events found for this session.</p>
      )}

      <div className="relative">
        {/* Vertical timeline line */}
        {events.length > 0 && (
          <div className="absolute left-4 top-0 bottom-0 w-px bg-gray-800" />
        )}

        <div className="space-y-3">
          {events.map((event) => (
            <div key={event.event_id} className="flex gap-4 items-start">
              {/* Timeline dot */}
              <div
                className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 z-10 ${DOT_CLASSES[event.severity]}`}
              />

              {/* Event card */}
              <div
                className={`flex-1 rounded p-3 bg-gray-900 border-l-2 ${BORDER_CLASSES[event.severity]}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-gray-200">{event.event_type}</span>
                  <span className="text-xs text-gray-500">{formatTs(event.timestamp)}</span>
                </div>
                <div className="text-xs text-sky-400 mb-1">{event.agent_id}</div>

                {event.flags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {event.flags.map((f) => (
                      <span
                        key={f}
                        className="text-xs px-1.5 py-0.5 bg-purple-900 text-purple-300 rounded"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                )}

                {Object.keys(event.payload).length > 0 && (
                  <pre className="text-xs text-gray-500 overflow-auto max-h-32 mt-1">
                    {JSON.stringify(event.payload, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
