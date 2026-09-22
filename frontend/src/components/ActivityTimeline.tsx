import type { ActivityEntry } from "../types";

const AVATAR_COLORS = ["avatar--purple", "avatar--blue", "avatar--teal"];

export function ActivityTimeline({ entries }: { entries: ActivityEntry[] }) {
  return (
    <div className="card">
      <ul className="timeline">
        {entries.map((entry, i) => (
          <li className="timeline__row" key={`${entry.title}-${i}`}>
            <div className={`avatar avatar--sm ${AVATAR_COLORS[i % AVATAR_COLORS.length]}`}>
              {entry.initials}
            </div>
            <div>
              <div className="timeline__timestamp">{entry.timestamp}</div>
              <div className="timeline__title">{entry.title}</div>
              <div className="timeline__actor">{entry.actor}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
