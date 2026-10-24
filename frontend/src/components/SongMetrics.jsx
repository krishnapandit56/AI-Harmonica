import React from 'react';
import { Clock, Activity, Layers, Music2, Sparkles } from 'lucide-react';

export default function SongMetrics({ summary, tags }) {
  if (!summary) return null;

  const metrics = [
    {
      label: 'Instruments',
      value: summary.total_instruments_detected,
      sub: 'Identified in mix',
      icon: <Music2 className="h-4 w-4" />,
      accent: '#6366f1',
      accentBg: 'rgba(99,102,241,0.12)',
    },
    {
      label: 'Dominant Family',
      value: summary.dominant_family || 'Mixed',
      sub: 'Primary sonic texture',
      icon: <Layers className="h-4 w-4" />,
      accent: '#a78bfa',
      accentBg: 'rgba(167,139,250,0.12)',
      small: true,
    },
    {
      label: 'Tempo / BPM',
      value: summary.tempo_bpm,
      unit: 'BPM',
      sub: 'Estimated beat track',
      icon: <Activity className="h-4 w-4" />,
      accent: '#38bdf8',
      accentBg: 'rgba(56,189,248,0.12)',
    },
    {
      label: 'Track Length',
      value: summary.duration_formatted,
      sub: `${summary.analysis_windows_count} analysis window${summary.analysis_windows_count !== 1 ? 's' : ''}`,
      icon: <Clock className="h-4 w-4" />,
      accent: '#34d399',
      accentBg: 'rgba(52,211,153,0.12)',
    },
  ];

  return (
    <div className="w-full">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {metrics.map((m, i) => (
          <div
            key={i}
            className="rounded-2xl p-4 flex flex-col justify-between"
            style={{
              background: 'rgba(15,20,42,0.80)',
              border: '1px solid rgba(51,65,110,0.55)',
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: '#64748b' }}>
                {m.label}
              </span>
              <div className="p-1.5 rounded-lg" style={{ background: m.accentBg, color: m.accent }}>
                {m.icon}
              </div>
            </div>
            <div>
              <div
                className={`font-black tracking-tight ${m.small ? 'text-base' : 'text-2xl'}`}
                style={{ color: '#f1f5f9' }}
              >
                {m.value}
                {m.unit && <span className="text-xs font-semibold ml-1" style={{ color: '#64748b' }}>{m.unit}</span>}
              </div>
              <div className="text-[11px] font-medium mt-0.5" style={{ color: '#475569' }}>{m.sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tags */}
      {tags && tags.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold flex items-center gap-1" style={{ color: '#64748b' }}>
            <Sparkles className="h-3 w-3" style={{ color: '#fbbf24' }} />
            AI Insights:
          </span>
          {tags.map((tag, idx) => (
            <span
              key={idx}
              className="px-3 py-1 rounded-full text-xs font-semibold"
              style={{
                background: 'rgba(99,102,241,0.12)',
                border: '1px solid rgba(99,102,241,0.25)',
                color: '#a5b4fc',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
