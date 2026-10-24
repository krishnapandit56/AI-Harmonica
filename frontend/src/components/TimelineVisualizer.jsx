import React from 'react';
import { Clock, Layers } from 'lucide-react';

const FAMILY_BAR_COLORS = {
  strings:            'linear-gradient(90deg,#10b981,#34d399)',
  bowed_strings:      'linear-gradient(90deg,#8b5cf6,#a78bfa)',
  keyboards:          'linear-gradient(90deg,#f59e0b,#fbbf24)',
  percussion:         'linear-gradient(90deg,#ef4444,#f87171)',
  brass:              'linear-gradient(90deg,#f97316,#fb923c)',
  woodwinds:          'linear-gradient(90deg,#0891b2,#22d3ee)',
  world_traditional:  'linear-gradient(90deg,#6366f1,#818cf8)',
};

const LEGEND = [
  { color: '#34d399', label: 'Strings' },
  { color: '#fbbf24', label: 'Keyboards' },
  { color: '#f87171', label: 'Percussion' },
  { color: '#a78bfa', label: 'Bowed Strings' },
  { color: '#22d3ee', label: 'Woodwinds' },
  { color: '#fb923c', label: 'Brass' },
];

export default function TimelineVisualizer({ detectedInstruments, durationSeconds }) {
  if (!detectedInstruments?.length || !durationSeconds) return null;

  const activeInstruments = detectedInstruments.filter(i => i.timeline?.length > 0).slice(0, 8);
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  const ticks = Array.from({ length: 5 }, (_, i) => ({
    formatted: fmt((i / 4) * durationSeconds),
    percent: (i / 4) * 100,
  }));

  return (
    <div className="rounded-2xl p-6"
      style={{ background: 'rgba(15,20,42,0.80)', border: '1px solid rgba(51,65,110,0.55)' }}>

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4" style={{ color: '#818cf8' }} />
            <h3 className="font-black text-lg" style={{ color: '#f1f5f9' }}>
              Arrangement &amp; Instrument Timeline
            </h3>
          </div>
          <p className="text-xs mt-0.5" style={{ color: '#475569' }}>
            Temporal presence of instruments across song progression
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-xl"
          style={{ background: 'rgba(10,13,26,0.6)', border: '1px solid rgba(51,65,110,0.4)', color: '#64748b' }}>
          <Clock className="h-3.5 w-3.5" />
          Total: {fmt(durationSeconds)}
        </div>
      </div>

      {/* Time ruler */}
      <div className="flex justify-between mb-3 pb-2 font-mono text-[11px]"
        style={{ color: '#475569', borderBottom: '1px solid rgba(51,65,110,0.35)' }}>
        {ticks.map((t, i) => <span key={i}>{t.formatted}</span>)}
      </div>

      {/* Tracks */}
      <div className="space-y-2.5">
        {activeInstruments.map((inst) => {
          const barGrad = FAMILY_BAR_COLORS[inst.family] || 'linear-gradient(90deg,#6366f1,#818cf8)';
          return (
            <div key={inst.id} className="flex items-center gap-4">
              <div className="w-36 sm:w-44 shrink-0 text-right">
                <span className="text-xs font-bold block truncate" style={{ color: '#e2e8f0' }} title={inst.name}>
                  {inst.name}
                </span>
                <span className="text-[10px] font-medium block" style={{ color: '#475569' }}>
                  {inst.family_name}
                </span>
              </div>
              <div className="relative flex-1 h-6 rounded-xl overflow-hidden"
                style={{ background: 'rgba(10,13,26,0.6)', border: '1px solid rgba(51,65,110,0.35)' }}>
                {inst.timeline.map((seg, si) => {
                  const left = Math.max(0, Math.min(100, (seg.start / durationSeconds) * 100));
                  const width = Math.max(2, Math.min(100 - left, ((seg.end - seg.start) / durationSeconds) * 100));
                  return (
                    <div
                      key={si}
                      style={{ left: `${left}%`, width: `${width}%`, background: barGrad, position: 'absolute', top: '4px', bottom: '4px', borderRadius: '6px', opacity: 0.85 }}
                      className="hover:opacity-100 transition-opacity cursor-pointer"
                      title={`${inst.name}: ${fmt(seg.start)} – ${fmt(seg.end)} (${seg.confidence}%)`}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-5 pt-4 flex flex-wrap items-center gap-3 text-[11px]"
        style={{ borderTop: '1px solid rgba(51,65,110,0.35)', color: '#475569' }}>
        <span className="font-bold" style={{ color: '#64748b' }}>Legend:</span>
        {LEGEND.map((l) => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: l.color }} />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
