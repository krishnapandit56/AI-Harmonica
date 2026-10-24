import React, { useState } from 'react';
import { Music, Zap, Brain, GitMerge, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

// Family accent palette (dark-mode friendly)
const FAMILY_ACCENT = {
  strings:            { bar: 'linear-gradient(90deg,#10b981,#34d399)', color: '#34d399', bg: 'rgba(16,185,129,0.1)' },
  bowed_strings:      { bar: 'linear-gradient(90deg,#8b5cf6,#a78bfa)', color: '#a78bfa', bg: 'rgba(139,92,246,0.1)' },
  keyboards:          { bar: 'linear-gradient(90deg,#f59e0b,#fbbf24)', color: '#fbbf24', bg: 'rgba(245,158,11,0.1)' },
  percussion:         { bar: 'linear-gradient(90deg,#ef4444,#f87171)', color: '#f87171', bg: 'rgba(239,68,68,0.1)' },
  brass:              { bar: 'linear-gradient(90deg,#f97316,#fb923c)', color: '#fb923c', bg: 'rgba(249,115,22,0.1)' },
  woodwinds:          { bar: 'linear-gradient(90deg,#0891b2,#22d3ee)', color: '#22d3ee', bg: 'rgba(8,145,178,0.1)' },
  world_traditional:  { bar: 'linear-gradient(90deg,#6366f1,#818cf8)', color: '#818cf8', bg: 'rgba(99,102,241,0.1)' },
};

const DEFAULT_ACCENT = { bar: 'linear-gradient(90deg,#334155,#475569)', color: '#94a3b8', bg: 'rgba(51,65,85,0.15)' };

// Source display config
const SOURCE_CONFIG = {
  ensemble: {
    label: 'Both Models',
    labelShort: 'Both',
    bg: 'linear-gradient(90deg,#6366f1,#8b5cf6)',
    color: '#fff',
    icon: <GitMerge className="h-3 w-3" />,
  },
  cnn14: {
    label: 'CNN14',
    labelShort: 'CNN14',
    bg: 'rgba(14,165,233,0.15)',
    color: '#38bdf8',
    border: '1px solid rgba(14,165,233,0.3)',
    icon: <Zap className="h-3 w-3" />,
  },
  openmic: {
    label: 'CNN (trained on OpenMIC)',
    labelShort: 'CNN',
    bg: 'rgba(124,58,237,0.15)',
    color: '#a78bfa',
    border: '1px solid rgba(124,58,237,0.3)',
    icon: <Brain className="h-3 w-3" />,
  },
};

export default function InstrumentCard({ instrument, rank, isBothSection }) {
  const [showBreakdown, setShowBreakdown] = useState(false);
  const [showFact, setShowFact]           = useState(false);

  const accent   = FAMILY_ACCENT[instrument.family] || DEFAULT_ACCENT;
  const src      = SOURCE_CONFIG[instrument.source]  || SOURCE_CONFIG.ensemble;
  const isBoth   = instrument.source === 'ensemble';
  const isHighConf = instrument.confidence >= 75;
  const isDominant = instrument.intensity === 'Dominant / Lead';
  const hasBreakdown = instrument.cnn14_confidence !== undefined;

  // Card style: both-models gets extra glow/border
  const cardStyle = isBoth
    ? {
        background: 'rgba(30,27,75,0.78)',
        border: '1px solid rgba(129,140,248,0.48)',
        boxShadow: '0 0 28px rgba(99,102,241,0.18), 0 4px 24px rgba(0,0,0,0.55)',
      }
    : {
        background: 'rgba(15,20,42,0.80)',
        border: '1px solid rgba(51,65,110,0.55)',
      };

  return (
    <div
      className="rounded-2xl p-5 flex flex-col transition-all duration-250 anim-in"
      style={cardStyle}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-2px)';
        if (isBoth) {
          e.currentTarget.style.borderColor = 'rgba(167,139,250,0.65)';
          e.currentTarget.style.boxShadow = '0 0 36px rgba(139,92,246,0.28), 0 8px 32px rgba(0,0,0,0.6)';
        } else {
          e.currentTarget.style.borderColor = 'rgba(99,102,241,0.38)';
          e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.55), 0 0 14px rgba(99,102,241,0.08)';
        }
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'translateY(0)';
        if (isBoth) {
          e.currentTarget.style.borderColor = 'rgba(129,140,248,0.48)';
          e.currentTarget.style.boxShadow = '0 0 28px rgba(99,102,241,0.18), 0 4px 24px rgba(0,0,0,0.55)';
        } else {
          e.currentTarget.style.borderColor = 'rgba(51,65,110,0.55)';
          e.currentTarget.style.boxShadow = 'none';
        }
      }}
    >
      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          {/* Family icon */}
          <div className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: accent.bg, color: accent.color, border: `1px solid ${accent.color}33` }}>
            <Music className="h-5 w-5" />
          </div>

          <div>
            {/* Badges row */}
            <div className="flex items-center gap-1.5 flex-wrap mb-0.5">
              {/* Rank */}
              <span className="text-[10px] font-extrabold px-1.5 py-0.5 rounded-md"
                style={{ background: 'rgba(30,36,72,0.8)', color: '#475569', border: '1px solid rgba(51,65,110,0.5)' }}>
                #{rank}
              </span>

              {/* Source badge */}
              <span className="flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full"
                style={{
                  background: src.bg,
                  color: src.color,
                  border: src.border || 'none',
                  boxShadow: isBoth ? '0 0 10px rgba(99,102,241,0.3)' : 'none',
                }}>
                {src.icon}
                {src.label}
              </span>

              {/* Dominant badge */}
              {isDominant && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full flex items-center gap-0.5"
                  style={{ background: 'rgba(251,191,36,0.15)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.3)' }}>
                  <Sparkles className="h-2.5 w-2.5" /> Lead
                </span>
              )}

              {/* Family */}
              <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md"
                style={{ background: accent.bg, color: accent.color }}>
                {instrument.family_name}
              </span>
            </div>

            {/* Instrument name */}
            <h3 className="font-black text-base tracking-tight" style={{ color: isBoth ? '#c4b5fd' : '#e2e8f0' }}>
              {instrument.name}
            </h3>
          </div>
        </div>

        {/* Confidence score */}
        <div className="text-right shrink-0">
          <div className="font-black text-2xl" style={{ color: isHighConf ? accent.color : '#94a3b8' }}>
            {instrument.confidence}%
          </div>
          <div className="text-[10px] font-bold uppercase tracking-wider"
            style={{ color: isDominant ? '#fbbf24' : isHighConf ? accent.color : '#475569' }}>
            {instrument.intensity}
          </div>
        </div>
      </div>

      {/* ── Main confidence bar ── */}
      <div className="w-full h-1.5 rounded-full mb-1 overflow-hidden" style={{ background: 'rgba(30,36,72,0.8)' }}>
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${Math.min(100, Math.max(6, instrument.confidence))}%`, background: accent.bar }} />
      </div>

      {/* ── Per-model mini bars (if ensemble data) ── */}
      {hasBreakdown && (
        <div className="flex gap-3 mt-2 mb-3">
          <div className="flex-1">
            <div className="flex justify-between text-[10px] mb-0.5">
              <span className="flex items-center gap-0.5" style={{ color: '#38bdf8' }}>
                <Zap className="h-2.5 w-2.5" /> CNN14
              </span>
              <span className="font-bold" style={{ color: '#38bdf8' }}>{instrument.cnn14_confidence}%</span>
            </div>
            <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(30,36,72,0.8)' }}>
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, instrument.cnn14_confidence)}%`, background: 'linear-gradient(90deg,#0ea5e9,#38bdf8)' }} />
            </div>
          </div>
          <div className="flex-1">
            <div className="flex justify-between text-[10px] mb-0.5">
              <span className="flex items-center gap-0.5" style={{ color: '#a78bfa' }}>
                <Brain className="h-2.5 w-2.5" /> CNN
              </span>
              <span className="font-bold" style={{ color: '#a78bfa' }}>{instrument.openmic_confidence}%</span>
            </div>
            <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(30,36,72,0.8)' }}>
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, instrument.openmic_confidence)}%`, background: 'linear-gradient(90deg,#7c3aed,#a78bfa)' }} />
            </div>
          </div>
        </div>
      )}

      {/* ── Description (short) ── */}
      {instrument.description && (
        <p className="text-xs leading-relaxed mb-3" style={{ color: '#64748b' }}>
          {instrument.description}
        </p>
      )}

      {/* ── Characteristics chips ── */}
      {instrument.characteristics?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {instrument.characteristics.slice(0, 3).map((c, i) => (
            <span key={i} className="text-[10px] font-medium px-2 py-0.5 rounded-md"
              style={{ background: 'rgba(30,36,72,0.7)', color: '#64748b', border: '1px solid rgba(51,65,110,0.4)' }}>
              {c}
            </span>
          ))}
        </div>
      )}

      {/* ── Footer ── */}
      <div className="mt-auto pt-3 space-y-1" style={{ borderTop: '1px solid rgba(51,65,110,0.35)' }}>

        {/* Breakdown toggle */}
        {hasBreakdown && (
          <>
            <button
              onClick={() => setShowBreakdown(!showBreakdown)}
              className="w-full flex items-center justify-between text-xs font-semibold py-1 transition-colors"
              style={{ color: '#64748b' }}
            >
              <span className="flex items-center gap-1.5">
                <GitMerge className="h-3.5 w-3.5" style={{ color: '#818cf8' }} />
                Ensemble Breakdown
              </span>
              {showBreakdown ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
            {showBreakdown && (
              <div className="p-3 rounded-xl text-xs space-y-2"
                style={{ background: 'rgba(10,13,26,0.6)', border: '1px solid rgba(51,65,110,0.4)' }}>
                <div className="flex justify-between">
                  <span style={{ color: '#94a3b8' }}>CNN14</span>
                  <span className="font-bold" style={{ color: '#38bdf8' }}>{instrument.cnn14_confidence}%</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: '#94a3b8' }}>CNN (trained on OpenMIC)</span>
                  <span className="font-bold" style={{ color: '#a78bfa' }}>{instrument.openmic_confidence}%</span>
                </div>
                <div className="flex justify-between pt-1" style={{ borderTop: '1px solid rgba(51,65,110,0.35)' }}>
                  <span className="font-semibold" style={{ color: '#e2e8f0' }}>Fused Score</span>
                  <span className="font-black" style={{ color: accent.color }}>{instrument.confidence}%</span>
                </div>
              </div>
            )}
          </>
        )}

        {/* Fun fact toggle */}
        {instrument.fun_fact && (
          <>
            <button
              onClick={() => setShowFact(!showFact)}
              className="w-full flex items-center justify-between text-xs font-semibold py-1"
              style={{ color: '#64748b' }}
            >
              <span className="flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5" style={{ color: '#fbbf24' }} />
                Historical Fact
              </span>
              {showFact ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
            {showFact && (
              <div className="p-3 rounded-xl text-xs leading-relaxed"
                style={{ background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.15)', color: '#a5b4fc' }}>
                {instrument.fun_fact}
              </div>
            )}
          </>
        )}

        {/* Footer meta */}
        <div className="flex items-center justify-between text-[11px] pt-1" style={{ color: '#334155' }}>
          <span>Active in {instrument.active_segments_count} segment{instrument.active_segments_count !== 1 ? 's' : ''}</span>
          <span className="font-mono text-[10px]">Score: {instrument.raw_score}</span>
        </div>
      </div>
    </div>
  );
}
