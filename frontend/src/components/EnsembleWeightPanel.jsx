import React from 'react';
import { GitMerge, Zap, Brain } from 'lucide-react';

export default function EnsembleWeightPanel({ cnn14Weight, openMicWeight, onWeightsChange, ensembleInfo }) {
  const totalW = cnn14Weight + openMicWeight;
  const cnn14Pct = Math.round((cnn14Weight / totalW) * 100);
  const cnnPct   = 100 - cnn14Pct;

  const handleChange = (val) => {
    const v = parseFloat(val);
    onWeightsChange({ cnn14: v, openmic: parseFloat((1.0 - v).toFixed(2)) });
  };

  return (
    <div className="rounded-2xl p-5"
      style={{
        background: 'rgba(15,20,42,0.80)',
        border: '1px solid rgba(99,102,241,0.22)',
      }}>

      {/* Header */}
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 0 14px rgba(99,102,241,0.4)' }}>
            <GitMerge className="h-4 w-4 text-white" />
          </div>
          <div>
            <h4 className="font-bold text-sm" style={{ color: '#e2e8f0' }}>Ensemble Model Weights</h4>
            <p className="text-[11px]" style={{ color: '#64748b' }}>Adjust how much each model influences the final result</p>
          </div>
        </div>
        <span className="text-[11px] px-3 py-1 rounded-xl font-semibold"
          style={{ background: 'rgba(99,102,241,0.12)', color: '#a5b4fc', border: '1px solid rgba(99,102,241,0.2)' }}>
          Weighted Score Fusion
        </span>
      </div>

      {/* Model info cards */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {/* CNN14 */}
        <div className="p-3 rounded-xl" style={{ background: 'rgba(14,165,233,0.08)', border: '1px solid rgba(14,165,233,0.18)' }}>
          <div className="flex items-center gap-2 mb-1">
            <div className="h-5 w-5 rounded-md flex items-center justify-center" style={{ background: '#0ea5e9' }}>
              <Zap className="h-3 w-3 text-white" />
            </div>
            <span className="text-xs font-bold" style={{ color: '#38bdf8' }}>CNN14</span>
          </div>
          <div className="text-[11px]" style={{ color: '#7dd3fc' }}>Google AudioSet · 527 classes · 2M+ clips</div>
          <div className="text-[11px] mt-0.5" style={{ color: '#0ea5e9' }}>Best for: Broad coverage</div>
        </div>

        {/* CNN (trained on OpenMIC) */}
        <div className="p-3 rounded-xl" style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.18)' }}>
          <div className="flex items-center gap-2 mb-1">
            <div className="h-5 w-5 rounded-md flex items-center justify-center" style={{ background: '#7c3aed' }}>
              <Brain className="h-3 w-3 text-white" />
            </div>
            <span className="text-xs font-bold" style={{ color: '#a78bfa' }}>CNN (trained on OpenMIC)</span>
          </div>
          <div className="text-[11px]" style={{ color: '#c4b5fd' }}>20 instrument classes · Custom trained</div>
          <div className="text-[11px] mt-0.5" style={{ color: '#7c3aed' }}>Best for: Precise recall</div>
        </div>
      </div>

      {/* Balance labels */}
      <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
        <span className="flex items-center gap-1" style={{ color: '#38bdf8' }}>
          <Zap className="h-3 w-3" /> CNN14 — {cnn14Pct}%
        </span>
        <span className="flex items-center gap-1" style={{ color: '#a78bfa' }}>
          CNN (OpenMIC) — {cnnPct}% <Brain className="h-3 w-3" />
        </span>
      </div>

      {/* Visual bar */}
      <div className="relative h-2 rounded-full overflow-hidden mb-2" style={{ background: 'rgba(30,36,72,0.8)' }}>
        <div className="absolute top-0 bottom-0 left-0 rounded-full" style={{ width: `${cnn14Pct}%`, background: 'linear-gradient(90deg,#0ea5e9,#38bdf8)' }} />
        <div className="absolute top-0 bottom-0 right-0 rounded-full" style={{ width: `${cnnPct}%`, background: 'linear-gradient(90deg,#7c3aed,#a78bfa)' }} />
      </div>

      {/* Slider */}
      <input
        type="range"
        min="0.1" max="0.9" step="0.05"
        value={cnn14Weight / (cnn14Weight + openMicWeight)}
        onChange={(e) => handleChange(e.target.value)}
        title={`CNN14: ${cnn14Pct}% | CNN (OpenMIC): ${cnnPct}%`}
      />
      <div className="flex justify-between text-[11px] mt-0.5 px-0.5" style={{ color: '#475569' }}>
        <span>← More CNN (OpenMIC)</span>
        <span>More CNN14 →</span>
      </div>

      {/* Last analysis info */}
      {ensembleInfo && (
        <div className="mt-3 pt-3 flex flex-wrap gap-2 text-[10px]" style={{ borderTop: '1px solid rgba(51,65,110,0.4)' }}>
          <span className="px-2 py-0.5 rounded-md" style={{ background: 'rgba(30,36,72,0.8)', color: '#64748b', border: '1px solid rgba(51,65,110,0.4)' }}>
            Strategy: {ensembleInfo.strategy}
          </span>
          <span className="px-2 py-0.5 rounded-md" style={{ background: 'rgba(30,36,72,0.8)', color: '#64748b', border: '1px solid rgba(51,65,110,0.4)' }}>
            {ensembleInfo.alignment_pairs} aligned label pairs
          </span>
        </div>
      )}
    </div>
  );
}
