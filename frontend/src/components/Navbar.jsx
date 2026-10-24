import React from 'react';
import { Music2, Cpu, Zap } from 'lucide-react';

export default function Navbar({ healthInfo }) {
  return (
    <header className="sticky top-0 z-40" style={{
      background: 'rgba(10,13,26,0.85)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(99,102,241,0.18)',
      boxShadow: '0 1px 24px rgba(0,0,0,0.4)'
    }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl flex items-center justify-center text-white"
              style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', boxShadow: '0 0 18px rgba(99,102,241,0.45)' }}>
              <Music2 className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-black text-xl tracking-tight text-white">
                  AI <span style={{ background: 'linear-gradient(90deg,#818cf8,#a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Harmonica</span>
                </span>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold border"
                  style={{ background: 'rgba(99,102,241,0.15)', borderColor: 'rgba(129,140,248,0.35)', color: '#a5b4fc' }}>
                  CNN14 + CNN Ensemble
                </span>
              </div>
              <p className="hidden sm:block text-[11px] font-medium" style={{ color: '#64748b' }}>
                by <span style={{ color: '#818cf8', fontWeight: 700 }}>Eliiviz</span> · Deep Audio Instrument Classifier
              </p>
            </div>
          </div>

          {/* Device Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs"
            style={{ background: 'rgba(18,24,52,0.72)', border: '1px solid rgba(99,102,241,0.2)' }}>
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ background: '#34d399' }} />
              <span className="relative inline-flex rounded-full h-2 w-2"
                style={{ background: '#34d399' }} />
            </span>
            <span className="font-medium" style={{ color: '#94a3b8' }}>Model Ready:</span>
            <span className="font-bold uppercase tracking-wide text-[11px]" style={{ color: '#e2e8f0' }}>
              {healthInfo?.device || 'CPU'}
            </span>
          </div>

        </div>
      </div>
    </header>
  );
}
