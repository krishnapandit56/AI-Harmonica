import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import {
  Sparkles,
  Layers,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  GitMerge,
  Music,
  Star,
} from 'lucide-react';

import Navbar from './components/Navbar';
import AudioDropzone from './components/AudioDropzone';
import AudioPlayerWaveform from './components/AudioPlayerWaveform';
import SongMetrics from './components/SongMetrics';
import InstrumentCard from './components/InstrumentCard';
import TimelineVisualizer from './components/TimelineVisualizer';
import EnsembleWeightPanel from './components/EnsembleWeightPanel';

export default function App() {
  const [healthInfo, setHealthInfo]       = useState(null);
  const [demos, setDemos]                 = useState([]);
  const [audioUrl, setAudioUrl]           = useState(null);
  const [fileName, setFileName]           = useState('');
  const [isAnalyzing, setIsAnalyzing]     = useState(false);
  const [analysisStep, setAnalysisStep]   = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage]   = useState(null);
  const [familyFilter, setFamilyFilter]   = useState('all');
  const [ensembleWeights, setEnsembleWeights] = useState({ cnn14: 0.60, openmic: 0.40 });

  const handleWeightsChange = ({ cnn14, openmic }) => {
    setEnsembleWeights({ cnn14: parseFloat(cnn14.toFixed(2)), openmic: parseFloat(openmic.toFixed(2)) });
  };

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setHealthInfo).catch(() => {});
    fetch('/api/demos').then(r => r.json()).then(setDemos).catch(() => {});
  }, []);

  const triggerCelebration = () => {
    try {
      confetti({ particleCount: 60, spread: 70, origin: { y: 0.6 },
        colors: ['#6366f1','#8b5cf6','#38bdf8','#34d399','#fbbf24'] });
    } catch (_) {}
  };

  const handleAnalyzeFile = async (file) => {
    setErrorMessage(null); setIsAnalyzing(true);
    setFileName(file.name); setAudioUrl(URL.createObjectURL(file));

    setAnalysisStep('Resampling audio to 32 kHz...');
    const t1 = setTimeout(() => setAnalysisStep('Extracting 64-band Log-Mel Spectrograms...'), 900);
    const t2 = setTimeout(() => setAnalysisStep('CNN14 forward pass across 527 AudioSet classes...'), 2000);
    const t3 = setTimeout(() => setAnalysisStep('CNN (trained on OpenMIC) instrument tagging...'), 3200);

    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch(
        `/api/analyze?cnn14_weight=${ensembleWeights.cnn14}&openmic_weight=${ensembleWeights.openmic}`,
        { method: 'POST', body: fd }
      );
      if (!res.ok) throw new Error((await res.json()).detail || 'Analysis failed');
      setAnalysisResult(await res.json());
      triggerCelebration();
    } catch (err) {
      setErrorMessage(err.message || 'Failed to analyze audio.');
    } finally {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
      setIsAnalyzing(false); setAnalysisStep('');
    }
  };

  const handleAnalyzeDemo = async (demoId) => {
    setErrorMessage(null); setIsAnalyzing(true);
    const demoObj = demos.find(d => d.id === demoId);
    setFileName(demoObj ? demoObj.name : `${demoId}.wav`);
    setAudioUrl(`/api/demos/${demoId}`);

    setAnalysisStep('Loading demo clip...');
    const t1 = setTimeout(() => setAnalysisStep('Computing 64-mel spectrogram...'), 700);
    const t2 = setTimeout(() => setAnalysisStep('CNN ensemble inference...'), 1600);

    try {
      const res = await fetch(
        `/api/analyze-demo/${demoId}?cnn14_weight=${ensembleWeights.cnn14}&openmic_weight=${ensembleWeights.openmic}`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error((await res.json()).detail || 'Demo analysis failed');
      setAnalysisResult(await res.json());
      triggerCelebration();
    } catch (err) {
      setErrorMessage(err.message || 'Failed to analyze demo track.');
    } finally {
      clearTimeout(t1); clearTimeout(t2);
      setIsAnalyzing(false); setAnalysisStep('');
    }
  };

  const handleReset = () => {
    setAudioUrl(null); setFileName(''); setAnalysisResult(null);
    setErrorMessage(null); setFamilyFilter('all');
  };

  const detectedList = analysisResult?.detected_instruments || [];

  // Split: both models first, then others
  const bothModels   = detectedList.filter(i => i.source === 'ensemble');
  const otherModels  = detectedList.filter(i => i.source !== 'ensemble');

  const filteredBoth  = familyFilter === 'all' ? bothModels  : bothModels.filter(i => i.family === familyFilter);
  const filteredOther = familyFilter === 'all' ? otherModels : otherModels.filter(i => i.family === familyFilter);
  const filteredAll   = [...filteredBoth, ...filteredOther];

  return (
    <div className="min-h-screen flex flex-col dark-mesh" style={{ color: '#e2e8f0' }}>
      <Navbar healthInfo={healthInfo} />

      <main className="flex-1 w-full max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-10 py-8">

        {/* ── PRE-ANALYSIS VIEW ── */}
        {!analysisResult && (
          <div className="space-y-8 anim-in">

            {/* Hero */}
            <div className="text-center max-w-2xl mx-auto pt-4">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold mb-4"
                style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', color: '#a5b4fc' }}>
                <Sparkles className="h-3.5 w-3.5" style={{ color: '#fbbf24' }} />
                Powered by CNN14 + CNN (trained on OpenMIC) Ensemble
              </div>
              <h1 className="text-4xl sm:text-5xl font-black tracking-tight leading-tight mb-4" style={{ color: '#f1f5f9' }}>
                Identify Every Musical<br />
                Instrument in{' '}
                <span style={{ background: 'linear-gradient(90deg,#818cf8,#a78bfa,#38bdf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  Any Song
                </span>
              </h1>
              <p className="text-base leading-relaxed" style={{ color: '#64748b' }}>
                Upload an audio track and our CNN ensemble will break down every acoustic &amp; electric instrument with confidence scores from both AI models.
              </p>
            </div>

            {/* Error */}
            {errorMessage && (
              <div className="max-w-xl mx-auto p-4 rounded-2xl flex items-start gap-3 text-sm"
                style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
                <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" style={{ color: '#ef4444' }} />
                <div><span className="font-bold block">Analysis Failed</span>{errorMessage}</div>
              </div>
            )}

            {/* Two-column: weight panel left, dropzone right */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 max-w-6xl mx-auto">
              <div className="lg:col-span-2">
                <EnsembleWeightPanel
                  cnn14Weight={ensembleWeights.cnn14}
                  openMicWeight={ensembleWeights.openmic}
                  onWeightsChange={handleWeightsChange}
                  ensembleInfo={null}
                />
              </div>
              <div className="lg:col-span-3">
                <AudioDropzone
                  onAnalyzeFile={handleAnalyzeFile}
                  onAnalyzeDemo={handleAnalyzeDemo}
                  isAnalyzing={isAnalyzing}
                  analysisStep={analysisStep}
                  selectedFileName={fileName}
                  demos={demos}
                />
              </div>
            </div>
          </div>
        )}

        {/* ── RESULTS VIEW ── */}
        {analysisResult && (
          <div className="space-y-7 anim-in">

            {/* Action bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl"
              style={{ background: 'rgba(15,20,42,0.80)', border: '1px solid rgba(99,102,241,0.22)' }}>
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-xl flex items-center justify-center"
                  style={{ background: 'rgba(52,211,153,0.12)', border: '1px solid rgba(52,211,153,0.25)', color: '#34d399' }}>
                  <CheckCircle2 className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="font-black text-lg" style={{ color: '#f1f5f9' }}>Analysis Complete</h2>
                  <p className="text-xs" style={{ color: '#64748b' }}>
                    Ensemble: {Math.round(ensembleWeights.cnn14 * 100)}% CNN14 + {Math.round(ensembleWeights.openmic * 100)}% CNN (OpenMIC)
                    &nbsp;·&nbsp;{analysisResult.summary.total_instruments_detected} instruments detected
                  </p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all duration-150"
                style={{ background: 'rgba(30,36,72,0.8)', color: '#94a3b8', border: '1px solid rgba(51,65,110,0.5)' }}
              >
                <RotateCcw className="h-3.5 w-3.5" /> Analyze Another Song
              </button>
            </div>

            {/* 3-column upper layout: player | metrics | weight panel */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-5">
                <AudioPlayerWaveform
                  audioUrl={audioUrl}
                  waveformPeaks={analysisResult.waveform}
                  durationSeconds={analysisResult.summary.duration_seconds}
                  fileName={fileName}
                />
              </div>
              <div className="lg:col-span-4 flex flex-col gap-4">
                <SongMetrics summary={analysisResult.summary} tags={analysisResult.tags} />
              </div>
              <div className="lg:col-span-3">
                <EnsembleWeightPanel
                  cnn14Weight={ensembleWeights.cnn14}
                  openMicWeight={ensembleWeights.openmic}
                  onWeightsChange={handleWeightsChange}
                  ensembleInfo={analysisResult.ensemble_info}
                />
              </div>
            </div>

            {/* Timeline */}
            <TimelineVisualizer
              detectedInstruments={analysisResult.detected_instruments}
              durationSeconds={analysisResult.summary.duration_seconds}
              timelineSegments={analysisResult.timeline_segments}
            />

            {/* Error (results page) */}
            {errorMessage && (
              <div className="p-4 rounded-2xl flex items-start gap-3 text-sm"
                style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
                <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" style={{ color: '#ef4444' }} />
                <div><span className="font-bold block">Error</span>{errorMessage}</div>
              </div>
            )}

            {/* ── Instrument Results ── */}
            <div>
              {/* Filter bar */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 mb-6"
                style={{ borderBottom: '1px solid rgba(51,65,110,0.45)' }}>
                <div>
                  <h3 className="font-black text-2xl tracking-tight" style={{ color: '#f1f5f9' }}>
                    Identified Instruments ({filteredAll.length})
                  </h3>
                  <p className="text-xs mt-0.5" style={{ color: '#64748b' }}>
                    Ranked by ensemble confidence score
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    onClick={() => setFamilyFilter('all')}
                    className="px-3 py-1.5 rounded-xl text-xs font-bold transition-all duration-150"
                    style={familyFilter === 'all'
                      ? { background: 'linear-gradient(90deg,#6366f1,#8b5cf6)', color: '#fff', boxShadow: '0 0 14px rgba(99,102,241,0.4)' }
                      : { background: 'rgba(15,20,42,0.80)', color: '#94a3b8', border: '1px solid rgba(51,65,110,0.5)' }}
                  >
                    All ({detectedList.length})
                  </button>
                  {analysisResult.families.map((fam) => (
                    <button
                      key={fam.family}
                      onClick={() => setFamilyFilter(fam.family)}
                      className="px-3 py-1.5 rounded-xl text-xs font-bold transition-all duration-150"
                      style={familyFilter === fam.family
                        ? { background: 'rgba(99,102,241,0.25)', color: '#a5b4fc', border: '1px solid rgba(99,102,241,0.4)' }
                        : { background: 'rgba(15,20,42,0.80)', color: '#64748b', border: '1px solid rgba(51,65,110,0.45)' }}
                    >
                      {fam.name} ({fam.count})
                    </button>
                  ))}
                </div>
              </div>

              {/* ── Section 1: Detected by BOTH models ── */}
              {filteredBoth.length > 0 && (
                <div className="mb-8">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="h-7 w-7 rounded-lg flex items-center justify-center"
                      style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 0 14px rgba(99,102,241,0.45)' }}>
                      <GitMerge className="h-3.5 w-3.5 text-white" />
                    </div>
                    <h4 className="font-black text-lg" style={{ color: '#a5b4fc' }}>
                      Confirmed by Both Models
                    </h4>
                    <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                      style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.3)' }}>
                      {filteredBoth.length} instrument{filteredBoth.length !== 1 ? 's' : ''}
                    </span>
                    <div className="flex-1 h-px" style={{ background: 'rgba(99,102,241,0.2)' }} />
                    <Star className="h-4 w-4" style={{ color: '#fbbf24' }} />
                    <span className="text-xs font-bold" style={{ color: '#64748b' }}>Highest confidence</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filteredBoth.map((instrument, idx) => (
                      <InstrumentCard key={instrument.id} instrument={instrument} rank={idx + 1} isBothSection />
                    ))}
                  </div>
                </div>
              )}

              {/* ── Section 2: Other instruments ── */}
              {filteredOther.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <div className="h-7 w-7 rounded-lg flex items-center justify-center"
                      style={{ background: 'rgba(51,65,110,0.6)', border: '1px solid rgba(51,65,110,0.6)' }}>
                      <Music className="h-3.5 w-3.5" style={{ color: '#94a3b8' }} />
                    </div>
                    <h4 className="font-black text-lg" style={{ color: '#94a3b8' }}>
                      Additional Instruments Detected
                    </h4>
                    <div className="flex-1 h-px" style={{ background: 'rgba(51,65,110,0.4)' }} />
                    <span className="text-xs" style={{ color: '#475569' }}>Detected by one model</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filteredOther.map((instrument, idx) => (
                      <InstrumentCard key={instrument.id} instrument={instrument} rank={filteredBoth.length + idx + 1} />
                    ))}
                  </div>
                </div>
              )}

              {filteredAll.length === 0 && (
                <div className="text-center py-14 rounded-2xl"
                  style={{ background: 'rgba(15,20,42,0.60)', border: '1px solid rgba(51,65,110,0.4)', color: '#475569' }}>
                  No instruments found for the selected family filter.
                </div>
              )}
            </div>

            {/* Family breakdown */}
            {analysisResult.families?.length > 0 && (
              <div className="rounded-2xl p-6"
                style={{ background: 'rgba(15,20,42,0.80)', border: '1px solid rgba(51,65,110,0.55)' }}>
                <div className="flex items-center gap-2 mb-4">
                  <Layers className="h-4 w-4" style={{ color: '#818cf8' }} />
                  <h4 className="font-black text-lg" style={{ color: '#f1f5f9' }}>Family Breakdown</h4>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {analysisResult.families.map((fam) => (
                    <div key={fam.family} className="p-4 rounded-xl"
                      style={{ background: 'rgba(10,13,26,0.6)', border: '1px solid rgba(51,65,110,0.4)' }}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-bold text-sm" style={{ color: '#e2e8f0' }}>{fam.name}</span>
                        <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                          style={{ background: 'rgba(99,102,241,0.12)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.2)' }}>
                          {fam.count}
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed mb-2" style={{ color: '#475569' }}>{fam.description}</p>
                      <div className="text-[11px]" style={{ color: '#334155' }}>{fam.instruments?.join(', ')}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto py-5 text-center text-xs"
        style={{ borderTop: '1px solid rgba(51,65,110,0.4)', color: '#334155', background: 'rgba(10,13,26,0.6)' }}>
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-black" style={{ color: '#818cf8' }}>AI Harmonica</span>
            <span>•</span>
            <span style={{ color: '#475569' }}>by <span style={{ color: '#a78bfa', fontWeight: 700 }}>Eliiviz</span></span>
            <span>•</span>
            <span style={{ color: '#334155' }}>CNN Ensemble Instrument Classifier</span>
          </div>
          <div style={{ color: '#334155' }}>Built with Vite, React.js, Tailwind CSS &amp; FastAPI</div>
        </div>
      </footer>
    </div>
  );
}
