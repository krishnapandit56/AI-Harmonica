import React, { useState, useRef } from 'react';
import { UploadCloud, Play, Sparkles, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

export default function AudioDropzone({
  onAnalyzeFile,
  onAnalyzeDemo,
  isAnalyzing,
  analysisStep,
  selectedFileName,
  demos,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [dragError, setDragError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragOver(false); };
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragOver(false); setDragError(null);
    if (e.dataTransfer.files?.length) validateAndUpload(e.dataTransfer.files[0]);
  };
  const handleFileChange = (e) => {
    if (e.target.files?.length) validateAndUpload(e.target.files[0]);
  };
  const validateAndUpload = (file) => {
    const valid = ['.mp3','.wav','.flac','.ogg','.m4a','.aac','.webm','.wma'];
    const ok = valid.some(ext => file.name.toLowerCase().endsWith(ext)) || file.type.startsWith('audio/');
    if (!ok) { setDragError('Please upload a valid audio file (.mp3, .wav, .flac, .ogg, .m4a, .aac)'); return; }
    setDragError(null);
    onAnalyzeFile(file);
  };

  return (
    <div className="w-full">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isAnalyzing && fileInputRef.current?.click()}
        className={`relative cursor-pointer rounded-2xl p-8 text-center transition-all duration-300 ${isAnalyzing ? 'pointer-events-none' : ''}`}
        style={{
          background: isDragOver ? 'rgba(99,102,241,0.10)' : 'rgba(15,20,42,0.60)',
          border: `2px dashed ${isDragOver ? 'rgba(129,140,248,0.7)' : 'rgba(51,65,110,0.6)'}`,
          boxShadow: isDragOver ? '0 0 32px rgba(99,102,241,0.15)' : 'none',
        }}
      >
        <input ref={fileInputRef} type="file" accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
          className="hidden" onChange={handleFileChange} disabled={isAnalyzing} />

        {isAnalyzing ? (
          <div className="flex flex-col items-center py-4">
            <div className="relative flex items-center justify-center h-18 w-18 mb-5">
              <div className="absolute inset-0 rounded-full animate-ping opacity-30"
                style={{ background: 'rgba(99,102,241,0.4)' }} />
              <div className="h-14 w-14 rounded-2xl flex items-center justify-center text-white"
                style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 0 24px rgba(99,102,241,0.5)' }}>
                <Loader2 className="h-7 w-7 animate-spin" />
              </div>
            </div>
            <h3 className="font-black text-xl mb-2" style={{ color: '#e2e8f0' }}>Analyzing Musical Instruments...</h3>
            <div className="w-full max-w-sm h-1.5 rounded-full mb-3 overflow-hidden" style={{ background: 'rgba(30,36,72,0.8)' }}>
              <div className="h-full rounded-full animate-pulse"
                style={{ width: '72%', background: 'linear-gradient(90deg,#6366f1,#8b5cf6)' }} />
            </div>
            <p className="text-sm font-medium" style={{ color: '#818cf8' }}>
              {analysisStep || 'Running CNN14 neural audio inference...'}
            </p>
          </div>
        ) : (
          <>
            <div className="h-16 w-16 rounded-2xl flex items-center justify-center mx-auto mb-5 transition-all duration-200"
              style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', color: '#818cf8' }}>
              <UploadCloud className="h-8 w-8" />
            </div>
            <h3 className="font-black text-xl mb-2 tracking-tight" style={{ color: '#e2e8f0' }}>
              Drop your song here, or{' '}
              <span style={{ color: '#818cf8', textDecoration: 'underline', textUnderlineOffset: 4 }}>browse files</span>
            </h3>
            <p className="text-sm mb-5" style={{ color: '#64748b' }}>
              Upload any audio track. Our CNN ensemble will detect every active instrument.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs font-bold">
              {['MP3','WAV','FLAC','M4A','OGG','AAC'].map(f => (
                <span key={f} className="px-2.5 py-1 rounded-md"
                  style={{ background: 'rgba(30,36,72,0.8)', border: '1px solid rgba(51,65,110,0.6)', color: '#94a3b8' }}>
                  {f}
                </span>
              ))}
            </div>
            {selectedFileName && (
              <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.25)', color: '#6ee7b7' }}>
                <CheckCircle2 className="h-4 w-4" />
                <span>Loaded: {selectedFileName}</span>
              </div>
            )}
          </>
        )}

        {dragError && (
          <div className="mt-4 flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#fca5a5' }}>
            <AlertCircle className="h-4 w-4" />
            <span>{dragError}</span>
          </div>
        )}
      </div>

      {/* Demo tracks */}
      <div className="mt-5 pt-5" style={{ borderTop: '1px solid rgba(51,65,110,0.4)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" style={{ color: '#fbbf24' }} />
            <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: '#64748b' }}>
              Or Try A 1-Click Demo Track
            </h4>
          </div>
          <span className="text-[11px]" style={{ color: '#475569' }}>Instant AI analysis without uploading</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {demos && demos.length > 0 ? demos.map((demo) => (
            <button
              key={demo.id}
              type="button"
              onClick={(e) => { e.stopPropagation(); onAnalyzeDemo(demo.id); }}
              disabled={isAnalyzing}
              className="group flex items-start gap-3 p-4 rounded-xl text-left transition-all duration-200 disabled:opacity-50"
              style={{
                background: 'rgba(15,20,42,0.80)',
                border: '1px solid rgba(51,65,110,0.5)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'rgba(99,102,241,0.45)';
                e.currentTarget.style.background = 'rgba(22,28,60,0.90)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'rgba(51,65,110,0.5)';
                e.currentTarget.style.background = 'rgba(15,20,42,0.80)';
              }}
            >
              <div className="h-9 w-9 rounded-lg flex items-center justify-center shrink-0 transition-all duration-200"
                style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}>
                <Play className="h-4 w-4 fill-current ml-0.5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-semibold text-sm truncate" style={{ color: '#e2e8f0' }}>{demo.name}</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0"
                    style={{ background: 'rgba(30,36,72,0.8)', color: '#64748b', border: '1px solid rgba(51,65,110,0.5)' }}>
                    {demo.duration}
                  </span>
                </div>
                <p className="text-xs line-clamp-1" style={{ color: '#475569' }}>{demo.description}</p>
              </div>
            </button>
          )) : (
            <div className="col-span-2 text-center text-xs py-2" style={{ color: '#475569' }}>
              Loading demo clips...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
