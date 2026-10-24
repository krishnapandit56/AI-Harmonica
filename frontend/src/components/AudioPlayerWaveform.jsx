import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Music, Clock } from 'lucide-react';

export default function AudioPlayerWaveform({ audioUrl, waveformPeaks, durationSeconds, fileName }) {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationSeconds || 0);
  const [volume, setVolume] = useState(0.85);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = isMuted ? 0 : volume;
  }, [volume, isMuted]);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) audioRef.current.currentTime = 0;
  }, [audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) { audioRef.current.pause(); setIsPlaying(false); }
    else { audioRef.current.play().catch(() => {}); setIsPlaying(true); }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      if (audioRef.current.duration && !isNaN(audioRef.current.duration))
        setDuration(audioRef.current.duration);
    }
  };

  const handleEnded = () => { setIsPlaying(false); setCurrentTime(0); };

  const handleSeek = (pct) => {
    if (!audioRef.current || !duration) return;
    audioRef.current.currentTime = pct * duration;
    setCurrentTime(pct * duration);
  };

  const restartAudio = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
    audioRef.current.play();
    setIsPlaying(true);
  };

  const fmt = (s) => {
    if (isNaN(s) || s < 0) return '0:00';
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  };

  const currentPct = duration > 0 ? currentTime / duration : 0;
  const peaks = (waveformPeaks?.length > 0) ? waveformPeaks : Array(100).fill(0.4);

  return (
    <div className="rounded-2xl p-5"
      style={{ background: 'rgba(15,20,42,0.80)', border: '1px solid rgba(51,65,110,0.55)' }}>
      {audioUrl && (
        <audio ref={audioRef} src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={() => audioRef.current && setDuration(audioRef.current.duration || durationSeconds)}
          onEnded={handleEnded}
        />
      )}

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', color: '#818cf8' }}>
            <Music className="h-4.5 w-4.5 h-[18px] w-[18px]" />
          </div>
          <div>
            <h4 className="font-bold text-sm truncate max-w-xs sm:max-w-md" style={{ color: '#e2e8f0' }}>
              {fileName || 'Loaded Audio Track'}
            </h4>
            <div className="flex items-center gap-2 text-xs" style={{ color: '#475569' }}>
              <Clock className="h-3 w-3" />
              {fmt(currentTime)} / {fmt(duration || durationSeconds)}
              <span>·</span>
              <span style={{ color: '#34d399', fontWeight: 600 }}>32 kHz Resampled</span>
            </div>
          </div>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2">
          <button onClick={() => setIsMuted(!isMuted)}
            className="p-1.5 rounded-lg transition-colors"
            style={{ color: isMuted ? '#ef4444' : '#64748b' }}>
            {isMuted || volume === 0 ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>
          <input type="range" min="0" max="1" step="0.05"
            value={isMuted ? 0 : volume}
            onChange={(e) => { setVolume(parseFloat(e.target.value)); setIsMuted(false); }}
            className="w-20 sm:w-24"
            style={{ background: `linear-gradient(90deg, #6366f1 ${(isMuted ? 0 : volume) * 100}%, rgba(30,36,72,0.8) ${(isMuted ? 0 : volume) * 100}%)` }}
          />
        </div>
      </div>

      {/* Waveform */}
      <div
        className="relative h-16 w-full rounded-xl p-2 flex items-center gap-[2px] cursor-pointer overflow-hidden select-none"
        style={{ background: 'rgba(10,13,26,0.6)', border: '1px solid rgba(51,65,110,0.4)' }}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          handleSeek(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)));
        }}
      >
        {/* Progress overlay */}
        <div className="absolute top-0 bottom-0 left-0 pointer-events-none"
          style={{ width: `${currentPct * 100}%`, background: 'rgba(99,102,241,0.08)' }} />
        {/* Playhead */}
        <div className="absolute top-0 bottom-0 w-0.5 pointer-events-none z-10"
          style={{ left: `${currentPct * 100}%`, background: '#8b5cf6', boxShadow: '0 0 8px rgba(139,92,246,0.8)' }} />
        {/* Bars */}
        {peaks.map((peak, idx) => {
          const isPassed = (idx / peaks.length) <= currentPct;
          return (
            <div key={idx} className="flex-1 flex items-center justify-center h-full">
              <div
                style={{
                  height: `${Math.max(4, peak * 48)}px`,
                  width: '100%',
                  borderRadius: '2px',
                  background: isPassed
                    ? 'linear-gradient(180deg,#8b5cf6,#6366f1)'
                    : 'rgba(51,65,110,0.6)',
                  transition: 'background 0.08s',
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mt-4 pt-4"
        style={{ borderTop: '1px solid rgba(51,65,110,0.35)' }}>
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay} disabled={!audioUrl}
            className="h-11 w-11 rounded-xl flex items-center justify-center text-white transition-all duration-150 active:scale-95 disabled:opacity-40"
            style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 0 18px rgba(99,102,241,0.45)' }}
          >
            {isPlaying ? <Pause className="h-5 w-5 fill-current" /> : <Play className="h-5 w-5 fill-current ml-0.5" />}
          </button>
          <button onClick={restartAudio} disabled={!audioUrl}
            className="p-2 rounded-lg transition-colors disabled:opacity-40"
            style={{ color: '#64748b' }}>
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
        <div className="text-right">
          <span className="text-xs font-semibold" style={{ color: '#64748b' }}>Click waveform to seek</span>
          <p className="text-[11px]" style={{ color: '#334155' }}>Synced with instrument timeline</p>
        </div>
      </div>
    </div>
  );
}
