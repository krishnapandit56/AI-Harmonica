import React from 'react';
import { Cpu, Layers, Sparkles, Activity, CheckCircle2, Sliders, Waves, ArrowRight } from 'lucide-react';

export default function ModelExplainer() {
  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="bg-white rounded-3xl p-8 border border-slate-200/90 shadow-soft">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-50 text-sky-700 text-xs font-bold border border-sky-200/80 mb-3">
          <Cpu className="h-3.5 w-3.5 text-sky-600" />
          <span>Deep Audio Neural Networks (PANNs)</span>
        </div>
        <h2 className="text-3xl font-display font-extrabold text-slate-900 tracking-tight mb-3">
          Inside the PANNs CNN14 Architecture
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed max-w-3xl">
          PANNs (Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition) was developed to learn rich acoustic representations across 527 AudioSet classes. CNN14 achieves a state-of-the-art mean Average Precision (mAP) of 0.431.
        </p>
      </div>

      {/* 5-Stage Neural Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        
        {/* Stage 1 */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col justify-between">
          <div>
            <div className="h-9 w-9 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center font-bold text-xs mb-3 border border-sky-100">
              01
            </div>
            <h4 className="font-display font-bold text-slate-800 text-sm mb-1">
              Audio Resampling
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Input music tracks are converted to mono at a calibrated 32,000 Hz sample rate, capturing acoustic overtones up to 16 kHz.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-mono font-semibold text-sky-700">
            sr = 32,000 Hz
          </div>
        </div>

        {/* Stage 2 */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col justify-between">
          <div>
            <div className="h-9 w-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-xs mb-3 border border-indigo-100">
              02
            </div>
            <h4 className="font-display font-bold text-slate-800 text-sm mb-1">
              Log-Mel Spectrogram
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Short-Time Fourier Transform (STFT) with 1024 window size and 320 hop size creates a 64-band log-mel energy representation.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-mono font-semibold text-indigo-700">
            64 Mel Bins (50-14k Hz)
          </div>
        </div>

        {/* Stage 3 */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col justify-between">
          <div>
            <div className="h-9 w-9 rounded-xl bg-violet-50 text-violet-600 flex items-center justify-center font-bold text-xs mb-3 border border-violet-100">
              03
            </div>
            <h4 className="font-display font-bold text-slate-800 text-sm mb-1">
              6 Conv Blocks
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              12 convolutional layers (each block: 2x Conv2D + BatchNorm + ReLU) followed by 2x2 average pooling extract hierarchical features.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-mono font-semibold text-violet-700">
            Channels: 64 to 2048
          </div>
        </div>

        {/* Stage 4 */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col justify-between">
          <div>
            <div className="h-9 w-9 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold text-xs mb-3 border border-amber-100">
              04
            </div>
            <h4 className="font-display font-bold text-slate-800 text-sm mb-1">
              Global Pooling
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Global pooling aggregates time-frequency feature maps into a high-dimensional 2,048-dimensional dense audio embedding vector.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-mono font-semibold text-amber-700">
            Embedding: ℝ²⁰⁴⁸
          </div>
        </div>

        {/* Stage 5 */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col justify-between">
          <div>
            <div className="h-9 w-9 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center font-bold text-xs mb-3 border border-emerald-100">
              05
            </div>
            <h4 className="font-display font-bold text-slate-800 text-sm mb-1">
              Multi-Label Sigmoid
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Linear projection layer outputs independent probability scores for each musical instrument concurrently present in the song.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-mono font-semibold text-emerald-700">
            527 AudioSet Classes
          </div>
        </div>

      </div>

      {/* Deep-Dive Specifications */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/90 shadow-soft">
        <h3 className="font-display font-bold text-xl text-slate-900 mb-4">
          Architectural Specifications & Parameters
        </h3>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Model Depth</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">14 Layers (12 Conv + 2 Linear)</div>
            <p className="text-xs text-slate-600 mt-1">Configured with Batch Normalization and ReLU activations.</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Pretrained Dataset</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">Google AudioSet (2M+ Clips)</div>
            <p className="text-xs text-slate-600 mt-1">Trained on 5,000+ hours of balanced and unbalanced audio.</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Checkpoints</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">Cnn14_mAP=0.431.pth (~327 MB)</div>
            <p className="text-xs text-slate-600 mt-1">High-capacity 80M+ learned parameter tensor weights.</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Window & Hop Analysis</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">10.0s Windows with 5.0s Hop</div>
            <p className="text-xs text-slate-600 mt-1">Enables multi-track temporal activity segment tracking.</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Chroma Key Profiling</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">Krumhansl-Schmuckler Profiling</div>
            <p className="text-xs text-slate-600 mt-1">12-pitch chromagram correlation against major/minor key profiles.</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/70">
            <span className="text-[11px] font-bold uppercase text-slate-600">Beat & Tempo Tracking</span>
            <div className="text-base font-bold text-slate-800 mt-0.5">Librosa Dynamic Programming</div>
            <p className="text-xs text-slate-600 mt-1">Estimates global BPM and rhythm envelope dynamically.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
