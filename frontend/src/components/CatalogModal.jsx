import React, { useState, useEffect } from 'react';
import { Search, Music, Sparkles, Filter, CheckCircle2, ChevronRight, Layers } from 'lucide-react';

export default function CatalogModal() {
  const [catalogData, setCatalogData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFamily, setSelectedFamily] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/catalog')
      .then(res => res.json())
      .then(data => {
        setCatalogData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load catalog:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-3xl p-12 text-center border border-slate-200">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-sky-500 border-t-transparent mb-3" />
        <p className="text-sm font-semibold text-slate-600">Loading Instrument Encyclopedia...</p>
      </div>
    );
  }

  const families = catalogData?.families || [];
  const catalogByFamily = catalogData?.catalog_by_family || {};

  // Flatten all instruments for filtering
  let allInstruments = [];
  Object.keys(catalogByFamily).forEach(famKey => {
    const fam = catalogByFamily[famKey];
    fam.instruments.forEach(inst => {
      allInstruments.push({
        ...inst,
        familyKey: famKey,
        familyName: fam.name,
        familyColor: fam.color
      });
    });
  });

  const filteredInstruments = allInstruments.filter(inst => {
    const matchesFamily = selectedFamily === 'all' || inst.familyKey === selectedFamily;
    const matchesQuery = !searchQuery || 
      inst.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inst.role.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inst.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFamily && matchesQuery;
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-sky-500 via-indigo-600 to-indigo-700 rounded-3xl p-8 text-white shadow-elevated relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 opacity-10 pointer-events-none flex items-center pr-10">
          <Music className="h-64 w-64 text-white" />
        </div>
        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/20 backdrop-blur-md text-xs font-bold text-sky-100 mb-3 border border-white/20">
            <Sparkles className="h-3.5 w-3.5 text-amber-300" />
            <span>AudioSet & PANNs CNN14 Taxonomy</span>
          </div>
          <h2 className="text-3xl font-display font-bold tracking-tight mb-2">
            Musical Instrument Knowledge Base
          </h2>
          <p className="text-sm text-sky-100 leading-relaxed">
            Explore all {allInstruments.length} musical instrument classes detectable by our PANNs CNN14 neural network, complete with acoustic characteristics, frequency footprints, and historical origins.
          </p>
        </div>
      </div>

      {/* Search & Family Filters */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-soft flex flex-col sm:flex-row gap-4 items-center justify-between">
        {/* Search Input */}
        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-600" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search instruments, roles, descriptions..."
            className="w-full pl-10 pr-4 py-2 text-sm bg-slate-50 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition-all text-slate-800 placeholder-slate-600"
          />
        </div>

        {/* Family Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 w-full sm:w-auto">
          <button
            onClick={() => setSelectedFamily('all')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              selectedFamily === 'all'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            All ({allInstruments.length})
          </button>
          {Object.keys(catalogByFamily).map(famKey => {
            const fam = catalogByFamily[famKey];
            return (
              <button
                key={famKey}
                onClick={() => setSelectedFamily(famKey)}
                className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  selectedFamily === famKey
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {fam.name} ({fam.instruments.length})
              </button>
            );
          })}
        </div>
      </div>

      {/* Instruments Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredInstruments.map((inst, idx) => (
          <div
            key={idx}
            className="bg-white rounded-3xl p-6 border border-slate-200/90 shadow-soft hover:shadow-elevated transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                    {inst.familyName}
                  </span>
                  <h3 className="font-display font-bold text-lg text-slate-900 mt-1">
                    {inst.name}
                  </h3>
                </div>
                <div className="flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200/60">
                  <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                  <span>CNN14</span>
                </div>
              </div>

              <div className="text-xs font-semibold text-sky-700 mb-2">
                {inst.role}
              </div>

              <p className="text-xs text-slate-600 leading-relaxed mb-4">
                {inst.description}
              </p>

              <div className="flex items-center justify-between text-xs py-2 border-t border-b border-slate-100 mb-4">
                <span className="text-slate-600 font-medium">Frequency Span:</span>
                <span className="font-mono text-xs font-semibold text-slate-700 bg-slate-50 px-2 py-0.5 rounded">
                  {inst.freq_range}
                </span>
              </div>

              {inst.characteristics && inst.characteristics.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-4">
                  {inst.characteristics.map((c, i) => (
                    <span key={i} className="text-[10px] font-medium px-2 py-0.5 rounded bg-slate-50 text-slate-600 border border-slate-200/60">
                      {c}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {inst.fun_fact && (
              <div className="p-3 rounded-2xl bg-slate-50 border border-slate-200/60 text-[11px] text-slate-700 leading-relaxed mt-2">
                <span className="font-semibold text-slate-900 block mb-0.5">Origin / Trivia:</span>
                {inst.fun_fact}
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredInstruments.length === 0 && (
        <div className="bg-white rounded-3xl p-12 text-center border border-slate-200 text-slate-600">
          No instruments found matching "{searchQuery}".
        </div>
      )}
    </div>
  );
}
