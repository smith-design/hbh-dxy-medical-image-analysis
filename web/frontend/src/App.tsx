import { useState } from 'react';
import { Upload, Activity, Shield, Microscope, AlertTriangle } from 'lucide-react';
import { ImageUploader } from './components/ImageUploader';
import { DiagnosisReport } from './components/DiagnosisReport';
import type { DiagnosisResponse } from './types';
import { API_BASE_URL } from './utils';



function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setDiagnosis(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch(`${API_BASE_URL}/api/diagnose`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Diagnosis failed');

      const data: DiagnosisResponse = await response.json();
      setDiagnosis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans selection:bg-primary-500/30">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex items-center justify-between pb-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary-500/10 rounded-lg border border-primary-500/20">
              <Microscope className="w-8 h-8 text-primary-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                DermaAI Analysis
              </h1>
              <p className="text-sm text-slate-400 font-mono">
                Multimodal Medical Imaging System
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-400">
              <Shield className="w-3 h-3 text-emerald-500" />
              <span>HIPAA Compliant</span>
            </div>
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-400">
              <Activity className="w-3 h-3 text-primary-500" />
              <span>Model v2.0 Active</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8 h-[calc(100vh-200px)] min-h-[600px]">
          
          {/* Left Column: Input */}
          <div className="flex flex-col gap-6">
            <div className="flex-1 bg-slate-900/50 rounded-2xl border border-slate-800 overflow-hidden relative group">
              <div className="absolute inset-0 bg-gradient-to-b from-primary-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
              <div className="p-6 h-full flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                    <Upload className="w-5 h-5 text-primary-400" />
                    Input Source
                  </h2>
                  {selectedImage && (
                    <span className="text-xs font-mono text-primary-400 bg-primary-950/50 px-2 py-1 rounded border border-primary-500/20">
                      {selectedImage.name}
                    </span>
                  )}
                </div>
                
                <div className="flex-1 min-h-0">
                  <ImageUploader 
                    onImageSelect={handleImageSelect}
                    selectedImage={selectedImage}
                    onClear={() => {
                      setSelectedImage(null);
                      setDiagnosis(null);
                    }}
                  />
                </div>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!selectedImage || loading}
              className={cn(
                "w-full py-4 rounded-xl font-bold text-lg tracking-wide transition-all duration-300 relative overflow-hidden",
                !selectedImage || loading
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-500 shadow-lg shadow-primary-900/20 active:scale-[0.98]"
              )}
            >
              {loading ? (
                <div className="flex items-center justify-center gap-3">
                  <Activity className="w-5 h-5 animate-spin" />
                  PROCESSING...
                </div>
              ) : (
                "START ANALYSIS"
              )}
            </button>
          </div>

          {/* Right Column: Output */}
          <div className="bg-slate-900/80 rounded-2xl border border-slate-800 overflow-hidden backdrop-blur-sm shadow-xl">
             <div className="h-full flex flex-col">
                <div className="p-6 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                  <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    Diagnostic Report
                  </h2>
                  {diagnosis && (
                    <div className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium border border-emerald-500/20">
                      Analysis Complete
                    </div>
                  )}
                </div>
                
                <div className="flex-1 overflow-hidden relative p-0">
                  {error && (
                    <div className="absolute top-0 left-0 right-0 z-50 p-4 bg-red-900/80 backdrop-blur text-red-100 border-b border-red-500/50 flex items-center gap-3 animate-in slide-in-from-top-2">
                       <AlertTriangle className="w-5 h-5 text-red-400" />
                       <span className="text-sm font-medium">{error}</span>
                    </div>
                  )}
                  {!diagnosis && !loading ? (

                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-600 space-y-4">
                      <Microscope className="w-16 h-16 opacity-20" />
                      <p className="font-mono text-sm">WAITING FOR INPUT...</p>
                    </div>
                  ) : (
                    <DiagnosisReport data={diagnosis!} loading={loading} />
                  )}
                </div>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Utility for class merging
function cn(...classes: (string | undefined | null | false)[]) {
  return classes.filter(Boolean).join(' ');
}

export default App;
