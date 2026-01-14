import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, AlertTriangle, Activity, Brain, ShieldCheck, Server, CloudCog, BarChart2, X, Maximize2 } from 'lucide-react';
import type { DiagnosisResponse } from '../types';
import { API_BASE_URL } from '../utils';

interface DiagnosisReportProps {
  data: DiagnosisResponse;
  loading?: boolean;
}

export const DiagnosisReport: React.FC<DiagnosisReportProps> = ({ data, loading }) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  if (loading) return (
    <div className="h-full flex flex-col items-center justify-center space-y-4 text-primary-400">
      <Activity className="w-12 h-12 animate-pulse" />
      <div className="text-lg font-mono animate-pulse">DUAL-CORE ANALYSIS IN PROGRESS...</div>
    </div>
  );

  // Static list of visualizations available on backend
  const visualizations = [
    { title: 'Training Curves', src: '/static/visualizations/fig1_training_curves.png' },
    { title: 'Confusion Matrix', src: '/static/visualizations/fig2_confusion_matrix.png' },
    { title: 'Per-Class Metrics', src: '/static/visualizations/fig3_per_class_metrics.png' },
    { title: 'Dataset Distribution', src: '/static/visualizations/fig4_dataset_distribution.png' },
  ];

  return (
    <>
      <motion.div 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="h-full overflow-y-auto custom-scrollbar p-6 space-y-6"
      >
        {/* Header Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
            <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Local Model</div>
            <div className="text-sm font-bold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-primary-500" />
              {data.local_disease_type || 'Analyzing...'}
            </div>
          </div>
          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
             <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Cloud Verification</div>
             <div className="text-sm font-bold text-white flex items-center gap-2">
               <CloudCog className="w-4 h-4 text-purple-400" />
               {data.verification_status === 'match' ? 'Confirmed' : 'Pending'}
             </div>
          </div>
          <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
            <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Confidence</div>
            <div className="text-xl font-bold text-emerald-400 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              {data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </div>
        </div>

         {/* Dual Verification Banner */}
         <div className="bg-emerald-900/20 border border-emerald-500/30 rounded-lg p-3 flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-200 text-sm font-medium">Dual-Model Verification Complete: Local Analysis + Cloud API Validation</span>
         </div>
        
        {/* Model Performance Visualizations */}
        <div className="bg-slate-800/30 rounded-xl border border-slate-700/50 overflow-hidden">
           <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50 flex items-center gap-2">
             <BarChart2 className="w-4 h-4 text-blue-400" />
             <span className="font-medium text-slate-200">Model Performance Metrics</span>
           </div>
           <div className="p-4 overflow-x-auto">
             <div className="flex gap-4 min-w-max pb-2">
               {visualizations.map((viz, index) => (
                 <div 
                    key={index} 
                    className="relative group cursor-pointer overflow-hidden rounded-lg border border-slate-700 w-48 h-32 bg-slate-900"
                    onClick={() => setSelectedImage(`${API_BASE_URL}${viz.src}`)}
                  >
                   <img 
                      src={`${API_BASE_URL}${viz.src}`} 
                      alt={viz.title} 
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-80 group-hover:opacity-100" 
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'https://placehold.co/200x150/1e293b/475569?text=No+Data';
                      }}
                   />
                   <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-100" />
                   <div className="absolute bottom-2 left-2 right-2">
                     <div className="text-xs font-medium text-white truncate">{viz.title}</div>
                   </div>
                   <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <div className="bg-black/50 p-1 rounded backdrop-blur-sm">
                        <Maximize2 className="w-3 h-3 text-white" />
                      </div>
                   </div>
                 </div>
               ))}
             </div>
           </div>
        </div>

        {/* Local Analysis Section */}
        <div className="bg-slate-800/30 rounded-xl border border-slate-700/50 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50 flex items-center gap-2">
            <Brain className="w-4 h-4 text-primary-400" />
            <span className="font-medium text-slate-200">Local Model Analysis (Qwen2-VL)</span>
          </div>
          <div className="p-5 text-slate-300 text-sm">
            {data.local_diagnosis}
          </div>
        </div>

        {/* Cloud Report Section */}
        <div className="bg-slate-800/30 rounded-xl border border-slate-700/50 overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50 flex items-center gap-2">
            <FileText className="w-4 h-4 text-purple-400" />
            <span className="font-medium text-slate-200">Comprehensive Cloud Report (ModelScope)</span>
          </div>
          <div className="p-5 text-slate-300 prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
               components={{
                  h1: ({node, ...props}) => <h3 className="text-xl font-bold text-purple-200 mt-4 mb-2" {...props} />,
                  h2: ({node, ...props}) => <h4 className="text-lg font-bold text-purple-300 mt-3 mb-2" {...props} />,
                  h3: ({node, ...props}) => <strong className="block text-white mt-2 mb-1" {...props} />,
                  strong: ({node, ...props}) => <span className="text-purple-300 font-bold" {...props} />,
                  ul: ({node, ...props}) => <ul className="space-y-1 my-2 list-none pl-4" {...props} />,
                  li: ({node, ...props}) => (
                    <li className="flex gap-2 items-start" {...props}>
                      <span className="text-purple-500 mt-1.5 text-xs">●</span>
                      <span>{props.children}</span>
                    </li>
                  ),
               }}
            >
              {data.cloud_report.replace(/#/g, '')} 
            </ReactMarkdown>
          </div>
        </div>

        {/* Recommendations */}
        {data.recommendations && (
          <div className="bg-blue-900/20 rounded-xl border border-blue-500/30 p-4 flex gap-4">
            <div className="shrink-0">
              <AlertTriangle className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h4 className="font-medium text-blue-200 mb-1">System Recommendation</h4>
              <p className="text-sm text-blue-300/80">{data.recommendations}</p>
            </div>
          </div>
        )}
      </motion.div>

      {/* Image Modal */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div 
               initial={{ scale: 0.9, opacity: 0 }}
               animate={{ scale: 1, opacity: 1 }}
               exit={{ scale: 0.9, opacity: 0 }}
               className="relative max-w-5xl w-full max-h-[90vh] bg-slate-900 rounded-xl overflow-hidden shadow-2xl border border-slate-700"
               onClick={(e) => e.stopPropagation()}
            >
               <button 
                 onClick={() => setSelectedImage(null)}
                 className="absolute top-4 right-4 z-10 p-2 bg-black/50 text-white rounded-full hover:bg-black/80 transition-colors"
               >
                 <X className="w-6 h-6" />
               </button>
               <div className="w-full h-full flex items-center justify-center bg-slate-950/50">
                 <img src={selectedImage} alt="Full size visualization" className="max-w-full max-h-[85vh] object-contain" />
               </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
