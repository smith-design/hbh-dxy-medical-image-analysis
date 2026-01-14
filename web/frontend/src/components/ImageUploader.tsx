import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileUp, Scan, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../utils';


interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ 
  onImageSelect, 
  selectedImage,
  onClear 
}) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      onImageSelect(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
    }
  }, [onImageSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1
  });

  const clearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    onClear();
  };

  return (
    <div className="w-full h-full min-h-[400px]">
      <div
        {...getRootProps()}
        className={cn(
          "relative h-full w-full rounded-xl border-2 border-dashed transition-all duration-300 overflow-hidden group cursor-pointer",
          isDragActive 
            ? "border-primary-500 bg-primary-500/10" 
            : "border-slate-700 hover:border-primary-500/50 hover:bg-slate-800/50",
          selectedImage ? "border-transparent" : "p-8"
        )}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {!selectedImage ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center justify-center h-full text-center space-y-4"
            >
              <div className="p-4 rounded-full bg-slate-800/50 ring-1 ring-white/10 group-hover:scale-110 transition-transform duration-300">
                <FileUp className="w-10 h-10 text-primary-400" />
              </div>
              <div>
                <p className="text-lg font-medium text-slate-200">
                  {isDragActive ? "Drop image here..." : "Drag & drop image"}
                </p>
                <p className="text-sm text-slate-400 mt-2">
                  or click to select from files
                </p>
              </div>
              <div className="text-xs text-slate-500 bg-slate-900/50 px-3 py-1 rounded-full border border-slate-800">
                Supports JPG, PNG
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="relative w-full h-full bg-black/40 flex items-center justify-center"
            >
              {/* Scan Effect Overlay */}
              <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden">
                <div className="w-full h-[2px] bg-primary-500/50 shadow-[0_0_15px_rgba(14,165,233,0.5)] animate-scan" />
              </div>
              
              {/* Grid Overlay */}
              <div className="absolute inset-0 z-0 bg-[linear-gradient(rgba(14,165,233,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(14,165,233,0.03)_1px,transparent_1px)] bg-[size:20px_20px]" />

              {/* Corner Markers */}
              <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary-500" />
              <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary-500" />
              <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary-500" />
              <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary-500" />

              <img
                src={preview!}
                alt="Upload preview"
                className="max-h-full max-w-full object-contain shadow-2xl"
              />

              <div className="absolute top-4 right-4 z-20 flex gap-2">
                <button
                  onClick={clearImage}
                  className="p-2 rounded-full bg-slate-900/80 text-slate-400 hover:text-white hover:bg-red-500/20 hover:ring-1 hover:ring-red-500 transition-all"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 px-4 py-2 bg-slate-900/90 backdrop-blur border border-primary-500/30 rounded-full flex items-center gap-2 text-primary-300 text-sm font-mono shadow-lg">
                <Scan className="w-4 h-4 animate-pulse" />
                <span>AI ANALYSIS READY</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
