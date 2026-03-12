import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { UploadCloud, FileText, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
    onUploadSuccess?: () => void;
}

export default function DocumentUpload({ onUploadSuccess }: Props) {
    const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
    const [filename, setFilename] = useState('');
    const [errorMsg, setErrorMsg] = useState('');

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        if (acceptedFiles.length === 0) return;

        const file = acceptedFiles[0];
        setFilename(file.name);
        setStatus('uploading');
        setErrorMsg('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:8000/api/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            if (response.status === 200) {
                setStatus('success');
                onUploadSuccess?.();
                setTimeout(() => setStatus('idle'), 4000);
            }
        } catch (e: any) {
            console.error(e);
            setErrorMsg(e?.response?.data?.detail || 'Upload failed');
            setStatus('error');
            setTimeout(() => setStatus('idle'), 5000);
        }
    }, [onUploadSuccess]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'] },
        multiple: false,
    });

    return (
        <div className="space-y-3">
            <div
                {...getRootProps()}
                className={`p-5 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all duration-300 relative overflow-hidden group
          ${isDragActive
                        ? 'border-primary bg-primary/10 scale-[1.02]'
                        : 'border-white/10 bg-surface/30 hover:bg-surface/50 hover:border-white/20'}`}
            >
                <input {...getInputProps()} />
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <AnimatePresence mode="wait">
                    {status === 'idle' && (
                        <motion.div key="idle" initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} className="flex flex-col items-center space-y-2 text-center relative z-10">
                            <UploadCloud className="w-7 h-7 text-primary" />
                            <div className="text-sm font-medium text-gray-300">Upload Document</div>
                            <div className="text-[11px] text-gray-500">Drag & drop PDF / TXT</div>
                        </motion.div>
                    )}
                    {status === 'uploading' && (
                        <motion.div key="uploading" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center space-y-2 z-10">
                            <Loader2 className="w-7 h-7 text-primary animate-spin" />
                            <div className="text-sm text-gray-300">Streaming to pipeline…</div>
                        </motion.div>
                    )}
                    {status === 'success' && (
                        <motion.div key="success" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center space-y-2 z-10">
                            <CheckCircle2 className="w-7 h-7 text-green-500" />
                            <div className="text-sm text-green-400">Ingestion queued!</div>
                            <div className="text-[11px] text-gray-500 truncate max-w-[170px]">{filename}</div>
                        </motion.div>
                    )}
                    {status === 'error' && (
                        <motion.div key="error" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center space-y-2 z-10">
                            <AlertCircle className="w-7 h-7 text-red-500" />
                            <div className="text-sm text-red-400">{errorMsg || 'Upload failed'}</div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <div className="bg-surface/40 p-2.5 rounded-lg flex items-center">
                <FileText className="w-3.5 h-3.5 text-gray-500 mr-2 flex-shrink-0" />
                <span className="text-[10px] text-gray-500 leading-tight">
                    PDF / TXT → chunks → Gemini embeddings → FAISS index
                </span>
            </div>
        </div>
    );
}
