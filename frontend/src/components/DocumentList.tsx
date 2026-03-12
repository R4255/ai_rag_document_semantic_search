// DocumentList.tsx
import { FileText, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

interface Props {
    documents: any[];
}

export default function DocumentList({ documents }: Props) {
    return (
        <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {documents.map((doc, i) => (
                <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-center bg-surface/40 hover:bg-surface/70 border border-white/5 rounded-lg px-3 py-2 transition-colors group"
                >
                    <FileText className="w-4 h-4 text-primary/80 mr-2 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                        <p className="text-xs text-gray-300 truncate font-medium">{doc.filename}</p>
                        <p className="text-[10px] text-gray-600">
                            {doc.chunks ? `${doc.chunks} chunks` : ''}
                            {doc.elapsed_s ? ` • ${doc.elapsed_s}s` : ''}
                        </p>
                    </div>
                    {doc.status === 'success' ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-500 flex-shrink-0" />
                    ) : doc.status === 'error' ? (
                        <XCircle className="w-3.5 h-3.5 text-red-500 flex-shrink-0" />
                    ) : (
                        <Clock className="w-3.5 h-3.5 text-yellow-500 animate-pulse flex-shrink-0" />
                    )}
                </motion.div>
            ))}
        </div>
    );
}
