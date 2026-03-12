import { useState, useEffect } from 'react';
import axios from 'axios';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';
import CacheStats from './components/CacheStats';
import DocumentList from './components/DocumentList';
import { Bot, Sparkles, Cpu } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Generate distinct session/client ID for multi-tenancy without login
const getClientId = () => {
  let id = localStorage.getItem('nexus_client_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('nexus_client_id', id);
  }
  return id;
};

const CLIENT_ID = getClientId();

// Configure global Axios defaults so every request sends the ID
axios.defaults.headers.common['X-User-ID'] = CLIENT_ID;

function App() {
  const [cacheStats, setCacheStats] = useState({ hits: 0, misses: 0, size: 0, capacity: 0 });
  const [documents, setDocuments] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    const poll = setInterval(() => {
      axios.get(`${API}/api/cache/stats`).then(r => setCacheStats(r.data)).catch(() => { });
      axios.get(`${API}/api/documents`).then(r => setDocuments(r.data.documents || [])).catch(() => { });
    }, 4000);

    // Initial fetch
    axios.get(`${API}/api/health`).then(r => setHealth(r.data)).catch(() => { });
    axios.get(`${API}/api/documents`).then(r => setDocuments(r.data.documents || [])).catch(() => { });

    return () => clearInterval(poll);
  }, []);

  return (
    <div className="flex overflow-hidden h-screen bg-background text-gray-100 font-sans">
      {/* ── Sidebar ── */}
      <aside className="w-80 min-w-[320px] border-r border-white/5 bg-background/60 backdrop-blur-3xl flex flex-col p-6 space-y-6 z-10 overflow-y-auto">
        {/* Logo */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center shadow-lg shadow-primary/20">
            <Sparkles className="text-white w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              Nexus RAG
            </h1>
            <span className="text-[10px] text-gray-500 font-mono">v2.0 • Gemini 3 Flash </span>
          </div>
        </div>

        {/* Upload */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Knowledge Base</h2>
          <DocumentUpload onUploadSuccess={() => {
            axios.get(`${API}/api/documents`).then(r => setDocuments(r.data.documents || [])).catch(() => { });
          }} />
        </section>

        {/* Ingested docs */}
        {documents.length > 0 && (
          <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Ingested Documents</h2>
            <DocumentList documents={documents} />
          </section>
        )}

        {/* Cache Stats */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">System Metrics</h2>
          <CacheStats stats={cacheStats} />
        </section>

        {/* Footer */}
        <div className="mt-auto pt-4 border-t border-white/5">
          <div className="flex items-center text-xs text-gray-600 space-x-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span>Online</span>
            <span className="text-gray-700">•</span>
            <Cpu className="w-3 h-3" />
            <span>{health?.engine || 'Connecting…'}</span>
          </div>
          <p className="text-[10px] text-gray-700 mt-1">
            AsyncIO + FAISS + LangChain pipeline
          </p>
        </div>
      </aside>

      {/* ── Main Chat Area ── */}
      <main className="flex-1 relative bg-surface shadow-2xl flex flex-col">
        <header className="h-14 flex items-center px-8 border-b border-white/5 bg-surface/50 backdrop-blur-md sticky top-0 z-10">
          <Bot className="w-5 h-5 mr-3 text-secondary" />
          <span className="font-semibold text-gray-200">Semantic Search Assistant</span>
          <span className="ml-auto text-[10px] text-gray-600 font-mono">gemini-3-flash • top-k=5 • LRU cached</span>
        </header>
        <div className="flex-1 overflow-hidden relative">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}

export default App;
