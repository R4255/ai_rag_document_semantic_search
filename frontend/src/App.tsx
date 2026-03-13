import { useState, useEffect } from 'react';
import axios from 'axios';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';
import CacheStats from './components/CacheStats';
import DocumentList from './components/DocumentList';
import { Bot, Sparkles, Cpu, Menu, X } from 'lucide-react';

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
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

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
    <div className="h-dvh md:h-screen flex flex-col md:flex-row bg-background text-gray-100 font-sans overflow-hidden">
      {/* Mobile Header */}
      <div className="md:hidden sticky top-0 z-40 flex items-center justify-between p-4 border-b border-white/5 bg-surface/95 backdrop-blur">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
            Nexus RAG
          </h1>
        </div>
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="p-2 rounded-md hover:bg-white/10 text-gray-300 transition-colors"
        >
          <Menu className="w-6 h-6" />
        </button>
      </div>

      {/* Overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed md:relative inset-y-0 left-0 z-50 md:z-20 transform ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0 transition-transform duration-300 ease-in-out
        w-80 max-w-[85vw] md:max-w-none min-w-[320px]
        border-r border-white/5 bg-background/95 md:bg-background/60 backdrop-blur-3xl
        flex flex-col`}
      >
        <div className="p-6 space-y-6 overflow-y-auto min-h-0">
          {/* Mobile Sidebar Close Button */}
          <div className="flex justify-end md:hidden">
            <button onClick={() => setIsSidebarOpen(false)} className="text-gray-400 hover:text-white">
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Logo (Hidden on mobile inside sidebar as it's in the top header) */}
          <div className="hidden md:flex items-center space-x-3">
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
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 min-w-0 flex flex-col min-h-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-surface via-background to-background">
        <header className="hidden md:flex sticky top-0 z-30 flex-shrink-0 items-center px-8 py-5 border-b border-white/5 bg-background/70 backdrop-blur-md">
          <Bot className="w-5 h-5 text-primary mr-3" />
          <span className="font-semibold text-gray-200">Semantic Search Assistant</span>
          <span className="ml-auto text-[10px] text-gray-600 font-mono">gemini-3-flash • top-k=5 • LRU cached</span>
        </header>

        <div className="flex-1 min-h-0">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}

export default App;
