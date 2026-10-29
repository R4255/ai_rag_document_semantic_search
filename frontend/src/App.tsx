import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import axios from 'axios';
import { AnimatePresence, motion, useDragControls } from 'framer-motion';
import { FileText, Trash2, Upload, X } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import DocumentPane, { StatusIcon } from './components/DocumentPane';
import { API, deleteDocument, fetchDocuments } from './lib/api';
import type { Citation, DocInfo, Focus } from './lib/api';

type Conn = 'connecting' | 'waking' | 'warming' | 'online' | 'offline';
type Sheet = null | 'docs' | 'source';

const DESKTOP = '(min-width: 1024px)';

function useIsDesktop() {
  const [desktop, setDesktop] = useState(() => window.matchMedia(DESKTOP).matches);
  useEffect(() => {
    const mq = window.matchMedia(DESKTOP);
    const onChange = () => setDesktop(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return desktop;
}

function App() {
  const isDesktop = useIsDesktop();
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [activeDoc, setActiveDoc] = useState<string | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  const [sheet, setSheet] = useState<Sheet>(null);
  const [conn, setConn] = useState<Conn>('connecting');
  const [engine, setEngine] = useState('');
  const [toast, setToast] = useState('');
  const toastTimer = useRef<number | undefined>(undefined);
  const fileInput = useRef<HTMLInputElement>(null);
  const prevDocs = useRef<Record<string, DocInfo['status']>>({});

  const say = useCallback((msg: string) => {
    setToast(msg);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(''), 3200);
  }, []);

  const refreshDocs = useCallback(async () => {
    let next: DocInfo[];
    try {
      next = await fetchDocuments();
    } catch {
      return; /* server unreachable: the status pill already says so */
    }
    for (const d of next) {
      if (prevDocs.current[d.filename] === 'processing' && d.status !== 'processing') {
        say(d.status === 'success' ? `${d.filename} is ready to search` : `${d.filename}: ${d.error || 'could not be processed'}`);
      }
    }
    prevDocs.current = Object.fromEntries(next.map(d => [d.filename, d.status]));
    setDocs(next);
  }, [say]);

  // Health: a sleeping free-tier server can take a while to wake, so the first
  // failures mean "waking up", not "offline".
  useEffect(() => {
    let timer: number | undefined;
    let everOnline = false;
    let askedWarmup = false;
    const check = async () => {
      try {
        const r = await axios.get(`${API}/api/health`, { timeout: 10000 });
        if (!everOnline) refreshDocs();
        everOnline = true;
        // Local models unload when idle; load them now so the first question doesn't wait
        const models: { chat: boolean; embed: boolean } | null = r.data.models ?? null;
        const cold = !!models && !(models.chat && models.embed);
        if (cold && !r.data.warming && !askedWarmup) {
          askedWarmup = true;
          axios.post(`${API}/api/warmup`).catch(() => { /* the next check reports the state */ });
        }
        setConn(cold ? 'warming' : 'online');
        setEngine(r.data.engine || '');
        timer = window.setTimeout(check, cold ? 3000 : 30000);
      } catch {
        setConn(everOnline ? 'offline' : 'waking');
        timer = window.setTimeout(check, 5000);
      }
    };
    check();
    return () => window.clearTimeout(timer);
  }, [refreshDocs]);

  useEffect(() => {
    window.addEventListener('focus', refreshDocs);
    return () => window.removeEventListener('focus', refreshDocs);
  }, [refreshDocs]);

  // Poll only while something is still being indexed
  useEffect(() => {
    if (!docs.some(d => d.status === 'processing')) return;
    const t = window.setTimeout(refreshDocs, 2500);
    return () => window.clearTimeout(t);
  }, [docs, refreshDocs]);

  // The open document: the one the user picked, else the first ready one
  const shownDoc = activeDoc && docs.some(d => d.filename === activeDoc)
    ? activeDoc
    : (docs.find(d => d.status === 'success') || docs[0])?.filename ?? null;

  const upload = async (files: File[]) => {
    let lastNew: string | null = null;
    for (const file of files) {
      const form = new FormData();
      form.append('file', file);
      try {
        const r = await axios.post(`${API}/api/upload`, form);
        if (r.data.status === 'duplicate') {
          say(`${file.name} is already in your library, so it was skipped.`);
        } else {
          lastNew = file.name;
          say(files.length > 1 ? `Uploading ${files.length} files…` : `Uploading ${file.name}…`);
        }
      } catch (e) {
        const detail = axios.isAxiosError(e) ? e.response?.data?.detail : undefined;
        say(detail || `Couldn't upload ${file.name}. Check your connection and try again.`);
      }
    }
    await refreshDocs();
    if (lastNew) setActiveDoc(lastNew);
  };

  const remove = async (d: DocInfo) => {
    if (!window.confirm(`Delete ${d.filename}? Answers will stop using it.`)) return;
    try {
      await deleteDocument(d.id);
      say(`${d.filename} was deleted`);
      if (focus?.source === d.filename) setFocus(null);
    } catch (e) {
      const detail = axios.isAxiosError(e) ? e.response?.data?.detail : undefined;
      say(detail || `Couldn't delete ${d.filename}. Try again.`);
    }
    await refreshDocs();
  };

  const openPicker = () => fileInput.current?.click();

  const cite = (c: Citation) => {
    setFocus({ source: c.source, chunkId: c.chunk_id, n: c.n });
    setActiveDoc(c.source);
    if (!isDesktop) setSheet('source');
  };

  const readDoc = (name: string) => {
    setActiveDoc(name);
    setFocus(null);
    setSheet('source');
  };

  const readyCount = docs.filter(d => d.status === 'success').length;
  const hasIssue = docs.some(d => d.status === 'error');
  const sourceDoc = docs.find(d => d.filename === shownDoc);

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-paper-200 text-ink-900">
      <input
        ref={fileInput}
        type="file"
        multiple
        accept=".pdf,.txt,application/pdf,text/plain"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={e => { upload(Array.from(e.target.files || [])); e.target.value = ''; }}
      />

      <header className="shrink-0 border-b-[1.5px] border-dashed border-rule-300 bg-paper-100 pt-[env(safe-area-inset-top)]">
        <div className="flex items-center gap-3 px-4 py-2.5 lg:px-7">
          <span className="font-mono text-xl font-semibold tracking-tight">nexus<span className="animate-pulse text-accent-600">_</span></span>
          <span className="hidden font-mono text-xs text-ink-600 md:inline">read along with every answer</span>
          <span className="flex-1" />
          <ConnPill conn={conn} engine={engine} />
          <button
            type="button"
            onClick={openPicker}
            className="hidden min-h-[40px] items-center gap-2 rounded border-[1.5px] border-ink-900 bg-ink-900 px-4 font-mono text-[13px] font-semibold text-paper-100 hover:bg-[#33322E] lg:inline-flex"
          >
            <Upload className="h-4 w-4" aria-hidden="true" />upload .pdf/.txt
          </button>
          <button
            type="button"
            onClick={() => setSheet('docs')}
            aria-label={`Documents, ${docs.length}${hasIssue ? ', one needs attention' : ''}`}
            className="relative inline-flex h-11 items-center gap-2 rounded border-[1.5px] border-ink-900 bg-paper-50 px-3.5 font-mono text-[14px] font-semibold lg:hidden"
          >
            <FileText className="h-[18px] w-[18px]" aria-hidden="true" />{docs.length}
            {hasIssue && <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-paper-100 bg-[#C0611A]" />}
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {isDesktop && (
          <section aria-label="Document reader" className="min-h-0 w-[56%] border-r-[1.5px] border-dashed border-rule-300 xl:w-[58%]">
            <DocumentPane
              docs={docs}
              activeDoc={shownDoc}
              onSelect={name => { setActiveDoc(name); setFocus(null); }}
              focus={focus}
              onUpload={upload}
              onDelete={remove}
            />
          </section>
        )}
        <section aria-label="Chat" className="min-h-0 min-w-0 flex-1 bg-paper-100">
          <ChatInterface
            readyCount={readyCount}
            hasDocs={docs.length > 0}
            onCite={cite}
            onAddDocs={() => (isDesktop ? openPicker() : setSheet('docs'))}
            activeChunkId={focus?.chunkId}
          />
        </section>
      </div>

      <BottomSheet open={sheet === 'docs'} onClose={() => setSheet(null)} label="Your documents">
        <div className="flex items-center gap-3 px-5 pb-2">
          <h2 className="flex-1 font-mono text-lg font-semibold">your documents</h2>
          <CloseButton onClick={() => setSheet(null)} />
        </div>
        <div className="flex-1 overflow-y-auto px-5 pb-3">
          {docs.length === 0 && <p className="py-6 text-center text-ink-600">No documents yet. Add a PDF or TXT file to start.</p>}
          <ul className="flex flex-col gap-2">
            {docs.map(d => (
              <li key={d.id} className="flex items-stretch gap-2">
                <button
                  type="button"
                  onClick={() => d.status !== 'processing' && readDoc(d.filename)}
                  className={`flex min-w-0 flex-1 items-center gap-3 rounded border border-dashed p-3 text-left
                    ${d.status === 'error' ? 'border-warn-200 bg-warn-50' : 'border-rule-300 bg-paper-50'}`}
                >
                  <StatusIcon status={d.status} />
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="break-words font-mono text-[13.5px] font-medium">{d.filename}</span>
                    <span className={`font-mono text-xs ${d.status === 'error' ? 'text-warn-700' : 'text-ink-600'}`}>
                      {d.status === 'processing' ? 'reading and indexing…' : d.status === 'error' ? d.error || 'could not be processed' : `${d.chunks} passage${d.chunks === 1 ? '' : 's'} · tap to read`}
                    </span>
                  </span>
                </button>
                {d.status !== 'processing' && (
                  <button
                    type="button"
                    onClick={() => remove(d)}
                    aria-label={`Delete ${d.filename}`}
                    className="flex w-12 shrink-0 items-center justify-center rounded border border-dashed border-[#B5523A] text-[#9A3B25] hover:bg-[#FBEDE8]"
                  >
                    <Trash2 className="h-[18px] w-[18px]" aria-hidden="true" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
        <div className="px-5 pb-[max(20px,env(safe-area-inset-bottom))] pt-2">
          <button
            type="button"
            onClick={openPicker}
            className="flex min-h-[52px] w-full items-center justify-center gap-2 rounded border-[1.5px] border-ink-900 bg-ink-900 font-mono text-[15px] font-semibold text-paper-100 hover:bg-[#33322E]"
          >
            <Upload className="h-5 w-5" aria-hidden="true" />upload .pdf/.txt
          </button>
        </div>
      </BottomSheet>

      <BottomSheet open={sheet === 'source'} onClose={() => setSheet(null)} label={sourceDoc?.filename || 'Document'}>
        <div className="flex items-center gap-3 px-5 pb-1">
          {focus && focus.source === shownDoc && (
            <span className="rounded-[3px] bg-accent-600 px-2 py-0.5 font-mono text-sm font-semibold text-white">{focus.n}</span>
          )}
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate font-mono text-sm font-semibold">{sourceDoc?.filename}</span>
            <span className="font-mono text-xs text-ink-600">{focus && focus.source === shownDoc ? 'the passage this answer used' : 'full document'}</span>
          </div>
          <CloseButton onClick={() => setSheet(null)} />
        </div>
        <div className="h-[68dvh] min-h-0">
          <DocumentPane docs={docs} activeDoc={shownDoc} onSelect={setActiveDoc} focus={focus} onUpload={upload} showTabs={false} />
        </div>
      </BottomSheet>

      <AnimatePresence>
        {toast && (
          <motion.div
            role="status"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="fixed inset-x-4 top-[calc(env(safe-area-inset-top)+68px)] z-50 mx-auto max-w-[520px] rounded bg-ink-900 px-4 py-3 font-mono text-[13.5px] text-paper-100 shadow-[0_8px_24px_rgba(30,30,28,0.25)]"
          >
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ConnPill({ conn, engine }: { conn: Conn; engine: string }) {
  const label = {
    connecting: 'Connecting…',
    waking: 'Waking up the server…',
    warming: 'Online · warming model…',
    online: 'Online',
    offline: 'Offline · retrying',
  }[conn];
  const dot = { connecting: 'bg-rule-400', waking: 'bg-[#D9A13B] animate-pulse', warming: 'bg-[#D9A13B] animate-pulse', online: 'bg-[#2E9E68]', offline: 'bg-[#C0611A]' }[conn];
  return (
    <span className="inline-flex items-center gap-2 rounded border border-dashed border-rule-300 px-2.5 py-1.5 font-mono text-xs text-[#3D3A34]" role="status">
      <span className={`h-[7px] w-[7px] rounded-full ${dot}`} aria-hidden="true" />
      {label.toLowerCase()}
      {conn === 'online' && engine && <span className="hidden text-ink-600 xl:inline">· {engine.toLowerCase()}</span>}
    </span>
  );
}

function CloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Close"
      className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded border border-dashed border-rule-300 text-ink-900 hover:bg-paper-150"
    >
      <X className="h-[18px] w-[18px]" aria-hidden="true" />
    </button>
  );
}

function BottomSheet({ open, onClose, label, children }: { open: boolean; onClose: () => void; label: string; children: ReactNode }) {
  const drag = useDragControls();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label={label}>
          <motion.div
            className="absolute inset-0 bg-ink-900/40"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
          <motion.div
            className="absolute inset-x-0 bottom-0 mx-auto flex max-h-[88dvh] w-full max-w-[640px] flex-col rounded-t-lg border-t-[1.5px] border-dashed border-rule-300 bg-paper-100 shadow-[0_-10px_40px_rgba(30,30,28,0.18)]"
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 32, stiffness: 320 }}
            drag="y"
            dragControls={drag}
            dragListener={false}
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0, bottom: 0.7 }}
            onDragEnd={(_, info) => { if (info.offset.y > 110 || info.velocity.y > 500) onClose(); }}
          >
            <div className="flex cursor-grab touch-none justify-center pb-2 pt-3" onPointerDown={e => drag.start(e)}>
              <span className="h-1.5 w-10 rounded-full bg-rule-300" aria-hidden="true" />
            </div>
            {children}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

export default App;
