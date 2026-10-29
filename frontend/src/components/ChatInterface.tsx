import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowRight, Check, Copy, FastForward, Plus, Square } from 'lucide-react';
import { API, CLIENT_ID } from '../lib/api';
import type { Citation } from '../lib/api';

type Status = 'searching' | 'writing' | 'done' | 'error' | 'stopped';

type Message = {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    citations: Citation[];
    status?: Status;
    error?: string;
    cacheHit?: boolean;
    /** From the server's `retrieval` event: how the passages were found and how long it took */
    retrieval?: { mode: string; ms?: number };
};

interface Props {
    readyCount: number;
    hasDocs: boolean;
    onCite: (c: Citation) => void;
    onAddDocs: () => void;
    /** chunk_id of the passage currently shown in the reader, to mark its chip as active */
    activeChunkId?: string;
}

const CITE = /\[(\d+)\](?!\()/g;

/** Turn [1] markers into links the markdown renderer hands to our citation chip. */
const linkCitations = (text: string) => text.replace(CITE, '[$1](#cite-$1)');

function citedNumbers(text: string, citations: Citation[]): Citation[] {
    const seen: number[] = [];
    for (const m of text.matchAll(CITE)) {
        const n = Number(m[1]);
        if (!seen.includes(n)) seen.push(n);
    }
    return seen.map(n => citations.find(c => c.n === n)).filter((c): c is Citation => !!c);
}

export default function ChatInterface({ readyCount, hasDocs, onCite, onAddDocs, activeChunkId }: Props) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [copiedId, setCopiedId] = useState<string | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const abortRef = useRef<AbortController | null>(null);
    // Follow the answer only while the reader is already at the bottom
    const stickToBottom = useRef(true);

    useEffect(() => {
        if (stickToBottom.current) bottomRef.current?.scrollIntoView({ behavior: 'auto' });
    }, [messages]);

    const handleScroll = () => {
        const el = scrollRef.current;
        if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };

    // Replace the last assistant message instead of mutating it (StrictMode runs updaters twice)
    const updateLast = (patch: (m: Message) => Partial<Message>) =>
        setMessages(prev => {
            const last = prev[prev.length - 1];
            if (!last || last.role !== 'assistant') return prev;
            return [...prev.slice(0, -1), { ...last, ...patch(last) }];
        });

    const resizeInput = () => {
        const el = inputRef.current;
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    };

    const send = async (raw: string) => {
        const query = raw.trim();
        if (!query || streaming) return;
        setInput('');
        requestAnimationFrame(resizeInput);
        stickToBottom.current = true;
        setStreaming(true);
        setMessages(prev => [
            ...prev,
            { id: crypto.randomUUID(), role: 'user', content: query, citations: [] },
            { id: crypto.randomUUID(), role: 'assistant', content: '', citations: [], status: 'searching' },
        ]);

        const controller = new AbortController();
        abortRef.current = controller;
        let failed = false;

        try {
            const response = await fetch(`${API}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-User-ID': CLIENT_ID },
                body: JSON.stringify({ query }),
                signal: controller.signal,
            });
            if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            for (;;) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value, { stream: !done });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    let data;
                    try { data = JSON.parse(line.slice(6)); } catch { continue; }
                    if (data.type === 'content') {
                        const text = typeof data.content === 'string' ? data.content : JSON.stringify(data.content);
                        updateLast(m => ({ content: m.content + text, status: 'writing' }));
                    } else if (data.type === 'citations') {
                        updateLast(() => ({ citations: data.citations || [] }));
                    } else if (data.type === 'retrieval') {
                        updateLast(() => ({ retrieval: { mode: data.mode, ms: data.timings_ms?.total } }));
                    } else if (data.type === 'cache_hit') {
                        updateLast(() => ({ cacheHit: true }));
                    } else if (data.type === 'error') {
                        failed = true;
                        updateLast(() => ({ status: 'error', error: data.message }));
                    }
                }
                if (done) break;
            }
            if (!failed) updateLast(() => ({ status: 'done' }));
        } catch {
            if (controller.signal.aborted) {
                updateLast(() => ({ status: 'stopped' }));
            } else {
                updateLast(() => ({ status: 'error', error: "Couldn't reach the server. Check your connection and try again." }));
            }
        } finally {
            abortRef.current = null;
            setStreaming(false);
        }
    };

    const copy = async (m: Message) => {
        const text = m.content.replace(CITE, '').replace(/\s+([.,;:])/g, '$1');
        try {
            await navigator.clipboard.writeText(text);
            setCopiedId(m.id);
            setTimeout(() => setCopiedId(null), 1500);
        } catch {
            /* clipboard blocked by the browser: nothing to do */
        }
    };

    const markdownFor = (m: Message): Components => ({
        a: ({ href, children }) => {
            if (href?.startsWith('#cite-')) {
                const c = m.citations.find(x => x.n === Number(href.slice(6)));
                const active = !!c && c.chunk_id === activeChunkId;
                return (
                    <button
                        type="button"
                        disabled={!c}
                        onClick={() => c && onCite(c)}
                        aria-label={c ? `Show source ${c.n}: ${c.source}, page ${c.page}` : 'Source'}
                        className={`mx-0.5 inline-flex min-w-[24px] items-center justify-center rounded-[3px] border px-1.5 align-[1px] font-mono text-[12px] font-semibold leading-[18px] transition-colors
                            ${active ? 'border-accent-600 bg-accent-600 text-white' : 'border-accent-600 text-accent-600 hover:bg-accent-600 hover:text-white'} disabled:border-rule-400 disabled:text-rule-400`}
                    >
                        {children}
                    </button>
                );
            }
            return <a href={href} target="_blank" rel="noreferrer" className="text-accent-600 underline">{children}</a>;
        },
    });

    const empty = messages.length === 0;

    return (
        <div className="h-full flex flex-col min-h-0">
            <div ref={scrollRef} onScroll={handleScroll} className="flex-1 min-h-0 overflow-y-auto">
                <div className="mx-auto flex w-full max-w-[720px] flex-col gap-6 px-4 py-6 lg:px-6">
                    {empty && (
                        <div className="flex flex-col items-center gap-3 px-2 py-16 text-center">
                            <h2 className="font-mono text-xl font-semibold">&gt; ask your documents_</h2>
                            <p className="max-w-sm text-ink-600">
                                {hasDocs
                                    ? readyCount > 0
                                        ? `Ask anything about your ${readyCount} ready document${readyCount === 1 ? '' : 's'}. Every answer points to the exact passage it used.`
                                        : 'Your documents are still being indexed. You can ask as soon as one is ready.'
                                    : 'Add a PDF or TXT file to get started. Answers will point to the exact passage they used.'}
                            </p>
                            {!hasDocs && (
                                <button type="button" onClick={onAddDocs} className="mt-2 inline-flex min-h-[44px] items-center gap-2 rounded border-[1.5px] border-ink-900 bg-ink-900 px-5 font-mono text-sm font-semibold text-paper-100 hover:bg-[#33322E]">
                                    <Plus className="h-4 w-4" aria-hidden="true" />add documents
                                </button>
                            )}
                        </div>
                    )}

                    {messages.map(m => {
                        if (m.role === 'user') {
                            return (
                                <div key={m.id} className="max-w-[85%] self-end whitespace-pre-wrap rounded border-[1.5px] border-ink-900 bg-paper-50 px-3.5 py-2.5 font-mono text-[14px] leading-relaxed text-ink-900">
                                    <span className="text-accent-600" aria-hidden="true">&gt; </span>{m.content}
                                </div>
                            );
                        }
                        const cited = citedNumbers(m.content, m.citations);
                        const finished = m.status === 'done' || m.status === 'stopped';
                        return (
                            <article key={m.id} className="flex flex-col gap-2.5" aria-live={m.status === 'writing' ? 'polite' : undefined}>
                                {m.status === 'searching' && (
                                    <span className="inline-flex items-center gap-2 self-start font-mono text-xs text-ink-600">
                                        <span className="h-2 w-2 animate-pulse rounded-full bg-accent-600" aria-hidden="true" />
                                        {readyCount > 0 ? `searching ${readyCount} document${readyCount === 1 ? '' : 's'}…` : 'searching your documents…'}
                                    </span>
                                )}

                                {(m.retrieval || m.cacheHit) && m.status !== 'searching' && (
                                    <span className="font-mono text-[11.5px] text-ink-600">
                                        {m.cacheHit
                                            ? 'instant answer · asked before'
                                            : `${MODE_LABEL[m.retrieval!.mode] ?? m.retrieval!.mode}${m.retrieval!.ms != null ? ` · ${Math.round(m.retrieval!.ms)} ms` : ''}`}
                                    </span>
                                )}

                                {m.content && (
                                    <div className="max-w-none text-[15.5px] leading-[1.7] text-ink-900 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_code]:font-mono [&_code]:text-[0.9em]">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownFor(m)}>
                                            {linkCitations(m.content)}
                                        </ReactMarkdown>
                                        {m.status === 'writing' && <span className="ml-0.5 inline-block h-[17px] w-[9px] animate-pulse bg-ink-900 align-[-3px]" aria-hidden="true" />}
                                    </div>
                                )}

                                {m.status === 'error' && (
                                    <p role="alert" className="rounded border border-dashed border-warn-200 bg-warn-50 px-4 py-3 text-warn-900">{m.error}</p>
                                )}

                                {finished && cited.length > 0 && (
                                    <div className="mt-1 flex flex-col gap-1.5">
                                        {cited.map(c => (
                                            <button
                                                key={c.n}
                                                type="button"
                                                onClick={() => onCite(c)}
                                                className={`flex min-h-[40px] w-full items-center gap-2.5 rounded border px-3 text-left font-mono text-[12.5px] transition-colors
                                                    ${c.chunk_id === activeChunkId ? 'border-solid border-accent-600 bg-paper-50' : 'border-dashed border-rule-300 hover:bg-paper-50'}`}
                                            >
                                                <span className="font-semibold text-accent-600">[{c.n}]</span>
                                                <span className="min-w-0 flex-1 truncate">{c.source}</span>
                                                <span className="shrink-0 text-ink-600">p.{c.page}</span>
                                            </button>
                                        ))}
                                    </div>
                                )}

                                {finished && (
                                    <div className="flex items-center gap-1 font-mono text-xs text-ink-600">
                                        <button
                                            type="button"
                                            onClick={() => copy(m)}
                                            aria-label="Copy answer"
                                            className="inline-flex h-11 w-11 items-center justify-center rounded hover:bg-paper-150"
                                        >
                                            {copiedId === m.id ? <Check className="h-[17px] w-[17px] text-accent-600" aria-hidden="true" /> : <Copy className="h-[17px] w-[17px]" aria-hidden="true" />}
                                        </button>
                                        {m.cacheHit && (
                                            <span className="inline-flex items-center gap-1"><FastForward className="h-3.5 w-3.5" aria-hidden="true" />from cache</span>
                                        )}
                                        {m.status === 'stopped' && <span>stopped</span>}
                                    </div>
                                )}
                            </article>
                        );
                    })}
                    <div ref={bottomRef} />
                </div>
            </div>

            <form
                onSubmit={e => { e.preventDefault(); send(input); }}
                className="shrink-0 px-3 pb-[max(14px,env(safe-area-inset-bottom))] pt-2 lg:px-6"
            >
                <label htmlFor="ask" className="sr-only">Ask a question about your documents</label>
                <div className="mx-auto flex max-w-[696px] items-end gap-1.5 rounded border-[1.5px] border-ink-900 bg-paper-50 p-1.5">
                    <button
                        type="button"
                        onClick={onAddDocs}
                        aria-label="Add documents"
                        className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded border border-dashed border-rule-300 text-ink-900 hover:bg-paper-150"
                    >
                        <Plus className="h-[18px] w-[18px]" aria-hidden="true" />
                    </button>
                    <span className="self-center pl-1 font-mono text-[15px] font-semibold text-accent-600" aria-hidden="true">&gt;</span>
                    <textarea
                        id="ask"
                        ref={inputRef}
                        rows={1}
                        value={input}
                        onChange={e => { setInput(e.target.value); resizeInput(); }}
                        onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                                e.preventDefault();
                                send(input);
                            }
                        }}
                        placeholder={streaming ? 'writing the answer…' : 'ask your documents…'}
                        className="max-h-40 min-w-0 flex-1 resize-none bg-transparent px-1 py-2.5 font-mono text-[15px] leading-snug text-ink-900 outline-none placeholder:text-ink-600 focus-visible:ring-0 focus-visible:ring-offset-0"
                    />
                    {streaming ? (
                        <button
                            type="button"
                            onClick={() => abortRef.current?.abort()}
                            aria-label="Stop writing the answer"
                            className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded bg-ink-900 text-white"
                        >
                            <Square className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
                        </button>
                    ) : (
                        <button
                            type="submit"
                            disabled={!input.trim()}
                            aria-label="Send"
                            className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded bg-accent-600 text-white transition-colors hover:bg-accent-700 disabled:bg-rule-400"
                        >
                            <ArrowRight className="h-5 w-5" aria-hidden="true" />
                        </button>
                    )}
                </div>
                <p className="mx-auto mt-1.5 hidden max-w-[696px] text-center font-mono text-[11px] text-ink-600 sm:block">enter to send · shift + enter for a new line</p>
            </form>
        </div>
    );
}

const MODE_LABEL: Record<string, string> = {
    bm25: 'bm25 · keywords',
    dense: 'dense · vectors',
    hybrid: 'hybrid · bm25 + dense · rrf',
    hybrid_rerank: 'hybrid · rrf · reranked',
};
