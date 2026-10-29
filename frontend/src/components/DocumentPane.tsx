import { useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { AlertCircle, CheckCircle2, Loader2, Trash2, UploadCloud } from 'lucide-react';
import { fetchPassages, readableText } from '../lib/api';
import type { DocInfo, Focus, Passage } from '../lib/api';

interface Props {
    docs: DocInfo[];
    activeDoc: string | null;
    onSelect: (name: string) => void;
    focus: Focus | null;
    onUpload: (files: File[]) => void;
    /** Shown as a delete button on the sheet (desktop); phones delete from the documents sheet. */
    onDelete?: (doc: DocInfo) => void;
    /** Tabs are hidden when the reader is shown inside the phone's source sheet. */
    showTabs?: boolean;
}

export default function DocumentPane({ docs, activeDoc, onSelect, focus, onUpload, onDelete, showTabs = true }: Props) {
    const [passages, setPassages] = useState<Record<string, Passage[]>>({});
    const [failed, setFailed] = useState<Record<string, boolean>>({});
    const scrollRef = useRef<HTMLDivElement | null>(null);

    const active = docs.find(d => d.filename === activeDoc) || null;
    const ready = active?.status === 'success';
    // Re-fetch when the same name is re-uploaded (new timestamp)
    const cacheKey = active ? `${active.filename}@${active.timestamp}` : '';
    const text = cacheKey ? passages[cacheKey] : undefined;
    const loadError = cacheKey && failed[cacheKey] ? 'Could not load this document. Try again in a moment.' : null;

    useEffect(() => {
        if (!active || !ready || passages[cacheKey] || failed[cacheKey]) return;
        let cancelled = false;
        fetchPassages(active.filename)
            .then(p => { if (!cancelled) setPassages(prev => ({ ...prev, [cacheKey]: p })); })
            .catch(() => { if (!cancelled) setFailed(prev => ({ ...prev, [cacheKey]: true })); });
        return () => { cancelled = true; };
    }, [active, ready, cacheKey, passages, failed]);

    // Bring the cited passage into view whenever a citation is opened
    useEffect(() => {
        scrollRef.current?.querySelector('mark[data-hit]')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }, [focus, text]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop: (files) => files.length && onUpload(files),
        accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'] },
        multiple: true,
    });

    if (docs.length === 0) {
        return (
            <div className="h-full flex items-center justify-center p-6">
                <div
                    {...getRootProps()}
                    className={`w-full max-w-md cursor-pointer rounded border-2 border-dashed px-6 py-12 text-center transition-colors
                        ${isDragActive ? 'border-accent-600 bg-paper-300' : 'border-rule-300 bg-paper-50 hover:bg-paper-100'}`}
                >
                    <input {...getInputProps()} />
                    <UploadCloud className="mx-auto mb-3 h-8 w-8 text-accent-600" aria-hidden="true" />
                    <p className="font-mono text-base font-semibold">drop .pdf or .txt files here</p>
                    <p className="mt-2 text-sm text-ink-600">or click to browse. Text-based PDFs work best; scanned pages can't be read yet.</p>
                </div>
            </div>
        );
    }

    const paragraphs = text ? readableText(text) : [];
    const pageCount = text ? new Set(text.map(p => p.page)).size : 0;
    const multiPage = pageCount > 1;

    return (
        <div className="h-full flex flex-col min-h-0">
            {showTabs && (
                <div role="tablist" aria-label="Your documents" className="flex gap-1.5 overflow-x-auto px-4 pt-4 pb-2 lg:px-7">
                    {docs.map(d => {
                        const selected = d.filename === activeDoc;
                        return (
                            <button
                                key={d.id}
                                type="button"
                                role="tab"
                                aria-selected={selected}
                                onClick={() => onSelect(d.filename)}
                                className={`flex min-h-[36px] shrink-0 items-center gap-2 rounded px-3 font-mono text-[12.5px] transition-colors
                                    ${selected ? 'border-[1.5px] border-ink-900 bg-paper-50 font-medium text-ink-900' : 'border border-dashed border-rule-300 text-ink-600 hover:bg-paper-50/70'}`}
                            >
                                <StatusIcon status={d.status} />
                                <span className="max-w-[240px] truncate">{d.filename}</span>
                            </button>
                        );
                    })}
                </div>
            )}

            <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto px-4 pb-8 pt-2 lg:px-7">
                {!active && <p className="py-10 text-center font-mono text-sm text-ink-600">pick a document to read it here.</p>}

                {active?.status === 'processing' && (
                    <div className="mx-auto flex max-w-[760px] items-center gap-3 rounded border-[1.5px] border-dashed border-rule-300 bg-paper-50 p-6 font-mono text-sm text-ink-600">
                        <Loader2 className="h-5 w-5 animate-spin text-accent-600" aria-hidden="true" />
                        reading and indexing {active.filename}… it will appear here when it's ready.
                    </div>
                )}

                {active?.status === 'error' && (
                    <div className="mx-auto max-w-[760px] rounded border-[1.5px] border-dashed border-warn-200 bg-warn-50 p-6">
                        <p className="font-semibold text-warn-700">{active.error || 'This file could not be processed.'}</p>
                        <p className="mt-2 text-warn-900">Scanned pages are images, so there is no text to search. Try a text-based PDF or a TXT export.</p>
                        {onDelete && <DeleteButton doc={active} onDelete={onDelete} className="mt-4" />}
                    </div>
                )}

                {ready && loadError && <p className="py-10 text-center text-warn-700">{loadError}</p>}

                {ready && !text && !loadError && (
                    <div className="mx-auto max-w-[760px] space-y-3 rounded border-[1.5px] border-dashed border-rule-300 bg-paper-50 p-10" aria-label="Loading document">
                        {[90, 100, 80, 95].map((w, i) => (
                            <div key={i} className="h-3.5 animate-pulse rounded-sm bg-paper-300" style={{ width: `${w}%` }} />
                        ))}
                    </div>
                )}

                {ready && text && (
                    <article className="mx-auto max-w-[760px] rounded-sm border-[1.5px] border-dashed border-[#9C9588] bg-paper-50 shadow-[0_1px_0_#D9D3C7,0_18px_40px_rgba(40,34,20,0.08)]">
                        <div className="flex flex-wrap items-center gap-3 border-b border-dashed border-rule-200 px-5 py-3 font-mono text-xs text-ink-600 sm:px-6">
                            <span>{text.length} passage{text.length === 1 ? '' : 's'}{pageCount > 0 && ` · ${pageCount} page${pageCount === 1 ? '' : 's'}`}</span>
                            <span className="flex-1" />
                            {onDelete && <DeleteButton doc={active} onDelete={onDelete} />}
                        </div>

                        <div className="px-6 py-7 font-mono text-[14.5px] leading-[1.85] text-[#24231F] sm:px-10 sm:py-8">
                            {text.map((p, i) => {
                                const isHit = focus?.source === active.filename && focus.chunkId === p.chunk_id;
                                const pageStart = multiPage && (i === 0 || text[i - 1].page !== p.page);
                                return (
                                    <div key={p.chunk_id}>
                                        {pageStart && (i === 0 ? (
                                            <p className="mb-3 text-[11.5px] uppercase tracking-[0.14em] text-ink-600">— page {p.page} —</p>
                                        ) : (
                                            <div role="separator" className="my-7 flex items-center gap-3 text-[11.5px] uppercase tracking-[0.14em] text-ink-600">
                                                <span className="flex-1 border-t-[1.5px] border-dashed border-rule-200" />page {p.page}<span className="flex-1 border-t-[1.5px] border-dashed border-rule-200" />
                                            </div>
                                        ))}
                                        {isHit ? (
                                            <p className="mb-4 whitespace-pre-line">
                                                <mark
                                                    data-hit=""
                                                    className="scroll-my-24 bg-highlight px-[3px] py-0.5 text-ink-900 shadow-[inset_0_-2px_0_#D9A900] [box-decoration-break:clone]"
                                                >
                                                    {paragraphs[i]}
                                                </mark>
                                                <span className="ml-1.5 inline-block min-w-[20px] rounded-[3px] bg-accent-600 px-1.5 text-center align-[2px] text-[11px] font-semibold leading-[18px] text-white">{focus.n}</span>
                                            </p>
                                        ) : (
                                            <p className="mb-4 whitespace-pre-line">{paragraphs[i]}</p>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </article>
                )}
            </div>
        </div>
    );
}

function DeleteButton({ doc, onDelete, className = '' }: { doc: DocInfo; onDelete: (doc: DocInfo) => void; className?: string }) {
    return (
        <button
            type="button"
            onClick={() => onDelete(doc)}
            aria-label={`Delete ${doc.filename}`}
            className={`inline-flex min-h-[32px] items-center gap-1.5 rounded border border-dashed border-[#B5523A] px-2.5 font-mono text-xs text-[#9A3B25] hover:bg-[#FBEDE8] ${className}`}
        >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />delete
        </button>
    );
}

export function StatusIcon({ status }: { status: DocInfo['status'] }) {
    if (status === 'success') return <CheckCircle2 className="h-4 w-4 shrink-0 text-accent-600" aria-label="Ready" />;
    if (status === 'error') return <AlertCircle className="h-4 w-4 shrink-0 text-warn-700" aria-label="Needs attention" />;
    return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-ink-600" aria-label="Indexing" />;
}
