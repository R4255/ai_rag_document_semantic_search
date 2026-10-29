import axios from 'axios';

export const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Distinct client ID for multi-tenancy without login
const getClientId = () => {
    let id = localStorage.getItem('nexus_client_id');
    if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem('nexus_client_id', id);
    }
    return id;
};

export const CLIENT_ID = getClientId();

// Every axios request sends the ID; fetch() calls add it themselves
axios.defaults.headers.common['X-User-ID'] = CLIENT_ID;

export type DocStatus = 'processing' | 'success' | 'error';

export interface DocInfo {
    id: string;
    filename: string;
    chunks: number;
    timestamp: number;
    status: DocStatus;
    error?: string | null;
}

export interface Citation {
    n: number;
    chunk_id: string;
    source: string;
    page: number;
    snippet: string;
}

export interface Passage {
    chunk_id: string;
    page: number;
    text: string;
}

/** Which passage to show highlighted in the document reader. */
export interface Focus {
    source: string;
    chunkId: string;
    n: number;
}

export async function fetchDocuments(): Promise<DocInfo[]> {
    const r = await axios.get(`${API}/api/documents`);
    return r.data.documents || [];
}

export async function deleteDocument(id: string): Promise<void> {
    await axios.delete(`${API}/api/documents/${id}`);
}

export async function fetchPassages(name: string): Promise<Passage[]> {
    const r = await axios.get(`${API}/api/document`, { params: { name } });
    return r.data.chunks || [];
}

/**
 * Some PDFs extract one word per line ("Build⏎ ⏎a⏎ ⏎simple"), with two or more blank
 * lines marking a real paragraph. Rejoin those words; leave normal text alone.
 * Mirrors rag_service._tidy_text, which fixes new uploads; this covers ones stored before it.
 */
export function tidyText(text: string): string {
    const lines = text.split('\n');
    const words = lines.map(l => l.trim()).filter(Boolean);
    if (words.length < 8 || words.filter(w => !/\s/.test(w)).length / words.length < 0.8) return text;
    let out = '';
    let blank = 0;
    for (const line of lines) {
        const w = line.trim();
        if (!w) { blank++; continue; }
        if (out) out += blank >= 2 ? '\n\n' : /^[-•*]$/.test(w) ? '\n' : ' ';
        out += w;
        blank = 0;
    }
    return out;
}

/**
 * Chunks overlap by up to 200 characters, so consecutive chunks repeat text.
 * For reading, drop the repeated start of each chunk.
 */
export function readableText(passages: Passage[]): string[] {
    const texts = passages.map(p => tidyText(p.text));
    return texts.map((text, i) => {
        if (i === 0 || passages[i - 1].page !== passages[i].page) return text;
        const prev = texts[i - 1];
        const max = Math.min(400, prev.length, text.length);
        for (let k = max; k >= 20; k--) {
            if (prev.endsWith(text.slice(0, k))) return text.slice(k).trimStart();
        }
        return text;
    });
}
