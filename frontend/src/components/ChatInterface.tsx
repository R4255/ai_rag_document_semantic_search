import { useState, useRef, useEffect } from 'react';
import { Send, Zap, BookOpen, User, FastForward, Layers } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion } from 'framer-motion';
import axios from 'axios';

type Message = {
    role: 'user' | 'assistant';
    content: string;
    citations: any[];
    cacheHit?: boolean;
};

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        // Poll for cache updates while chatting
        const poll = setInterval(async () => {
             try {
                // Trigger a cache stats update if available
                await axios.get(`${API}/api/cache/stats`); 
             } catch(e) {}
        }, 2000);
        return () => clearInterval(poll);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const query = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: query, citations: [] }]);
        setIsLoading(true);

        // Initial placeholder for assistant message
        setMessages(prev => [...prev, { role: 'assistant', content: '', citations: [] }]);

        const clientId = localStorage.getItem('nexus_client_id') || 'unknown';

        try {
            const response = await fetch(`${API}/api/chat`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-User-ID': clientId 
                },
                body: JSON.stringify({ query }),
            });

            if (!response.body) throw new Error('No response body');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            let buffer = '';

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value, { stream: !done });
                buffer += chunkValue;
                
                // Process complete lines
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line

                for (const line of lines) {
                    if (line.trim().startsWith('data: ')) {
                        const jsonStr = line.replace('data: ', '').trim();
                        if (jsonStr === '[DONE]') break;
                        
                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.type === 'content') {
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    const lastMsg = newMsgs[newMsgs.length - 1];
                                    if (lastMsg.role === 'assistant') {
                                        // --- FIX: Safely cast text content ---
                                        const text = typeof data.content === 'string' 
                                            ? data.content 
                                            : JSON.stringify(data.content);
                                        lastMsg.content += text; 
                                    }
                                    return newMsgs;
                                });
                            } else if (data.type === 'citations') {
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    const lastMsg = newMsgs[newMsgs.length - 1];
                                    if (lastMsg.role === 'assistant') {
                                        lastMsg.citations = data.citations;
                                    }
                                    return newMsgs;
                                });
                            }
                        } catch (e) {
                            console.error("JSON Parse Error", e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: 'Error: Could not fetch response.', 
                citations: [] 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col min-h-0 bg-surface">
            {/* Scrollable messages */}
            <div className="flex-1 min-h-0 overflow-y-auto px-4 md:px-8 py-4 md:py-8 space-y-6">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center opacity-70 p-4">
                        <Zap className="w-16 h-16 text-primary mb-6 animate-pulse" />
                        <h3 className="text-xl md:text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-200 to-gray-500 mb-2">
                            Waiting for query...
                        </h3>
                        <p className="text-xs md:text-sm text-gray-500 max-w-sm">
                            Upload documents on the left, then ask questions here.
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div className={`flex w-full xl:max-w-[80%] space-x-3 md:space-x-4 ${msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                            <div className={`w-8 h-8 md:w-10 md:h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg ${msg.role === 'user'
                                ? 'bg-gray-800 border border-gray-700'
                                : 'bg-gradient-to-br from-primary to-secondary'
                                }`}>
                                {msg.role === 'user' ? <User className="w-5 h-5 text-gray-300" /> : <Layers className="w-5 h-5 text-white" />}
                            </div>

                            <div className={`flex flex-col space-y-3 max-w-[85%] md:max-w-[90%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                <div className={`px-4 py-3 md:px-6 md:py-4 rounded-2xl shadow-sm ${msg.role === 'user' ? 'bg-primary/10 border border-primary/20 text-gray-200' : 'bg-[#181825] border border-white/5 text-gray-300'
                                    }`}>
                                    <div className="prose prose-invert prose-sm md:prose-base prose-p:leading-relaxed max-w-none">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {msg.content || (isLoading && i === messages.length - 1 ? '...' : '')}
                                        </ReactMarkdown>
                                    </div>
                                </div>

                                {msg.cacheHit && (
                                    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex items-center text-[10px] text-green-400 bg-green-900/20 px-3 py-1 rounded-full border border-green-500/30">
                                        <FastForward className="w-3 h-3 mr-1" /> Served instantly from Semantic Cache (Lat: &lt;10ms)
                                    </motion.div>
                                )}

                                {msg.citations && msg.citations.length > 0 && (
                                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="w-full">
                                        <div className="flex items-center text-xs text-gray-500 mb-2 uppercase tracking-wide font-semibold ml-1">
                                            <BookOpen className="w-3 h-3 mr-1" /> Sources
                                        </div>
                                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 w-full">
                                            {msg.citations.map((cit, idx) => (
                                                <div key={idx} className="bg-background/80 hover:bg-background border border-white/5 rounded-lg p-3 text-[11px] md:text-xs text-gray-400 transition-colors shadow-sm">
                                                    <span className="font-semibold text-gray-300 mb-1 block">📌 {cit.source} (Pg {cit.page})</span>
                                                    <span className="line-clamp-3 italic opacity-80">"{cit.snippet}"</span>
                                                </div>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                ))}
                <div ref={bottomRef} />
            </div>

            {/* Sticky input bar */}
            <div className="sticky bottom-0 z-20 bg-gradient-to-t from-background via-background/95 to-transparent px-4 md:px-8 pt-4 pb-[max(12px,env(safe-area-inset-bottom))]">
                <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        className="w-full bg-[#1e1e2d] border border-white/10 rounded-full px-6 py-4 pr-16 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/50 shadow-2xl"
                        placeholder="Ask anything..."
                        autoComplete="off"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 bottom-2 aspect-square rounded-full bg-primary flex items-center justify-center text-white hover:bg-secondary disabled:opacity-50 disabled:hover:bg-primary transition-colors"
                    >
                        {isLoading ? <Zap className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5 ml-1" />}
                    </button>
                </form>
            </div>
        </div>
    );
}
