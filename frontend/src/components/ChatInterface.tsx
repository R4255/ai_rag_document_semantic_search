import { useState, useRef, useEffect } from 'react';
import { Send, Zap, BookOpen, User, FastForward, Layers } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion } from 'framer-motion';

type Message = {
    role: 'user' | 'assistant';
    content: string;
    citations: any[];
    cacheHit?: boolean;
};

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const query = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: query, citations: [] }]);
        setIsLoading(true);

        // Initial placeholder for assistant message
        setMessages(prev => [...prev, { role: 'assistant', content: '', citations: [] }]);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (!response.body) throw new Error('No response body');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value);

                // Parse SSE stream
                const events = chunkValue.split('\n\n');
                for (const event of events) {
                    if (event.startsWith('data: ')) {
                        const dataStr = event.replace('data: ', '');
                        try {
                            const payload = JSON.parse(dataStr);
                            if (payload.type === 'content') {
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    const last = newMsgs[newMsgs.length - 1];
                                    last.content += payload.content;
                                    return newMsgs;
                                });
                            } else if (payload.type === 'citations') {
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    const last = newMsgs[newMsgs.length - 1];
                                    last.citations = payload.citations;
                                    return newMsgs;
                                });
                            } else if (payload.type === 'cache_hit') {
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    const last = newMsgs[newMsgs.length - 1];
                                    last.cacheHit = true;
                                    return newMsgs;
                                });
                            }
                        } catch (err) {
                            console.error("Parse err", err, dataStr);
                        }
                    }
                }
            }
        } catch (error) {
            console.error(error);
            setMessages(prev => {
                const newMsgs = [...prev];
                const last = newMsgs[newMsgs.length - 1];
                last.content = "Connection error. Ensure the backend is running (`uvicorn app:app`) and GOOGLE_API_KEY is set in .env";
                return newMsgs;
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-surface relative">
            <div className="flex-1 overflow-y-auto p-8 space-y-6 pb-32">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center opacity-70">
                        <Zap className="w-16 h-16 text-primary mb-6 animate-pulse" />
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-200 to-gray-500 mb-2">
                            Waiting for query...
                        </h3>
                        <p className="text-sm text-gray-500 max-w-sm">
                            Upload documents on the left, then ask questions here. The system uses LRU semantic caching to accelerate redundant questions.
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
                        <div className={`flex max-w-[80%] space-x-4 ${msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg ${msg.role === 'user'
                                ? 'bg-gray-800 border border-gray-700'
                                : 'bg-gradient-to-br from-primary to-secondary'
                                }`}>
                                {msg.role === 'user' ? <User className="w-5 h-5 text-gray-300" /> : <Layers className="w-5 h-5 text-white" />}
                            </div>

                            <div className={`flex flex-col space-y-3 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                <div className={`px-6 py-4 rounded-2xl shadow-sm ${msg.role === 'user' ? 'bg-primary/10 border border-primary/20 text-gray-200' : 'bg-[#181825] border border-white/5 text-gray-300'
                                    }`}>
                                    <div className="prose prose-invert prose-p:leading-relaxed max-w-none text-sm">
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
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                                            {msg.citations.map((cit, idx) => (
                                                <div key={idx} className="bg-background/80 hover:bg-background border border-white/5 rounded-lg p-3 text-xs text-gray-400 transition-colors shadow-sm">
                                                    <span className="font-semibold text-gray-300 mb-1 block">📌 {cit.source} (Pg {cit.page})</span>
                                                    <span className="line-clamp-2 italic opacity-80">"{cit.snippet}"</span>
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

            <div className="absolute bottom-0 w-full p-6 bg-gradient-to-t from-surface via-surface to-transparent">
                <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative group">
                    <input
                        type="text"
                        className="w-full bg-[#1e1e2d] border border-white/10 rounded-full px-6 py-4 pr-16 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/50 shadow-2xl transition-all group-hover:border-white/20"
                        placeholder="Ask anything..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        autoComplete="off"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-2 bottom-2 aspect-square rounded-full bg-primary flex items-center justify-center text-white hover:bg-secondary disabled:opacity-50 disabled:hover:bg-primary transition-colors cursor-pointer"
                    >
                        {isLoading ? <Zap className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5 ml-1" />}
                    </button>
                </form>
            </div>
        </div>
    );
}
