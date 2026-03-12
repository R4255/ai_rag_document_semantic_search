// CacheStats.tsx
import { Database, Activity, Target } from 'lucide-react';

interface StatsProps {
    stats: {
        hits: number;
        misses: number;
        size: number;
        capacity: number;
    };
}

export default function CacheStats({ stats }: StatsProps) {
    const hitRate = stats.hits + stats.misses > 0
        ? Math.round((stats.hits / (stats.hits + stats.misses)) * 100)
        : 0;

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
                <div className="bg-surface/50 p-4 rounded-xl border border-white/5 flex flex-col justify-between hover:bg-surface/80 transition-colors">
                    <div className="flex items-center text-gray-500 mb-2">
                        <Target className="w-4 h-4 mr-1 text-green-400" />
                        <span className="text-xs uppercase font-medium tracking-wide">Hits</span>
                    </div>
                    <span className="text-2xl font-bold font-mono text-gray-200">{stats.hits}</span>
                </div>

                <div className="bg-surface/50 p-4 rounded-xl border border-white/5 flex flex-col justify-between hover:bg-surface/80 transition-colors">
                    <div className="flex items-center text-gray-500 mb-2">
                        <Activity className="w-4 h-4 mr-1 text-red-400" />
                        <span className="text-xs uppercase font-medium tracking-wide">Misses</span>
                    </div>
                    <span className="text-2xl font-bold font-mono text-gray-200">{stats.misses}</span>
                </div>
            </div>

            <div className="bg-surface/50 p-4 rounded-xl border border-white/5 relative overflow-hidden group">
                <div className="flex justify-between items-center mb-3 relative z-10">
                    <div className="flex items-center text-gray-400 text-xs uppercase font-medium">
                        <Database className="w-4 h-4 mr-2 text-primary" /> Memory Usage
                    </div>
                    <span className="text-xs font-mono bg-gray-800 px-2 py-0.5 rounded text-gray-300">
                        {stats.size} / {stats.capacity}
                    </span>
                </div>
                <div className="w-full bg-black/40 h-2 rounded-full overflow-hidden relative z-10">
                    <div
                        className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500 ease-out"
                        style={{ width: `${(stats.size / (stats.capacity || 100)) * 100}%` }}
                    />
                </div>
                <div className="mt-3 text-[10px] text-gray-500 italic relative z-10">
                    *LRU semantic cache with {hitRate}% hit rate reducing LLM load by ~{hitRate}%.
                </div>
            </div>
        </div>
    );
}
