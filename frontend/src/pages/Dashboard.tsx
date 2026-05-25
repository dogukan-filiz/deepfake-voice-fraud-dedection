import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';
import {
  BarChart3, ChevronDown, Check, X, AlertTriangle, HelpCircle,
  Upload, Calendar, Mic, Square, ShieldCheck, ShieldAlert, ShieldX, Globe,
} from 'lucide-react';
import { type Locale, type Translations, getTranslations, detectLocale, persistLocale } from '../i18n';

type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

const RiskIcon: React.FC<{ level: RiskLevel; size?: number }> = ({ level, size = 18 }) => {
  switch (level) {
    case 'low': return <Check size={size} />;
    case 'medium': return <HelpCircle size={size} />;
    case 'high': return <AlertTriangle size={size} />;
    case 'critical': return <X size={size} />;
  }
};

interface CallRecord {
  call_id: string;
  authenticity_score: number;
  is_suspected_fraud: boolean;
  risk_level: RiskLevel;
  timestamp: string;
  notes?: string | null;
}

interface PredictionResult {
  call_id: string;
  authenticity_score: number;
  is_suspected_fraud: boolean;
  risk_level: RiskLevel;
  p_real: number;
  p_fake: number;
  spectral_residual: number;
  num_chunks: number;
  max_chunk_p_fake: number;
  timestamp: string;
}

interface HealthResponse {
  status: string;
  ffmpeg_available?: boolean;
  pid?: number;
}

function timeAgo(ts: string, locale: Locale): string {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (locale === 'tr') {
    if (mins < 1) return 'az önce';
    if (mins < 60) return `${mins}dk önce`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}sa önce`;
    return `${Math.floor(hrs / 24)}g önce`;
  }
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export const Dashboard: React.FC = () => {
  const [locale, setLocale] = useState<Locale>(detectLocale);
  const t: Translations = getTranslations(locale);

  const toggleLocale = () => {
    const next: Locale = locale === 'en' ? 'tr' : 'en';
    setLocale(next);
    persistLocale(next);
  };

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [showAllCalls, setShowAllCalls] = useState(false);

  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'processing'>('idle');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const recordingModeRef = useRef<'mediarecorder' | 'pcm' | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pcmSampleRateRef = useRef<number | null>(null);

  const api = axios.create({ baseURL: '/api' });

  const fetchCalls = async () => {
    try { setCalls((await api.get<CallRecord[]>('/calls')).data); } catch {}
  };

  useEffect(() => {
    fetchCalls();
    const iv = setInterval(fetchCalls, 5000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const refresh = async () => {
      try { setHealth((await api.get<HealthResponse>('/health')).data); } catch { setHealth(null); }
    };
    refresh();
    const iv = setInterval(refresh, 10000);
    return () => clearInterval(iv);
  }, []);

  const mergeFloat32 = (chunks: Float32Array[]) => {
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const m = new Float32Array(total);
    let off = 0;
    for (const c of chunks) { m.set(c, off); off += c.length; }
    return m;
  };

  const encodeWav16BitMono = (samples: Float32Array, sr: number) => {
    const bps = 2, ds = samples.length * bps;
    const buf = new ArrayBuffer(44 + ds), v = new DataView(buf);
    const w = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0,'RIFF'); v.setUint32(4,36+ds,true); w(8,'WAVE'); w(12,'fmt ');
    v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
    v.setUint32(24,sr,true); v.setUint32(28,sr*bps,true);
    v.setUint16(32,bps,true); v.setUint16(34,16,true); w(36,'data'); v.setUint32(40,ds,true);
    let o = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
  };

  const analyzeBlob = async (blob: Blob) => {
    setRecordingState('processing');
    try {
      if (!blob.size) { setError(t.errors.emptyRecording); return; }
      const mime = (blob.type || '').toLowerCase();
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('webm') ? 'webm' : mime.includes('wav') ? 'wav' : mime.includes('mp4') ? 'm4a' : 'bin';
      const fd = new FormData();
      fd.append('file', blob, `recording.${ext}`);
      const res = await api.post<PredictionResult>('/analyze', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(res.data); setError(null);
      await fetchCalls();
    } catch (e) {
      if (axios.isAxiosError(e)) {
        const detail = e.response?.data?.detail as string | undefined;
        setError(detail ?? (e.response ? t.errors.analysisFailed(e.response.status) : t.errors.backendUnreachable));
      } else { setError(t.errors.generic); }
    } finally { setRecordingState('idle'); }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const canChoose = typeof MediaRecorder !== 'undefined' && 'isTypeSupported' in MediaRecorder;
      const wavOk = canChoose ? MediaRecorder.isTypeSupported('audio/wav') : false;
      if (typeof MediaRecorder !== 'undefined' && wavOk) {
        const mr = new MediaRecorder(stream, { mimeType: 'audio/wav' });
        recordingModeRef.current = 'mediarecorder'; recorderRef.current = mr; audioChunksRef.current = [];
        mr.ondataavailable = (ev) => { if (ev.data.size > 0) audioChunksRef.current.push(ev.data); };
        mr.onstop = async () => { const b = new Blob(audioChunksRef.current, { type: 'audio/wav' }); stream.getTracks().forEach(t => t.stop()); streamRef.current = null; await analyzeBlob(b); };
        mr.start(); setRecordingState('recording'); setError(null); return;
      }
      recordingModeRef.current = 'pcm'; recorderRef.current = null; audioChunksRef.current = []; pcmChunksRef.current = [];
      const AC = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext | undefined;
      if (!AC) { stream.getTracks().forEach(t => t.stop()); streamRef.current = null; setError(t.errors.audioContextUnsupported); return; }
      const ctx = new AC(); audioContextRef.current = ctx; pcmSampleRateRef.current = ctx.sampleRate;
      const src = ctx.createMediaStreamSource(stream); sourceNodeRef.current = src;
      const proc = ctx.createScriptProcessor(4096, 1, 1); processorNodeRef.current = proc;
      proc.onaudioprocess = (ev) => { pcmChunksRef.current.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
      src.connect(proc); proc.connect(ctx.destination);
      setRecordingState('recording'); setError(null);
    } catch { setError(t.errors.micDenied); }
  };

  const stopRecording = () => {
    if (recordingState !== 'recording') return;
    if (recordingModeRef.current === 'mediarecorder') { recorderRef.current?.stop(); setRecordingState('processing'); return; }
    if (recordingModeRef.current === 'pcm') {
      setRecordingState('processing');
      try { processorNodeRef.current?.disconnect(); sourceNodeRef.current?.disconnect(); } catch {}
      processorNodeRef.current = null; sourceNodeRef.current = null;
      streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null;
      audioContextRef.current?.close().catch(() => null); audioContextRef.current = null;
      const samples = mergeFloat32(pcmChunksRef.current); pcmChunksRef.current = [];
      void analyzeBlob(encodeWav16BitMono(samples, pcmSampleRateRef.current ?? 48000));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { if (e.target.files?.[0]) setSelectedFile(e.target.files[0]); };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true); setError(null);
    try {
      const fd = new FormData(); fd.append('file', selectedFile);
      const res = await api.post<PredictionResult>('/analyze', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(res.data); await fetchCalls();
    } catch (e: any) {
      if (axios.isAxiosError(e)) setError(e.response?.data?.detail ?? t.errors.analysisFailed(e.response?.status ?? '??'));
      else setError(t.errors.generic);
    } finally { setLoading(false); }
  };

  const totalCalls = calls.length;
  const flaggedCalls = calls.filter(c => c.is_suspected_fraud).length;
  const cleanCalls = totalCalls - flaggedCalls;

  const chartData = [...calls].reverse().map((c) => ({
    time: new Date(c.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    score: parseFloat(c.authenticity_score.toFixed(3)),
  }));

  const miniBars = [...calls].reverse().slice(0, 30).map(c => ({ id: c.call_id, level: c.risk_level, score: c.authenticity_score }));

  const visibleCalls = showAllCalls ? calls : calls.slice(0, 6);

  const riskLabel = (level: RiskLevel) => t.risk[level];

  return (
    <div className="app-shell">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand">
          <img src="/logo.png" alt="VoxGuard" className="brand-logo" />
          <span className="brand-name">{t.brand}</span>
        </div>
        <div className="nav-right">
          <button className="lang-toggle" onClick={toggleLocale}>
            <Globe size={14} />
            {t.nav.langSwitch}
          </button>
        </div>
      </nav>

      {/* Dashboard */}
      <div className="dashboard">
        <div className="main-col">
          {/* Score Tracker */}
          <section className="card card-hero">
            <div className="card-header">
              <div className="card-title-group">
                <div className="card-icon"><BarChart3 size={20} /></div>
                <div>
                  <h2>{t.tracker.title}</h2>
                  <p className="card-subtitle">{t.tracker.subtitle}</p>
                </div>
              </div>
              <div className="dropdown-select">
                <span>{t.tracker.session}</span>
                <ChevronDown size={14} />
              </div>
            </div>
            <div className="hero-chart-area">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.6} />
                    <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 13, color: '#fff' }} labelStyle={{ color: 'rgba(255,255,255,0.6)', fontSize: 11 }} itemStyle={{ color: '#fff' }} />
                    <Line type="monotone" dataKey="score" stroke="#6b7fb8" strokeWidth={2.5} dot={{ r: 4, fill: '#6b7fb8', stroke: '#fff', strokeWidth: 2 }} activeDot={{ r: 6 }} name="Score" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-chart">{t.tracker.emptyChart}</div>
              )}
            </div>
            <div className="hero-stats">
              <div>
                <div className={`hero-stat-change ${flaggedCalls > 0 ? 'negative' : 'positive'}`}>
                  {flaggedCalls > 0 ? flaggedCalls : totalCalls > 0 ? <ShieldCheck size={28} /> : '0'}
                </div>
                <div className="hero-stat-label">
                  {flaggedCalls > 0 ? t.tracker.flagged(flaggedCalls) : totalCalls > 0 ? t.tracker.allClean : t.tracker.waiting}
                </div>
              </div>
            </div>
          </section>

          {/* Action Cards */}
          <div className="cards-row">
            {/* Live Recording */}
            <section className="card card-action">
              <h3>{t.recording.title}</h3>
              <p className="card-action-desc">{t.recording.desc}</p>
              <div className="recording-controls">
                {recordingState === 'recording' ? (
                  <button className="btn btn-danger" onClick={stopRecording}><Square size={14} /> {t.recording.stop}</button>
                ) : (
                  <button className="btn btn-primary" onClick={startRecording} disabled={recordingState === 'processing'}>
                    <Mic size={14} /> {recordingState === 'processing' ? t.recording.processing : t.recording.start}
                  </button>
                )}
                <div className="recording-status">
                  <div className={`status-dot ${recordingState}`} />
                  <span>
                    {recordingState === 'idle' && t.recording.ready}
                    {recordingState === 'recording' && t.recording.recordingStatus}
                    {recordingState === 'processing' && t.recording.analyzing}
                  </span>
                </div>
              </div>
              {health && (
                <div className="system-info">
                  <span className="system-tag">
                    <span style={{ color: health.status === 'ok' ? '#22c55e' : '#ef4444' }}>{'●'}</span>
                    {t.system.backendStatus(health.status)}
                  </span>
                </div>
              )}
            </section>

            {/* Upload */}
            <section className="card card-action">
              <h3>{t.upload.title}</h3>
              <p className="card-action-desc">{t.upload.desc}</p>
              <div className="upload-zone">
                <input type="file" accept="audio/*" onChange={handleFileChange} />
                <div className="upload-icon"><Upload size={24} color="#94a3b8" /></div>
                <div className="upload-text">{t.upload.selectFile}</div>
              </div>
              {selectedFile && <div className="upload-filename">{selectedFile.name}</div>}
              <button className="btn btn-primary" onClick={handleUpload} disabled={!selectedFile || loading} style={{ marginTop: '0.75rem', width: '100%' }}>
                {loading ? t.upload.analyzeLoading : t.upload.analyze}
              </button>
            </section>

            {/* Detection Summary */}
            <section className="card">
              <div className="card-header">
                <h3>{t.summary.title}</h3>
                <span className="stat-date"><Calendar size={13} /> {t.summary.thisSession}</span>
              </div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-label">{t.summary.totalScanned}</span>
                  <span className="stat-value">{totalCalls}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">{t.summary.flagged}</span>
                  <span className="stat-value flagged">{flaggedCalls}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">{t.summary.clean}</span>
                  <span className="stat-value clean">{cleanCalls}</span>
                </div>
              </div>
              {miniBars.length > 0 && (
                <div className="mini-bars">
                  {miniBars.map(b => (
                    <div key={b.id} className={`mini-bar ${b.level === 'low' || b.level === 'medium' ? 'safe' : 'danger'}`} style={{ height: `${Math.max(4, b.score * 30)}px` }} />
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>

        {/* Sidebar */}
        <div className="side-col">
          {/* Recent Analyses */}
          <section className="card">
            <div className="card-header">
              <h3>{t.recent.title}</h3>
              {calls.length > 6 && (
                <button className="link-btn" onClick={() => setShowAllCalls(prev => !prev)}>
                  {showAllCalls ? t.recent.collapse : t.recent.seeAll}
                </button>
              )}
            </div>
            <div className="analyses-list">
              {calls.length === 0 && (
                <div className="empty-list">{t.recent.empty}</div>
              )}
              {visibleCalls.map(c => {
                const shortId = c.call_id.split('-').pop()?.slice(0, 8) ?? '';
                return (
                  <div className="analysis-item" key={c.call_id}>
                    <div className={`item-icon ${c.risk_level}`}><RiskIcon level={c.risk_level} /></div>
                    <div className="item-info">
                      <div className="item-title">
                        <span>{t.recent.callPrefix}{shortId}</span>
                        <span className={`badge ${c.risk_level}`}>{riskLabel(c.risk_level)}</span>
                      </div>
                      <span className="item-score">{t.recent.scorePrefix}{c.authenticity_score.toFixed(3)}</span>
                    </div>
                    <div className="item-meta"><span className="item-time">{timeAgo(c.timestamp, locale)}</span></div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Latest Result */}
          {result && (
            <section className="card card-result">
              <div className="card-header">
                <h3>{t.result.title}</h3>
                <span className="stat-date">{new Date(result.timestamp).toLocaleTimeString()}</span>
              </div>
              <div>
                <div className="result-row"><span className="result-label">{t.result.authenticity}</span><span className="result-value">{(result.authenticity_score * 100).toFixed(1)}%</span></div>
                <div className="result-row"><span className="result-label">{t.result.riskLevel}</span><span className={`result-value result-risk ${result.risk_level}`}>{riskLabel(result.risk_level)}</span></div>
                <div className="result-row"><span className="result-label">{t.result.probabilities}</span><span className="result-value">{result.p_real.toFixed(3)} / {result.p_fake.toFixed(3)}</span></div>
                <div className="result-row"><span className="result-label">{t.result.spectralAnomaly}</span><span className="result-value">{result.spectral_residual.toFixed(3)}</span></div>
                {result.num_chunks > 0 && (
                  <div className="result-row"><span className="result-label">{t.result.chunksWorst}</span><span className="result-value">{result.num_chunks} / {result.max_chunk_p_fake.toFixed(3)}</span></div>
                )}
              </div>
              <div className={`result-verdict ${result.is_suspected_fraud ? 'fraud' : 'genuine'}`}>
                {result.is_suspected_fraud ? (<><ShieldX size={18} /> {t.result.suspectedDeepfake}</>) : (<><ShieldCheck size={18} /> {t.result.appearsGenuine}</>)}
              </div>
            </section>
          )}
        </div>
      </div>

      {error && (
        <div className="error-toast" onClick={() => setError(null)}>
          <ShieldAlert size={16} style={{ flexShrink: 0 }} /> {error}
        </div>
      )}
    </div>
  );
};
