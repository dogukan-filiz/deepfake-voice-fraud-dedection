import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts';
import {
  BarChart3, Check, X, AlertTriangle, HelpCircle,
  Upload, Calendar, Mic, Square, ShieldCheck, ShieldAlert, ShieldX, Globe,
  FlaskConical, FileAudio, Loader2, Trash2,
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
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const [lastBlobInfo, setLastBlobInfo] = useState<{mime: string; sizeKB: number; durationSec: number | null} | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const recordingModeRef = useRef<'mediarecorder' | 'pcm' | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pcmSampleRateRef = useRef<number | null>(null);
  const recordingStartRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Analysis progress bar.
  // Phase 1 (0->75%): fills over a time proportional to audio duration.
  // Phase 2 (75->100%): races to full once the backend result actually arrives,
  // so the bar never claims "done" before the analysis is truly complete.
  const [progress, setProgress] = useState(0);
  const [progressActive, setProgressActive] = useState(false);
  const progressRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const doneRef = useRef(false);

  const setProg = (v: number) => { progressRef.current = v; setProgress(v); };

  // Read audio duration (seconds) from a Blob/File via metadata. Resolves null on failure.
  const getAudioDuration = (file: Blob): Promise<number | null> => new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(file);
      const a = new Audio();
      a.preload = 'metadata';
      a.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(isFinite(a.duration) ? a.duration : null); };
      a.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      a.src = url;
    } catch { resolve(null); }
  });

  const startProgress = (estSec: number | null) => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    doneRef.current = false;
    setProgressActive(true);
    setProg(0);
    // Phase 1 duration tracks expected inference time (~0.13s per second of
    // audio: chunked at 4.04s/chunk, ~0.5s inference each). Clamped to [0.8s, 30s]
    // so the bar reaches ~75% around when the real result arrives, not before.
    const dur = Math.min(30000, Math.max(800, (estSec ?? 3) * 130));
    const start = performance.now();
    const tick = (now: number) => {
      if (doneRef.current) return;
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 2); // ease-out
      const v = eased * 75;
      if (v > progressRef.current) setProg(v);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else setProg(75); // hold at 75 until the result arrives
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const finishProgress = () => {
    doneRef.current = true;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const from = progressRef.current;
    const start = performance.now();
    const dur = 400;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      setProg(from + (100 - from) * t);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else setTimeout(() => { setProgressActive(false); setProg(0); }, 300);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

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

  const analyzeBlob = async (blob: Blob, durationSec?: number) => {
    setRecordingState('processing');
    startProgress(durationSec ?? null);
    try {
      if (!blob.size) { setError(t.errors.emptyRecording); return; }
      const mime = (blob.type || '').toLowerCase();
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('webm') ? 'webm' : mime.includes('wav') ? 'wav' : mime.includes('mp4') ? 'm4a' : 'bin';
      setLastBlobInfo({ mime: mime || 'unknown', sizeKB: Math.round(blob.size / 1024), durationSec: durationSec ?? null });
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
    } finally { finishProgress(); setRecordingState('idle'); }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      if (typeof MediaRecorder !== 'undefined') {
        // Fallback chain: Chrome/Firefox support webm+opus natively; Safari may fall through to PCM path
        const MIME_CHAIN = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/wav'];
        const chosenMime = MIME_CHAIN.find(m => MediaRecorder.isTypeSupported(m)) ?? '';
        const mrOpts = chosenMime ? { mimeType: chosenMime } : {};
        const mr = new MediaRecorder(stream, mrOpts);
        const actualMime = mr.mimeType || chosenMime || 'audio/webm';
        recordingModeRef.current = 'mediarecorder'; recorderRef.current = mr; audioChunksRef.current = [];
        recordingStartRef.current = Date.now(); setRecordingElapsed(0);
        timerRef.current = setInterval(() => setRecordingElapsed(Math.floor((Date.now() - (recordingStartRef.current ?? Date.now())) / 1000)), 500);
        mr.ondataavailable = (ev) => { if (ev.data.size > 0) audioChunksRef.current.push(ev.data); };
        mr.onstop = async () => {
          if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
          const dur = Math.round((Date.now() - (recordingStartRef.current ?? Date.now())) / 1000);
          const b = new Blob(audioChunksRef.current, { type: actualMime });
          stream.getTracks().forEach(t => t.stop()); streamRef.current = null;
          await analyzeBlob(b, dur);
        };
        mr.start(1000); setRecordingState('recording'); setError(null); return;
      }
      // TODO: Future improvement — replace ScriptProcessor fallback with AudioWorklet.
      recordingModeRef.current = 'pcm'; recorderRef.current = null; audioChunksRef.current = []; pcmChunksRef.current = [];
      recordingStartRef.current = Date.now(); setRecordingElapsed(0);
      timerRef.current = setInterval(() => setRecordingElapsed(Math.floor((Date.now() - (recordingStartRef.current ?? Date.now())) / 1000)), 500);
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
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      try { processorNodeRef.current?.disconnect(); sourceNodeRef.current?.disconnect(); } catch {}
      processorNodeRef.current = null; sourceNodeRef.current = null;
      streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null;
      audioContextRef.current?.close().catch(() => null); audioContextRef.current = null;
      const samples = mergeFloat32(pcmChunksRef.current); pcmChunksRef.current = [];
      const dur = Math.round((Date.now() - (recordingStartRef.current ?? Date.now())) / 1000);
      void analyzeBlob(encodeWav16BitMono(samples, pcmSampleRateRef.current ?? 48000), dur);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { if (e.target.files?.[0]) setSelectedFile(e.target.files[0]); };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true); setError(null);
    startProgress(await getAudioDuration(selectedFile));
    try {
      const fd = new FormData(); fd.append('file', selectedFile);
      const res = await api.post<PredictionResult>('/analyze', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(res.data); await fetchCalls();
    } catch (e: any) {
      if (axios.isAxiosError(e)) setError(e.response?.data?.detail ?? t.errors.analysisFailed(e.response?.status ?? '??'));
      else setError(t.errors.generic);
    } finally { finishProgress(); setLoading(false); }
  };

  const totalCalls = calls.length;
  const flaggedCalls = calls.filter(c => c.is_suspected_fraud).length;
  const cleanCalls = totalCalls - flaggedCalls;

  // Display-only rescale: backend AUTH_THRESHOLD (0.01) maps to chart midline (0.5)
  // so real samples (just above threshold) don't look tiny next to high-confidence ones.
  // Raw score still shown in tooltip.
  const DISPLAY_THRESHOLD = 0.01;
  const rescaleForDisplay = (score: number): number => {
    if (score <= DISPLAY_THRESHOLD) return (score / DISPLAY_THRESHOLD) * 0.5;
    return 0.5 + ((score - DISPLAY_THRESHOLD) / (1 - DISPLAY_THRESHOLD)) * 0.5;
  };

  const chartData = [...calls].reverse().map((c) => ({
    time: new Date(c.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    score: parseFloat(rescaleForDisplay(c.authenticity_score).toFixed(3)),
    rawScore: parseFloat(c.authenticity_score.toFixed(3)),
  }));

  const miniBars = [...calls].reverse().slice(0, 30).map(c => ({ id: c.call_id, level: c.risk_level, score: c.authenticity_score }));

  const visibleCalls = showAllCalls ? calls : calls.slice(0, 6);

  const riskLabel = (level: RiskLevel) => t.risk[level];

  // Test Library modal
  const [showLibrary, setShowLibrary] = useState(false);
  const [libraryFiles, setLibraryFiles] = useState<{ real: string[]; fake: string[] }>({ real: [], fake: [] });
  const [libraryLoading, setLibraryLoading] = useState<string | null>(null);

  const openLibrary = async () => {
    setShowLibrary(true);
    try {
      const res = await api.get<{ real: string[]; fake: string[] }>('/test-library');
      setLibraryFiles(res.data);
    } catch {
      setLibraryFiles({ real: [], fake: [] });
    }
  };

  const analyzeTestFile = async (category: 'real' | 'fake', filename: string) => {
    const key = `${category}/${filename}`;
    setLibraryLoading(key);
    startProgress(null); // test-library duration unknown -> default estimate
    try {
      const res = await api.post<PredictionResult>(`/analyze-test?category=${category}&filename=${encodeURIComponent(filename)}`);
      setResult(res.data);
      setError(null);
      await fetchCalls();
      setShowLibrary(false);
    } catch (e) {
      if (axios.isAxiosError(e)) {
        setError(e.response?.data?.detail ?? t.errors.generic);
      } else {
        setError(t.errors.generic);
      }
    } finally {
      finishProgress();
      setLibraryLoading(null);
    }
  };

  const deleteCall = async (callId: string) => {
    try {
      await api.delete(`/calls/${encodeURIComponent(callId)}`);
      await fetchCalls();
      if (result?.call_id === callId) setResult(null);
    } catch {}
  };

  const deleteAllCalls = async () => {
    if (!confirm(t.recent.confirmDeleteAll)) return;
    try {
      await api.delete('/calls');
      setCalls([]);
      setResult(null);
    } catch {}
  };

  return (
    <div className="app-shell">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand">
          <img src="/logo.png" alt="VoxGuard" className="brand-logo" />
          <span className="brand-name">{t.brand}</span>
        </div>
        <div className="nav-right">
          <button className="lang-toggle" onClick={openLibrary}>
            <FlaskConical size={14} />
            {t.testLibrary.openBtn}
          </button>
          <button className="lang-toggle" onClick={toggleLocale}>
            <Globe size={14} />
            {t.nav.langSwitch}
          </button>
        </div>
      </nav>

      {/* Dashboard */}
      <div className="dashboard">
        {/* Analysis progress bar */}
        {progressActive && (
          <div className="analyze-progress" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}>
            <div className="analyze-progress-head">
              <span>{t.upload.analyzeLoading}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="analyze-progress-track">
              <div className="analyze-progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {/* Verdict Banner - visible immediately after analysis */}
        {result && (
          <section className={`verdict-banner ${result.is_suspected_fraud ? 'fraud' : 'genuine'}`}>
            <div className="verdict-icon">
              {result.is_suspected_fraud ? <ShieldX size={36} /> : <ShieldCheck size={36} />}
            </div>
            <div className="verdict-content">
              <div className="verdict-title">
                {result.is_suspected_fraud ? t.result.suspectedDeepfake : t.result.appearsGenuine}
              </div>
            </div>
          </section>
        )}

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
            <span className="session-label">{t.tracker.session}</span>
          </div>
          <div className="hero-chart-area">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.6} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => v === 0.5 ? 'threshold' : v === 1 ? 'real' : 'fake'} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={70} />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8, fontSize: 13, color: '#fff' }}
                    labelStyle={{ color: 'rgba(255,255,255,0.6)', fontSize: 11 }}
                    itemStyle={{ color: '#fff' }}
                    formatter={(_v: number, _n: string, item: any) => [item?.payload?.rawScore?.toFixed(3) ?? '-', 'p_real']}
                  />
                  <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="4 4" strokeOpacity={0.6} />
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

        {/* Action Cards Row */}
        <div className="cards-row">
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
                  {recordingState === 'recording' && `${t.recording.recordingStatus} ${recordingElapsed}s`}
                  {recordingState === 'processing' && t.recording.analyzing}
                </span>
              </div>
              {lastBlobInfo && recordingState === 'idle' && (
                <div className="blob-debug" style={{fontSize:'0.7rem',color:'#64748b',marginTop:'0.25rem',display:'flex',gap:'0.5rem',flexWrap:'wrap'}}>
                  <span>{lastBlobInfo.mime || 'unknown'}</span>
                  <span>·</span>
                  <span>{lastBlobInfo.sizeKB} KB</span>
                  {lastBlobInfo.durationSec !== null && <><span>·</span><span>{lastBlobInfo.durationSec}s</span></>}
                </div>
              )}
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

        {/* Bottom Row: Recent Analyses + Latest Result side by side */}
        <div className="bottom-row">
          <section className="card card-recent">
            <div className="card-header">
              <h3>{t.recent.title}</h3>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                {calls.length > 0 && (
                  <button className="link-btn danger-link" onClick={deleteAllCalls}>
                    <Trash2 size={12} /> {t.recent.deleteAll}
                  </button>
                )}
                {calls.length > 6 && (
                  <button className="link-btn" onClick={() => setShowAllCalls(prev => !prev)}>
                    {showAllCalls ? t.recent.collapse : t.recent.seeAll}
                  </button>
                )}
              </div>
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
                    <div className="item-meta">
                      <span className="item-time">{timeAgo(c.timestamp, locale)}</span>
                      <button className="item-delete" onClick={() => deleteCall(c.call_id)}><Trash2 size={13} /></button>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

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

      {/* Test Library Modal */}
      {showLibrary && (
        <div className="modal-overlay" onClick={() => setShowLibrary(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{t.testLibrary.title}</h2>
              <button className="modal-close" onClick={() => setShowLibrary(false)}><X size={18} /></button>
            </div>
            <div className="modal-body">
              {libraryFiles.real.length === 0 && libraryFiles.fake.length === 0 ? (
                <div className="empty-list">{t.testLibrary.empty}</div>
              ) : (
                <div className="library-grid">
                  <div className="library-col">
                    <h3 className="library-col-title genuine-text">
                      <ShieldCheck size={16} /> {t.testLibrary.real} ({libraryFiles.real.length})
                    </h3>
                    <div className="library-files">
                      {libraryFiles.real.map(f => (
                        <button
                          key={f}
                          className="library-file"
                          disabled={libraryLoading !== null}
                          onClick={() => analyzeTestFile('real', f)}
                        >
                          <FileAudio size={14} />
                          <span className="library-file-name">{f}</span>
                          {libraryLoading === `real/${f}` && <Loader2 size={14} className="spin" />}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="library-col">
                    <h3 className="library-col-title fraud-text">
                      <ShieldX size={16} /> {t.testLibrary.fake} ({libraryFiles.fake.length})
                    </h3>
                    <div className="library-files">
                      {libraryFiles.fake.map(f => (
                        <button
                          key={f}
                          className="library-file"
                          disabled={libraryLoading !== null}
                          onClick={() => analyzeTestFile('fake', f)}
                        >
                          <FileAudio size={14} />
                          <span className="library-file-name">{f}</span>
                          {libraryLoading === `fake/${f}` && <Loader2 size={14} className="spin" />}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="error-toast" onClick={() => setError(null)}>
          <ShieldAlert size={16} style={{ flexShrink: 0 }} /> {error}
        </div>
      )}
    </div>
  );
};
