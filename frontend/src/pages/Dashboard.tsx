import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface CagriKaydi {
  cagri_id: string;
  authenticity_score: number;
  is_suspected_fraud: boolean;
  timestamp: string;
  notlar?: string | null;
}

interface TahminSonucu {
  cagri_id: string;
  authenticity_score: number;
  is_suspected_fraud: boolean;
  p_real: number;
  p_fake: number;
  spectral_residual: number;
  timestamp: string;
}

interface HealthResponse {
  status: string;
  ffmpeg_available?: boolean;
  ffmpeg_exe?: string | null;
  pid?: number;
  module_file?: string;
}

export const Dashboard: React.FC = () => {
  const [dosya, setDosya] = useState<File | null>(null);
  const [sonuc, setSonuc] = useState<TahminSonucu | null>(null);
  const [cagrilar, setCagrilar] = useState<CagriKaydi[]>([]);
  const [yukleniyor, setYukleniyor] = useState(false);
  const [hata, setHata] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [recordMime, setRecordMime] = useState<string | null>(null);
  const [lastBlobSize, setLastBlobSize] = useState<number | null>(null);

  // Canli kayit durumu
  const [kayitDurumu, setKayitDurumu] = useState<'idle' | 'recording' | 'processing'>('idle');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  // WAV fallback kayit (MediaRecorder audio/wav desteklemeyen tarayicilar icin)
  const recordingModeRef = useRef<'mediarecorder' | 'pcm' | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pcmSampleRateRef = useRef<number | null>(null);

  const api = axios.create({ baseURL: '/api' });

  const cagrilariGetir = async () => {
    try {
      const res = await api.get<CagriKaydi[]>('/calls');
      setCagrilar(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    cagrilariGetir();
    const interval = setInterval(cagrilariGetir, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const refresh = async () => {
      try {
        const res = await api.get<HealthResponse>('/health');
        setHealth(res.data);
      } catch (e) {
        console.error('Health check failed', e);
        setHealth(null);
      }
    };
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, []);

  // ---------------- Canli Ses Kaydi ile Analiz ----------------

  const mergeFloat32 = (chunks: Float32Array[]) => {
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    return merged;
  };

  const encodeWav16BitMono = (samples: Float32Array, sampleRate: number) => {
    // 16-bit PCM mono WAV
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeAscii = (offset: number, text: string) => {
      for (let i = 0; i < text.length; i++) {
        view.setUint8(offset + i, text.charCodeAt(i));
      }
    };

    writeAscii(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM header size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    writeAscii(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const sesiAnalyzeEtBlob = async (blob: Blob) => {
    setKayitDurumu('processing');
    try {
      setLastBlobSize(blob.size);
      if (!blob.size) {
        setHata('Kayit bos geldi (0 byte). Mikrofon izni ve tarayici destegini kontrol edin.');
        return;
      }

      const mime = (blob.type || '').toLowerCase();
      const filename = mime.includes('ogg')
        ? 'kayit.ogg'
        : mime.includes('webm')
          ? 'kayit.webm'
          : mime.includes('wav')
            ? 'kayit.wav'
          : mime.includes('mp4') || mime.includes('m4a')
            ? 'kayit.m4a'
            : 'kayit.bin';

      const formData = new FormData();
      formData.append('file', blob, filename);
      const res = await api.post<TahminSonucu>('/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSonuc(res.data);
      await cagrilariGetir();
    } catch (e) {
      console.error('Canli kayit analiz hatasi', e);
      if (axios.isAxiosError(e)) {
        const data: any = e.response?.data;
        const detail = (data && (data.detail ?? data.message)) as string | undefined;
        if (!e.response) {
          setHata(
            detail ??
              'Backend baglantisi kurulamadi (proxy/port). Backend calisiyor mu ve /api/health aciliyor mu kontrol edin.',
          );
        } else {
          setHata(detail ?? `Canli kayit analizinde hata (HTTP ${e.response?.status ?? '??'})`);
        }
      } else {
        setHata('Canli kayit analizinde bir hata olustu');
      }
    } finally {
      setKayitDurumu('idle');
    }
  };

  const baslatKayit = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const canChoose = typeof MediaRecorder !== 'undefined' && 'isTypeSupported' in MediaRecorder;
      const wavSupported = canChoose ? MediaRecorder.isTypeSupported('audio/wav') : false;

      // 1) Ideal case: MediaRecorder can output WAV -> no FFmpeg needed.
      if (typeof MediaRecorder !== 'undefined' && wavSupported) {
        const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/wav' });
        recordingModeRef.current = 'mediarecorder';
        recorderRef.current = mediaRecorder;

        setRecordMime(mediaRecorder.mimeType || 'audio/wav');
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const blob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'audio/wav' });
          setRecordMime(mediaRecorder.mimeType || 'audio/wav');
          // Mikrofon stream'ini serbest birak
          stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          await sesiAnalyzeEtBlob(blob);
        };

        mediaRecorder.start();
        setKayitDurumu('recording');
        setHata(null);
        return;
      }

      // 2) Fallback: capture PCM via WebAudio and encode WAV client-side.
      // This avoids requiring FFmpeg for live recording.
      recordingModeRef.current = 'pcm';
      recorderRef.current = null;
      audioChunksRef.current = [];
      pcmChunksRef.current = [];

      const AudioCtx = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext | undefined;
      if (!AudioCtx) {
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        setHata('Bu tarayicida AudioContext desteklenmiyor. Chrome/Edge ile deneyin.');
        return;
      }

      const ctx = new AudioCtx();
      audioContextRef.current = ctx;
      pcmSampleRateRef.current = ctx.sampleRate;

      const source = ctx.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      // ScriptProcessor is widely supported and sufficient for short recordings.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorNodeRef.current = processor;

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        // Copy to detach from underlying buffer.
        pcmChunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      setRecordMime('audio/wav');
      setKayitDurumu('recording');
      setHata(null);
    } catch (e) {
      console.error('Mikrofona erisilemedi', e);
      setHata('Mikrofona erisilemiyor. Tarayici izinlerini kontrol edin.');
    }
  };

  const durdurKayit = () => {
    if (kayitDurumu !== 'recording') return;

    if (recordingModeRef.current === 'mediarecorder') {
      const recorder = recorderRef.current;
      if (recorder) {
        recorder.stop();
        setKayitDurumu('processing');
      }
      return;
    }

    if (recordingModeRef.current === 'pcm') {
      setKayitDurumu('processing');

      const stream = streamRef.current;
      const ctx = audioContextRef.current;
      const source = sourceNodeRef.current;
      const processor = processorNodeRef.current;
      const sampleRate = pcmSampleRateRef.current ?? 48000;
      const chunks = pcmChunksRef.current;

      // Cleanup nodes + stream
      try {
        processor?.disconnect();
        source?.disconnect();
      } catch {
        // ignore
      }
      processorNodeRef.current = null;
      sourceNodeRef.current = null;

      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      streamRef.current = null;

      if (ctx) {
        ctx.close().catch(() => null);
      }
      audioContextRef.current = null;

      const samples = mergeFloat32(chunks);
      pcmChunksRef.current = [];
      const wavBlob = encodeWav16BitMono(samples, sampleRate);
      void sesiAnalyzeEtBlob(wavBlob);
      return;
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDosya(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!dosya) return;
    setYukleniyor(true);
    setHata(null);
    try {
      const formData = new FormData();
      formData.append('file', dosya);
      const res = await api.post<TahminSonucu>('/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSonuc(res.data);
      await cagrilariGetir();
    } catch (e: any) {
      console.error(e);
      if (axios.isAxiosError(e)) {
        const data: any = e.response?.data;
        const detail = (data && (data.detail ?? data.message)) as string | undefined;
        setHata(detail ?? `Analiz sirasinda hata (HTTP ${e.response?.status ?? '??'})`);
      } else {
        setHata('Analiz sirasinda bir hata olustu');
      }
    } finally {
      setYukleniyor(false);
    }
  };

  const riskVerisi = [...cagrilar]
    .slice()
    .reverse()
    .map((c) => ({
      time: new Date(c.timestamp).toLocaleTimeString(),
      score: c.authenticity_score,
    }));

  return (
    <div className="app">
      <header>
        <h1>Deepfake Ses Tespit Paneli</h1>
      </header>

      <main>
        <section className="panel">
          <h2>Canli Ses Kaydi ile Analiz</h2>
          <p>Mikrofonunuzdan kisa bir kayit alarak aninda analiz ettirebilirsiniz.</p>
          {health && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', opacity: 0.85 }}>
              Backend: <strong>{health.status}</strong>
              {typeof health.ffmpeg_available === 'boolean' && (
                <>
                  {' '}| FFmpeg: <strong>{health.ffmpeg_available ? 'var' : 'yok'}</strong>
                </>
              )}
              {typeof health.pid === 'number' && (
                <>
                  {' '}| PID: <strong>{health.pid}</strong>
                </>
              )}
            </p>
          )}
          {(recordMime || typeof lastBlobSize === 'number') && (
            <p style={{ marginTop: '0.25rem', fontSize: '0.85rem', opacity: 0.75 }}>
              Kayit MIME: <strong>{recordMime ?? 'bilinmiyor'}</strong>
              {typeof lastBlobSize === 'number' && (
                <>
                  {' '}| Boyut: <strong>{lastBlobSize}</strong> byte
                </>
              )}
            </p>
          )}
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginTop: '0.5rem' }}>
            {kayitDurumu === 'recording' ? (
              <button onClick={durdurKayit}>Kaydi Durdur ve Analiz Et</button>
            ) : (
              <button onClick={baslatKayit} disabled={kayitDurumu === 'processing'}>
                {kayitDurumu === 'processing' ? 'Isleniyor...' : 'Kaydi Baslat'}
              </button>
            )}
            <span>
              Durum:{' '}
              <strong>
                {kayitDurumu === 'idle' && 'Beklemede'}
                {kayitDurumu === 'recording' && 'Kayit Yapiliyor...'}
                {kayitDurumu === 'processing' && 'Analiz Ediliyor...'}
              </strong>
            </span>
          </div>
          {hata && <p className="error">{hata}</p>}
        </section>

        <section className="panel">
          <h2>Ses Yukle ve Analiz Et</h2>
          <input type="file" accept="audio/*" onChange={handleFileChange} />
          <button onClick={handleUpload} disabled={!dosya || yukleniyor}>
            {yukleniyor ? 'Analiz ediliyor...' : 'Analiz Et'}
          </button>
          {hata && <p className="error">{hata}</p>}

          {sonuc && (
            <div className="card">
              <h3>Son Analiz Sonucu</h3>
              <p>
                <strong>Cagri ID:</strong> {sonuc.cagri_id}
              </p>
              <p>
                <strong>Guvenilirlik Skoru:</strong> {sonuc.authenticity_score.toFixed(6)}
              </p>
              <p>
                <strong>Durum:</strong>{' '}
                {sonuc.is_suspected_fraud ? 'Süpheli (Olası Fraud)' : 'Normal'}
              </p>
              <p>
                <strong>p_real:</strong> {sonuc.p_real.toFixed(6)} | <strong>p_fake:</strong>{' '}
                {sonuc.p_fake.toFixed(6)}
              </p>
              <p>
                <strong>Spectral Residual:</strong> {sonuc.spectral_residual.toFixed(3)}
              </p>
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Cagri Gecmisi / Risk Skorlari</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={riskVerisi}>
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="score" stroke="#8884d8" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <table className="table">
            <thead>
              <tr>
                <th>Cagri ID</th>
                <th>Skor</th>
                <th>Durum</th>
                <th>Zaman</th>
              </tr>
            </thead>
            <tbody>
              {cagrilar.map((c) => (
                <tr key={c.cagri_id} className={c.is_suspected_fraud ? 'risk' : ''}>
                  <td>{c.cagri_id}</td>
                  <td>{c.authenticity_score.toFixed(3)}</td>
                  <td>{c.is_suspected_fraud ? 'Süpheli' : 'Normal'}</td>
                  <td>{new Date(c.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
};
