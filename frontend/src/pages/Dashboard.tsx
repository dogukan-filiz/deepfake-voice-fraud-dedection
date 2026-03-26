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

export const Dashboard: React.FC = () => {
  const [dosya, setDosya] = useState<File | null>(null);
  const [sonuc, setSonuc] = useState<TahminSonucu | null>(null);
  const [cagrilar, setCagrilar] = useState<CagriKaydi[]>([]);
  const [yukleniyor, setYukleniyor] = useState(false);
  const [hata, setHata] = useState<string | null>(null);

  // Canli kayit durumu
  const [kayitDurumu, setKayitDurumu] = useState<'idle' | 'recording' | 'processing'>('idle');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

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

  // ---------------- Canli Ses Kaydi ile Analiz ----------------

  const sesiAnalyzeEtBlob = async (blob: Blob) => {
    setKayitDurumu('processing');
    try {
      const formData = new FormData();
      formData.append('file', blob, 'kayit.webm');
  const res = await api.post<TahminSonucu>('/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSonuc(res.data);
      await cagrilariGetir();
    } catch (e) {
  console.error('Canli kayit analiz hatasi', e);
  setHata('Canli kayit analizinde bir hata olustu');
    } finally {
      setKayitDurumu('idle');
    }
  };

  const baslatKayit = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      let mediaRecorder: MediaRecorder;
      const oggMime = 'audio/ogg; codecs=opus';

      if (typeof MediaRecorder !== 'undefined' && 'isTypeSupported' in MediaRecorder && MediaRecorder.isTypeSupported(oggMime)) {
        mediaRecorder = new MediaRecorder(stream, { mimeType: oggMime });
      } else {
        mediaRecorder = new MediaRecorder(stream);
      }

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || oggMime,
        });
        // Mikrofon stream'ini serbest birak
        stream.getTracks().forEach((t) => t.stop());
        await sesiAnalyzeEtBlob(blob);
      };

      mediaRecorder.start();
      recorderRef.current = mediaRecorder;
      setKayitDurumu('recording');
      setHata(null);
    } catch (e) {
      console.error('Mikrofona erisilemedi', e);
      setHata('Mikrofona erisilemiyor. Tarayici izinlerini kontrol edin.');
    }
  };

  const durdurKayit = () => {
    const recorder = recorderRef.current;
    if (recorder && kayitDurumu === 'recording') {
      recorder.stop();
      setKayitDurumu('processing');
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
      setHata('Analiz sirasinda bir hata olustu');
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
                <strong>Guvenilirlik Skoru:</strong> {sonuc.authenticity_score.toFixed(3)}
              </p>
              <p>
                <strong>Durum:</strong>{' '}
                {sonuc.is_suspected_fraud ? 'Süpheli (Olası Fraud)' : 'Normal'}
              </p>
              <p>
                <strong>p_real:</strong> {sonuc.p_real.toFixed(3)} | <strong>p_fake:</strong>{' '}
                {sonuc.p_fake.toFixed(3)}
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
