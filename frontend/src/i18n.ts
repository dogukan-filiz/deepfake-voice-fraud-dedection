export type Locale = 'en' | 'tr';

const translations = {
  en: {
    brand: 'VOXGUARD',
    nav: {
      langSwitch: 'TR',
    },
    tracker: {
      title: 'Authenticity Score Tracker',
      subtitle: 'Monitor voice authenticity scores over time and track flagged calls across all analyses.',
      session: 'Session',
      emptyChart: 'No analyses yet. Upload or record audio to get started.',
      allClean: 'All calls clean this session',
      flagged: (n: number) => `${n} call${n > 1 ? 's' : ''} flagged this session`,
      waiting: 'Waiting for first analysis',
    },
    recording: {
      title: 'Live Recording',
      desc: 'Record a short clip from your microphone for instant analysis.',
      start: 'Start Recording',
      stop: 'Stop & Analyze',
      processing: 'Processing...',
      ready: 'Ready',
      recordingStatus: 'Recording...',
      analyzing: 'Analyzing...',
    },
    upload: {
      title: 'Upload & Analyze',
      desc: 'Drop an audio file or click below to select one for analysis.',
      selectFile: 'Click to select audio file',
      analyze: 'Analyze',
      analyzeLoading: 'Analyzing...',
    },
    summary: {
      title: 'Detection Summary',
      thisSession: 'This Session',
      totalScanned: 'Total Scanned',
      flagged: 'Flagged',
      clean: 'Clean',
    },
    recent: {
      title: 'Recent Analyses',
      seeAll: 'See all',
      collapse: 'Collapse',
      empty: 'No analyses yet',
      callPrefix: 'Call #',
      scorePrefix: 'Score: ',
    },
    result: {
      title: 'Latest Analysis',
      authenticity: 'Authenticity',
      riskLevel: 'Risk Level',
      probabilities: 'p(real) / p(fake)',
      spectralAnomaly: 'Spectral Anomaly',
      chunksWorst: 'Chunks / Worst',
      suspectedDeepfake: 'Suspected Deepfake',
      appearsGenuine: 'Appears Genuine',
    },
    risk: {
      low: 'Clean',
      medium: 'Uncertain',
      high: 'Suspicious',
      critical: 'Fraud',
    },
    system: {
      backendStatus: (s: string) => `Backend ${s}`,
    },
    errors: {
      emptyRecording: 'Recording came back empty. Check your microphone permissions.',
      backendUnreachable: 'Could not reach the backend. Is the server running?',
      analysisFailed: (code: number | string) => `Analysis failed (HTTP ${code})`,
      generic: 'Something went wrong during analysis.',
      micDenied: 'Cannot access the microphone. Check browser permissions.',
      audioContextUnsupported: 'AudioContext not supported. Try Chrome or Edge.',
    },
  },
  tr: {
    brand: 'VOXGUARD',
    nav: {
      langSwitch: 'EN',
    },
    tracker: {
      title: 'Doğruluk Skoru Takibi',
      subtitle: 'Ses analizlerinin doğruluk skorlarını anlık olarak takip edin, şüpheli çağrıları görüntüleyin.',
      session: 'Oturum',
      emptyChart: 'Henüz bir analiz yapılmadı. Ses kaydı yükleyin ya da mikrofon ile kayıt alın.',
      allClean: 'Bu oturumda şüpheli çağrı yok',
      flagged: (n: number) => `Bu oturumda ${n} çağrı şüpheli olarak işaretlendi`,
      waiting: 'İlk analiz bekleniyor',
    },
    recording: {
      title: 'Canlı Kayıt',
      desc: 'Mikrofonunuzdan kısa bir ses kaydı alıp anında analiz sonucunu görün.',
      start: 'Kaydı Başlat',
      stop: 'Durdur ve Analiz Et',
      processing: 'İşleniyor...',
      ready: 'Hazır',
      recordingStatus: 'Kayıt alınıyor...',
      analyzing: 'Analiz ediliyor...',
    },
    upload: {
      title: 'Dosya Yükle ve Analiz Et',
      desc: 'Bilgisayarınızdan bir ses dosyası seçerek analiz başlatın.',
      selectFile: 'Ses dosyası seçmek için tıklayın',
      analyze: 'Analiz Et',
      analyzeLoading: 'Analiz ediliyor...',
    },
    summary: {
      title: 'Tespit Özeti',
      thisSession: 'Bu Oturum',
      totalScanned: 'Toplam',
      flagged: 'Şüpheli',
      clean: 'Temiz',
    },
    recent: {
      title: 'Son Analizler',
      seeAll: 'Tümünü gör',
      collapse: 'Daralt',
      empty: 'Henüz analiz yapılmadı',
      callPrefix: 'Çağrı #',
      scorePrefix: 'Skor: ',
    },
    result: {
      title: 'Son Analiz Sonucu',
      authenticity: 'Doğruluk',
      riskLevel: 'Risk Düzeyi',
      probabilities: 'p(gerçek) / p(sahte)',
      spectralAnomaly: 'Spektral Anomali',
      chunksWorst: 'Parça / En Kötü',
      suspectedDeepfake: 'Muhtemel Deepfake',
      appearsGenuine: 'Gerçek Ses',
    },
    risk: {
      low: 'Temiz',
      medium: 'Belirsiz',
      high: 'Şüpheli',
      critical: 'Sahte',
    },
    system: {
      backendStatus: (s: string) => `Sunucu ${s}`,
    },
    errors: {
      emptyRecording: 'Kayıt boş geldi. Mikrofon izinlerinizi kontrol edin.',
      backendUnreachable: 'Sunucuya bağlanılamadı. Sunucu çalışıyor mu?',
      analysisFailed: (code: number | string) => `Analiz başarısız oldu (HTTP ${code})`,
      generic: 'Analiz sırasında beklenmeyen bir hata oluştu.',
      micDenied: 'Mikrofon erişimi reddedildi. Tarayıcı izinlerini kontrol edin.',
      audioContextUnsupported: 'Bu tarayıcı ses kaydını desteklemiyor. Chrome veya Edge kullanın.',
    },
  },
} as const;

export type Translations = typeof translations.en;

export function getTranslations(locale: Locale): Translations {
  return translations[locale];
}

export function detectLocale(): Locale {
  const stored = localStorage.getItem('voxguard-locale');
  if (stored === 'tr' || stored === 'en') return stored;
  const browserLang = navigator.language.toLowerCase();
  if (browserLang.startsWith('tr')) return 'tr';
  return 'en';
}

export function persistLocale(locale: Locale): void {
  localStorage.setItem('voxguard-locale', locale);
}
