ASVspoof 2019 Database - Logical Access (LA) Subset
====================================================
Source  : Edinburgh DataShare — https://datashare.ed.ac.uk/handle/10283/3336
License : Creative Commons Attribution 4.0 International (CC-BY 4.0)
Paper   : Todisco et al., ASVspoof 2019: Future Horizons in Spoofed and Fake
          Audio Detection, Interspeech 2019.

Directory layout
----------------
ASVspoof2019_LA/
  cm_protocols/
    ASVspoof2019.LA.cm.eval.trl.txt   -- evaluation trial list
    ASVspoof2019.LA.cm.train.trn.txt  -- training trial list
  real/   LA_E_1000001.wav … bonafide (genuine) utterances, eval partition
  fake/   LA_E_2000001.wav … spoofed utterances, eval partition

Protocol columns: SPEAKER_ID  FILENAME  SYSTEM_ID  KEY  SUBSET
  SYSTEM_ID '-' = bonafide; A07–A19 = TTS/VC spoofing systems
  KEY: 'bonafide' | 'spoof'

This subset (17 bonafide + 17 spoof) is drawn from the evaluation partition.
