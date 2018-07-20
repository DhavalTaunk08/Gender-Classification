import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.preprocessing import StandardScaler
import librosa

model = pickle.load(open('Audio_model.sav', 'rb'))

data = pd.read_csv('Dataset_audio.csv')

x = np.array(data[[str(i) for i in range(1,205)]])
y = np.array(data['205'])

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

for file in glob.glob("try/*"):
	y, sr = librosa.load(file, sr=8000)

	mfcc = librosa.feature.mfcc(y=y,sr=sr, n_mfcc=13, n_fft=160, hop_length=80)

	delta = librosa.feature.delta(mfcc)

	delta2 = librosa.feature.delta(mfcc, order=2)

	flatness = librosa.feature.spectral_flatness(y=y, n_fft=160, hop_length=80)

	cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=160, hop_length=80)

	flux = librosa.onset.onset_strength(y=y, sr=sr, n_fft=160, hop_length=80)

	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=160, hop_length=80)

	z = librosa.zero_crossings(y=y, axis=0)

	pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, hop_length=80, n_fft=160)

	energy = librosa.feature.rmse(y=y, frame_length=160, hop_length=80)

	for i in range(0, (len(y)//80)):
		x = []
		if energy[0][i] >= 0.002:
			for j in range(0, len(mfcc)):
				x.append(mfcc[j][i])

			for k in range(0, len(delta)):
				x.append(delta[k][i])

			for l in range(0, len(delta2)):
				x.append(delta2[l][i])

			x.append(flatness[0][i])

			x.append(cent[0][i])

			x.append(flux[i])

			x.append(rolloff[0][i])

			p = i*80
			for m in range(p, p+160):
				x.append(z[m])

			b=0
			for k in range(0,len(pitches)):
				if pitches[k][i] > 0.0:
					b+=1
					if b == 1:
						x.append(pitches[k][i])
						break

			for n in range(len(x)):
				x[n]=float(x[n])

			a = np.array([x]).reshape(1,-1)
			print(np.shape(a))
			b = scaler.transform(a)
			result = model.predict_proba(a)
			print("frame", i, "=", result)