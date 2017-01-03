"""
MyRWA_fishcount_sampledata_sklearnGP.py

Thus

Author: Nathan Sanders
Date: 1/2/2016
Version: 0.1
"""

################################
### Imports
################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


################################
### Settings
################################

max_count = 2000 # Maximum possible count in a 10 minute span
false_flag = 1/100. # false positives per minute
max_duration = 1. # Maximum video duration, in minutes


################################
### Load data
################################

## 2016 data
historic_data = pd.read_excel('herring 2016 data FINALa.xlsx') 
## 2015 data
#historic_data = pd.read_excel('Raw_Data_6.xlsx')

historic_data['startdate'] = historic_data.apply(lambda x: pd.Timestamp(str(x['Date']).split()[0] + ' ' + str(x['Start_Time'])), axis=1)
historic_data = historic_data.sort('startdate')
historic_data['timedelta'] = (historic_data['startdate'] - historic_data['startdate'].iloc[0]).astype(int) / (1e9 * 60 * 10) # 10min intervals

historic_data.index = historic_data['timedelta']

################################
### Define rate model
################################

ftrans = lambda x: x
ftransi = lambda x: x

rate_model = interp1d(historic_data['timedelta'].values, historic_data['Count'].values, kind='linear')

## Generate counts every 10 minutes
X_mod_10 = np.arange(historic_data['timedelta'].min(), historic_data['timedelta'].max(), 1)
y_rep_10 = rate_model(X_mod_10)

## Interpolate to every minute
X_mod_min = np.arange(historic_data['timedelta'].min(), historic_data['timedelta'].max(), 0.1)
y_rep_min = pd.Series(ftransi(y_rep_10), index = X_mod_10).ix[X_mod_min].interpolate(method='linear')

## Cap counts at the max_count
y_rep_min[y_rep_min > max_count/10] = max_count/10

## Generate fish counts every minute
c_rep_min = np.random.poisson(np.abs(y_rep_min.values / 10.))

## Assign random crossing times to each fish
t_rep_min = np.empty(np.sum(c_rep_min))
j = 0
for i in range(len(c_rep_min)):
	if c_rep_min[i] > 0:
		t_rep_min[j:j+c_rep_min[i]] = X_mod_min[i]*10 + np.random.uniform(0, 1, c_rep_min[i])
		j += c_rep_min[i]

## Add in false positives at random times
N_false = round(false_flag * len(c_rep_min))
t_rep_min_fp = np.random.uniform(0, len(c_rep_min), N_false)

## Combine real and false positives
t_rep_comb = np.append(t_rep_min, t_rep_min_fp)
t_rep_comb_key = np.append(np.ones(len(t_rep_min)), np.zeros(len(t_rep_min_fp)))

## Mix in the false positives by sorting
ao = np.argsort(t_rep_comb)
t_rep_comb = t_rep_comb[ao]
t_rep_comb_key = t_rep_comb_key[ao]


################################
### Generate synthetic videos
################################

video_starts = []
video_ends = []
video_counts = []
a = []
## Step through fish
for i in range(len(t_rep_comb)):
	## Not fist video
	if i > 0:
		## Have we surpassed the max duration of the last video?
		if (t_rep_comb[i] - video_starts[-1]) > max_duration:
			## Was the previous video a leftover over an isolated segment?
			if (i>1) and (video_starts[-1] == (video_starts[-2] + max_duration)):
				## leftover - video records until 1min after last fish
				## If it's a leftover, we need to start by inserting a video 1 minute from the last start point to complete the last full segment
				video_ends += [round(video_starts[-1] + max_duration, 2)]
				## Then add the leftover itself
				video_starts += [video_ends[-1]]
				video_counts += [0] # last <1min chunk will always be empty if recording goes one minute from the last fish
				video_ends += [round(t_rep_comb[i-1] + max_duration, 2)]
				print 'leftover', video_ends[-1] - video_starts[-1]
				a += [video_ends[-1] - video_starts[-1]]
				print video_starts[-1], video_starts[-2], t_rep_comb[i-1], video_ends[-1]
			else:
				## full segment
				video_ends += [round(video_starts[-1] + max_duration, 2)]
				#print 'full', video_ends[-1] - video_starts[-1]
			video_starts += [round(t_rep_comb[i], 2)]
			video_counts += [t_rep_comb_key[i]]	
		## Within a video
		else:
			video_counts[-1] += t_rep_comb_key[i]
	## First video
	else:
		video_starts += [round(t_rep_comb[i], 2)]
		video_counts += [t_rep_comb_key[i]]

## End the last video
if video_starts[-1] > (video_starts[-2] + max_duration):
	## leftover - video records until 1min after last fish
	video_ends += [round(t_rep_comb[-1] + max_duration)]
else:
	## full segment
	video_ends += [round(video_starts[-1] + max_duration, 2)]

video_starts = np.array(video_starts)
video_ends = np.array(video_ends)
video_counts = np.array(video_counts)


################################
### Make plot
################################

def interleave(*args):
	c = np.empty(sum([len(a) for a in args]), dtype=a.dtype)
	for i,arg in enumerate(args):
		c[i::len(args)] = arg
	return c

plt.figure(figsize=(10,6))
plt.plot(X_mod_min * 10, c_rep_min, lw=2, label='Simulated fish counts per minute', marker='o', mec='none', color='.5', drawstyle='steps-mid')
plt.plot(interleave(video_starts, video_ends, video_ends), 
	 interleave(np.zeros(len(video_starts)), video_counts, np.zeros(len(video_starts))),
	 lw = 1, label='Counts per video', ls='dashed', color='g')
plt.legend(prop={'size':8})
plt.xlabel('Time in season (minutes)')
plt.ylabel('Fish counts')
plt.savefig('fish_counts_minute_sim.png', dpi=150)

plt.axis([24508, 24590, 0, 4.3])
plt.savefig('fish_counts_minute_sim_zoom.png', dpi=150)


################################
### Write out
################################

video_df = pd.DataFrame({
	'Start Time (min)':np.round(video_starts, 2),
	'End Time (min)':np.round(video_ends, 2),
	'True Fish Count':np.round(video_counts, 2),
	'Duration (s)':np.round((video_ends - video_starts) * 60, 0),
	})

## Print out video metadata
video_df[['Start Time (min)', 'End Time (min)', 'Duration (s)', 'True Fish Count']].to_csv('simulated_fish_videos.csv', index=0)

## Print out individual fish crossing times
pd.Series(np.round(t_rep_min, 2)).to_csv('simulated_fish_crossings.csv', index=0)






