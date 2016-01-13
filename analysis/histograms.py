#!/usr/bin/env python

import itertools
import numpy as np
import operator

from matplotlib import pyplot as pl

CONDITIONS = (
    ('follow', [
        'GabeFollow1',
        'JM_1_NoNoise_HiFollow_LoSpeed',
        'WR_060911_C1_HF_LS_NN',
        'SL_1_071511_NoNoise_HiF_LoS',
        'SL_2_071511_NoNoise_HiF_LoS',
        'MS_071811_1_NoNoise_HiF_LoS',
        'MS_071811_2_NoNoise_HiF_LoS',
        ]),
    ('follow+noise', [
        'GabeFollow+Noise3',
        'JL_070111_1_Noise_HiFollow_LoSpeed',
        'IR_1_071411_Noise_HiF_LoS',
        'IR_2_071411_Noise_HiF_LoS',
        'EM_1_071511_Noise_HiF_LoS',
        'EM_2_071511_Noise_HiF_LoS',
        'CV_1_071811_Noise_HiF_LoS',
        'CV_2_071811_Noise_HiF_LoS',
        'DR_102711_1_NewNoiseHiF',
        'DR_102711_2_NewNoiseHiF',
        'ET_102711_1_NewNoiseHiF',
        'ET_102711_2_NewNoiseHiF',
        'AA_102811_1_NewNoiseHiF',
        'AA_102811_2_NewNoiseHiF',
        'AK_102811_1_NewNoiseHiF',
        'AK_102811_2_NewNoiseHiF',
        'HW_102811_1_NewNoiseHiF',
        'HW_102811_2_NewNoiseHiF',
        'JA_102711_1_NewNoiseHiF',
        'JA_102711_2_NewNoiseHiF',
        'JM_102811_1_NewNoiseHiF',
        'JM_102811_2_NewNoiseHiF',
        'JO_102811_1_NewNoiseHiF',
        'JO_102811_2_NewNoiseHiF',
        'NM_102711_1_NewNoiseHiF',
        'NM_102711_2_NewNoiseHiF',
        'WJ_102711_1_NewNoiseHiF',
        'WJ_102711_2_NewNoiseHiF',
        ]),
    ('speed', [
        'GabeSpeed2',
        'RK_1_HiS_LoF_NoN_070711',
        'RK_2_HiS_LoF_NoN_070711',
        'KV_072211_1_HiS_LoF',
        'KV_072211_2_HiS_LoF',
        'MC_082311_1_NoNoise_HiSLoF',
        'VV_082511_1_NoNoise_HiSLoF',
        'VV_082511_2_NoNoise_HiSLoF',
        'AD_091311_1_HiSpeed_NoNoise',
        'AD_091311_2_HiSpeed_NoNoise',
        'ST_091511_1_NoNoise_HiSLoF',
        'ST_091511_2_NoNoise_HiSLoF',
        'SW_091511_1_HiSpeed_NoNoise',
        'SW_091511_2_HiSpeed_NoNoise',
        'AW_092011_1_HiSpeed_NoNoise',
        'AW_092011_2_HiSpeed_NoNoise',
        'BM_092011_1_HiSpeed_NoNoise',
        'BM_092011_2_HiSpeed_NoNoise',
        'CL_092011_1_HiSpeed_NoNoise',
        'CL_092011_2_HiSpeed_NoNoise',
        'AW_092211_1_HiSpeed_NoNoise',
        'AW_092211_2_HiSpeed_NoNoise',
        'KD_092211_1_HiSpeed_NoNoise',
        'KD_092211_2_HiSpeed_NoNoise',
        'KM_092011_1_HiSpeed_NoNoise',
        'KM_092011_2_HiSpeed_NoNoise',
        'SL_092211_1_HiSpeed_NoNoise',
        'SL_092211_2_HiSpeed_NoNoise',
        ]),
    ('speed+noise', [
        'GabeSpeed+Noise4',
        'Isha_1_Noise_HiS_LoF',
        'Isha_2_Noise_HiS_LoF',
        'JM_2_Noise_HiSpeed_LoFollow',
        'MJ_1_Noise_HiSpeed_LoFollow',
        'MJ_2_Noise_HiSpeed_LoFollow',
        'AD_062111_WithNoise_HiSpeed_LoLeader',
        'GC_1_072011_Noise_HiS_LoF',
        #'CB_1_072111_Noise_HiS_LoF', # file is empty
        #'CB_2_072111_Noise_HiS_LoF', # file is empty
        'GM_081611_1_HiSpeed_Noise',
        'GM_081611_2_HiSpeed_Noise',
        'EL_102111_1_NewNoise_HiS',
        'EL_102111_2_NewNoise_HiS',
        'IN_102111_1_NewNoise_HiS',
        'IN_102111_2_NewNoise_HiS',
        'MA_102111_1_NewNoise_HiS',
        'MA_102111_2_NewNoise_HiS',
        'ND_102111_1_NewNoise_HiS',
        'ND_102111_2_NewNoise_HiS',
        'PN_102111_1_NewNoise_HiS',
        'PN_102111_2_NewNoise_HiS',
        'JH_102411_1_NewNoise_HiS',
        'JH_102411_2_NewNoise_HiS',
        'MY_102411_1_NewNoise_HiS',
        'MY_102411_2_NewNoise_HiS',
        'SM_102411_1_NewNoise_HiS',
        'SM_102411_2_NewNoise_HiS',
        'IS_102511_1_NewNoise_HiS',
        'IS_102511_2_NewNoise_HiS',
        'FO_102511_1_NewNoise_HiS',
        'FO_102511_2_NewNoise_HiS',
        'MB_102511_1_NewNoise_HiS',
        ]))


def counts(filename):
    counts = [[], []]
    print 'counting frames in', filename
    d = np.loadtxt('xml/%s.txt.gz' % filename)
    for m, fs in itertools.groupby(d, key=operator.itemgetter(5)):
        L = counts[int(m)]
        start = prev = None
        for f in fs:
            if start is None:
                start = prev = f[0]
            if f[0] - prev > 10 and start > 600:
                d = (f[0] - start) / 60.
                if d > 0.1:
                    L.append(d)
                start = prev = None
            else:
                prev = f[0]
        if start is not None:
            d = (f[0] - start) / 60.
            if d > 0.1:
                L.append(d)
    return (np.asarray(c) for c in counts)


def build_histograms():
    for label, files in CONDITIONS:
        print label, len(files)
        yield label, (counts(f) for f in files)


def main():
    kw = dict(bins=np.linspace(0, 5, 11), align='mid', rwidth=0.6, alpha=0.7)
    totals = {}
    for label, hists in build_histograms():
        Speed = []
        Follow = []
        for i, (speed, follow) in enumerate(hists):
            ax = pl.subplot(7, 5, i + 1)
            if len(speed):
                Speed.append(speed)
                _, _, rs = ax.hist(speed, **kw)
                [r.set_x(r.get_x() - 0.06) for r in rs]
            if len(follow):
                Follow.append(follow)
                _, _, rs = ax.hist(follow, **kw)
                [r.set_x(r.get_x() + 0.06) for r in rs]
            ax.set_xlim(0, 5)
            ax.set_xticks([])
            ax.set_ylim(0, 30)
            ax.set_yticks([])
            if i == 29:
                break
        pl.tight_layout()
        pl.savefig('/tmp/histogram-%s.pdf' % label)
        pl.clf()
        def w(z): return np.asarray(list(itertools.chain(*z)))
        totals[label] = w(Speed), w(Follow)

    for i, (label, _) in enumerate(CONDITIONS):
        speed, follow = totals[label]
        ax = pl.subplot(2, 2, i + 1)
        _, _, rs = ax.hist(speed, **kw)
        [r.set_x(r.get_x() - 0.06) for r in rs]
        _, _, rs = ax.hist(follow, **kw)
        [r.set_x(r.get_x() + 0.06) for r in rs]
        ax.set_xlim(0, 5)
        ax.set_xticks([])
        #ax.set_ylim(0, 100)
        #ax.set_yticks([])
        ax.set_title(label)
    pl.tight_layout()
    pl.savefig('/tmp/histograms.pdf')


if __name__ == '__main__':
    main()
