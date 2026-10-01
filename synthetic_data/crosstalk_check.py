"""Cross-talk check for the synthetic MAP reconstructions (Sec. III-A3).

Run from Bayes_mc_CT/synthetic_data. Needs numpy and h5py only.

For the delta-only inclusion P (label 5) and the eps-dominant inclusion F
(label 4), the spurious contrast in the channels where the inclusion should
be (nearly) invisible is the reconstructed region-mean contrast against a
ring of surrounding disc pixels, minus the true contrast, in percent of the
disc value. A boundary-band RMSE (4-pixel band across the inclusion edge)
is printed as a second measure. TV and JTV are compared per condition.
"""
import h5py
import numpy as np

lab = h5py.File("obs/synthetic_data.h5", "r")["labels"][...]


def dilate(m, k=1):
    out = m.copy()
    for _ in range(k):
        o = out.copy()
        o[1:, :] |= out[:-1, :]
        o[:-1, :] |= out[1:, :]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def erode(m, k=1):
    return ~dilate(~m, k)


for s in ["full", "sparse_angle", "sparse_step", "combined"]:
    d = np.load(f"obs/scenarios/multicontrast/{s}/recon_arrays.npz")
    print(s)
    for region, code, chans in [("P", 5, ["mu", "eps"]), ("F", 4, ["mu", "delta"])]:
        m = lab == code
        band = dilate(m, 2) & ~erode(m, 2)
        ring = dilate(m, 6) & ~dilate(m, 3) & (lab == 1)
        for ch in chans:
            t = d[f"{ch}_true"]
            body = t[ring].mean()
            c_true = (t[m].mean() - t[ring].mean()) / body
            line = f"  {region} in {ch:5s}:"
            for p in ["TV", "JTV"]:
                r = d[f"{p}_{ch}"]
                c_rec = (r[m].mean() - r[ring].mean()) / body
                rmse = np.sqrt(np.mean((r[band] - t[band]) ** 2)) / body
                line += f"  {p}: spurious {100 * (c_rec - c_true):+.2f}%  band RMSE {100 * rmse:.2f}%"
            print(line)
