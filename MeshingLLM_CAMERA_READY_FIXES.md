# Camera-Ready Corrections — MeshingLLM (ICEBE 2026)

**Status:** one numeric correction required before submission.
**Cause:** the index-code ablation (Fig. 5 / finding F4) was re-measured on the
**real** Qwen2.5-1.5B weights (bf16) with the corrected reader and a
delta + LEB128 varint index code. Only the two *endpoints* move; the dense
bit-map value reproduces exactly.

---

## 1. Abstract — item (iii)  ★ CHANGE

**Location:** last sentence of the Abstract.

| | Text |
|---|---|
| **Current** | ...a gzip-compressed bit-map raises the serialized ratio from 7.87× to **10.61×**. |
| **Change to** | ...a gzip-compressed bit-map raises the serialized ratio from 7.87× to **11.39×**. |

---

## 2. Section V-F, finding F4  ★ CHANGE

| | Text |
|---|---|
| **Current** | ...from **3.94×** (varint) to 9.54× (dense bit-map) to **10.61×** (gzip-compressed bit-map)—a factor of **2.7** gained without touching the model (Fig. 5). |
| **Change to** | ...from **7.87×** (varint) to 9.54× (dense bit-map) to **11.39×** (gzip-compressed bit-map)—a factor of **1.45** gained without touching the model (Fig. 5). |

---

## 3. Figure 5 (IndexCode)  ★ CHANGE

In-figure labels (bar heights and annotations):

| Index code | Current | Change to |
|---|---|---|
| varint | 3.94× | **7.87×** |
| dense bit-map | 9.54× | 9.54× (unchanged) |
| gzip bit-map | 10.61× | **11.39×** |
| per-code index cost `b_idx` | 8.00 / 5.90 / 4.90 bit | ⚠️ **re-check** against the new run (see §6) |

The dense bit-map value (`9.54×`) is reproduced exactly by the new measurement,
which confirms the protocol is unchanged and only the varint/gzip endpoints differ.

---

## 4. Also affected — already fixed in this release

* `README.md` → "Key findings" table: `CR_s` from **3.94x to 10.61x** → **7.87x to 11.39x** ✅ done.

---

## 5. NOT affected — no action needed

* **Table II** (main comparison) — quantisation baselines and `τ = 0.99/0.95/0.85` rows unchanged.
* **Table III** (frequency-selection ablation) — unchanged.
* **Table IV** (downstream performance) — unchanged.
* **§V-C** cliff statement (`1.26× → 1.63×`) — unchanged.
* **F1 / F2 / F3** — unchanged.
* The truncation identity and its machine-precision verification — unchanged.

---

## 6. What is still needed to redraw Fig. 5

The new per-code index cost `b_idx` (bits per **retained** coefficient) from the
latest baseline run. Export it with:

```bash
cd /root/autodl-tmp/MeshingLLM && cat outputs/journal/baselines/baselines_report.md
```

Once provided, Fig. 5 can be regenerated with the corrected numbers and the
updated `b_idx` annotations.

---

## 7. One-line change sheet (for the copy-editor)

```
Abstract  (iii):  10.61×           -> 11.39×
F4        text :  3.94× / 10.61×   -> 7.87× / 11.39×   ;  "factor of 2.7" -> "factor of 1.45"
Fig. 5    bars :  3.94× / 10.61×   -> 7.87× / 11.39×
README    table:  3.94x→10.61x     -> 7.87x→11.39x      (already done in this release)
```
