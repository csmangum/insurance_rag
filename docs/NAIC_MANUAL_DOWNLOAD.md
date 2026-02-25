# Manually Downloading NAIC Auto Insurance Documents

NAIC (content.naic.org) often returns **403 Forbidden** for scripted requests. You can download the PDFs in a browser and place them in the project so the pipeline can use them.

## 1. Download in a Browser

### Regulations (model laws and state adoption charts)

1. Open **https://content.naic.org/model-laws**
2. Use the table to find each model and open its "MO-xxx" or chart link
3. Use the browser’s “Save as” / download for each PDF

**Documents to get (auto insurance–related):**

| File to save as | Description |
|-----------------|-------------|
| `mo-725.pdf` | NAIC Automobile Insurance Declination, Termination and Disclosure Model Act |
| `mo-720.pdf` | Property Insurance Declination Termination and Disclosure Model Act |
| `mo-777.pdf` | Property and Casualty Commercial Rate and Policy Form Model Law |
| `mo-745.pdf` | Property and Casualty Actuarial Opinion Model Law |
| `mo-751.pdf` | Model Regulation – P&C Statistical Data Reporting |
| `mo-710.pdf` | Mass Marketing of Property and Liability Insurance Model Regulation |
| `mo-680.pdf` | Insurance Fraud Prevention Model Act |
| `chart-pa-30.pdf` | Chart: Compulsory Motor Vehicle Insurance |
| `chart-pa-10.pdf` | Chart: Rate Filing Methods for P&C |
| `chart-pa-15.pdf` | Chart: Form Filing Methods for P&C |

### Consumer guide (forms)

- Open: **https://content.naic.org/sites/default/files/publication-aut-pp-consumer-auto.pdf**
- Download the PDF (or find it via the NAIC consumer publications page).

---

## 2. Where to Put the Files

Place the files so the auto domain’s download layout matches what the pipeline expects.

### Regulations

**Directory:** `data/auto/raw/regulations/naic/`

**Filenames (exact):**

- `mo-710.pdf`, `mo-680.pdf`, `mo-720.pdf`, `mo-725.pdf`, `mo-745.pdf`, `mo-751.pdf`, `mo-777.pdf`
- `chart-pa-30.pdf`, `chart-pa-10.pdf`, `chart-pa-15.pdf`

### Consumer guide

**Directory:** `data/auto/raw/forms/naic/`

**Filename:** `publication-aut-pp-consumer-auto.pdf`

---

## 3. Refresh Manifests (optional)

After adding files, you can refresh manifests so they list the new files and hashes:

```bash
.venv/bin/python scripts/download_all.py --domain auto --force
```

This re-writes manifests; NAIC URLs will still 403 and be skipped, but existing files in the directories above will be included in the manifests.

If you skip this step, ingest will still pick up any PDFs present in those directories when you run extract/ingest.
